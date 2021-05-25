#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 24th 2020
2d-to-3d
@author: Kaleb Vinehout

This code combines 2D data into 3d volume

Input: 2p blood vessel imaging plane
        reconstructed 3D barseq data

Output: Barseq data that matches A/C 2P imaging plane in size and orientation

"""

import warnings
import ants
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.registration
import skimage.segmentation
import skimage.transform
import skimage.io as skio


def parse_args(add_help=True):
    # this adds help to main function
    parser = argparse.ArgumentParser(description='3d to 3D registration', add_help=add_help)

    # make parser function for boolean
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # these are required inputs arguments
    parser.add_argument("--remotesubjectpath", required=True, type=str,
                        help="This is full path where to files are located eaither locally or on remote server (ex:'/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/'")

    parser.add_argument("--file_path_2p", required=True, type=str,
                        help="This is full path where to files are located eaither locally or on remote server (ex:'/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/'")
    parser.add_argument("--file_path_barseq", required=True, type=str,
                        help="This is full path where to files are located eaither locally or on remote server (ex:'/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/'")

    # for 2P: seg_smooth=1, checkerboard_size=3, seg_interations=10

    # this adds optional input arguments
    parser.add_argument("--input_overlap", default=None, type=int,
                        help="# number of PIXELS to overlap, if not defined then just use calculated value")
    # this adds optional input arguments
    parser.add_argument("--linear_or_nonlinear", default=False, type=str2bool,
                        help="This is set to True for non-linear registration, otherwise linear registration perfromed")
    return parser


def load_data(file_path):
    file_type = file_path.split(".", 1)[1]
    # if file path is numpy array
    if (file_type == 'tiff') or (file_type == 'tif'):
        # if file_path is tiff file
        data = skio.imread(file_path, plugin="tifffile")
    elif (file_type == 'npy') or (file_type == 'npz'):
        data = np.load(file_path)
    else:
        warnings.warn(message="WARNING: file type {} not tiff or numpy arrary in {}".format(file_type, file_path))
        data = None
    return data


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list. THIS IS USED FOR SEGMENTATION
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def segment_data(A, seg_smooth, checkerboard_size, seg_interations):
    """
    Denoise the BARSEQ 3d or 2P  data and segments data
        Args:
        -	A: Numpy array 3D registered image

        Returns:
        -	A_seg: Numpy array 3D registered image deionised and segmented

    """

    # for 2P: seg_smooth=1, checkerboard_size=3, seg_interations=10

    # todo edit this so done in 3D not 2D

    # Initial level set
    init_ls = skimage.segmentation.checkerboard_level_set(A.shape, checkerboard_size)  # default is 5
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    # smoothing of 3 works
    A_seg = skimage.segmentation.morphological_chan_vese(A, seg_interations, init_level_set=init_ls,
                                                         smoothing=seg_smooth,
                                                         iter_callback=callback)  # here 35 is the number of iterations and smoothing=3 is number of times smoothing applied/iteration

    # A_close=skimage.morphology.closing(A_seg, skimage.morphology.square(20))
    # A_seg = skimage.segmentation.clear_border(A_seg)  # this removes segmentation on edge
    # binarize imaage
    thresh = skimage.filters.threshold_otsu(A_seg)
    A_bin = A_seg > thresh

    # check to make sure background =0 and segment brain is =1
    if np.count_nonzero(A_bin == 1) > np.count_nonzero(A_bin == 0):
        # inverse image
        A_bin = 1 - A_bin

    return A_seg


def denoise_2p(volume, FFT_max_gaussian):
    """
    Visualizes the 3D registared brain
        Args:
        -	volume: Numpy array 3D registered image

        Returns:
        -	volume_seg: Numpy array 3D registered image deionised and segmented

    """
    # here we want image C
    sigma_est_ori = np.mean(skimage.restoration.estimate_sigma(volume, multichannel=True))
    print(f"estimated noise standard deviation = {sigma_est_ori}")
    # use band pass filtering to remove shadow
    filt_A = skimage.filters.difference_of_gaussians(volume, 1, FFT_max_gaussian)
    # denoise with wavlet
    denoise_wave = skimage.restoration.denoise_wavelet(filt_A, sigma=None, wavelet='haar', mode='soft',
                                                       wavelet_levels=None, multichannel=False, convert2ycbcr=False,
                                                       method='BayesShrink', rescale_sigma=True)
    # Non - localmeans
    sigma_est = np.mean(skimage.restoration.estimate_sigma(denoise_wave, multichannel=True))
    patch_kw = dict(patch_size=5, patch_distance=6, multichannel=True)
    denoise_nl_mean = skimage.restoration.denoise_nl_means(denoise_wave, h=1.15 * sigma_est, fast_mode=False,
                                                           preserve_range=True, **patch_kw)
    # TV norm
    denoise = skimage.restoration.denoise_tv_chambolle(denoise_nl_mean, weight=0.1, eps=0.0002, n_iter_max=200,
                                                       multichannel=False)
    # differance of gaussian again
    filt_B = skimage.filters.difference_of_gaussians(denoise, 1, FFT_max_gaussian)
    denoised = filt_B
    # todo come up with stepps here
    # >> > import ants
    # >> > img = ants.image_read(ants.get_ants_data('r16'))
    # >> > img = ants.resample_image(img, (64, 64), 1, 0)
    # >> > mask = ants.get_mask(img)
    # >> > ants.atropos(a=img, m='[0.2,1x1]', c='[2,0]', i='kmeans[3]', x=mask)
    # volume_seg = volume
    # denoiseA = ants.denoise_image(A_ant)
    return denoised


def reg_2p_barseq(data2p, databarseq):
    """
    Visualizes the 3D registared brain
        Args:
        -	volume: Numpy array 3D registered image

        Returns:
        -	volume_seg: Numpy array 3D registered image deionised and segmented
    """

    # https://github.com/slicereg

    # step 1 find the part in whole registration b/c C 2p data is only PART of the barseq data or use 2p whole image to initalize?

    # feature matching registration????

    # STEP 2 register these two parts together (fine tune above? --> is this needed?)

    # cv2.matchTemplate()
    # we want affine or rigid. do NOT use non-linear. This is because of perspective issue with 2P data
    # actually perspective not much of an issue... so lets have linear and non-linear option and ignore perspective issue
    # denoise with feature matching?
    # here we want 3D.... maybe use o3d software here??? for ridgid?
    # or try: multiresolution matching
    # convert to ants data fromat
    bar_ant = ants.from_numpy(data=bar_seg)
    p2_ant = ants.from_numpy(data=p2_seg)
    # register 2 datasets
    mytx = ants.registration(fixed=bar_ant, moving=p2_ant, type_of_transform="Rigid", initial_transform=None,
                             outprefix="test", dimension=2)  # “Similarity”
    # register 2 datasets wiht point matching ?
    mytx1 = ants.registration(fixed=bar_ant, moving=p2_ant, type_of_transform="Rigid", initial_transform=None,
                              outprefix="test", dimension=2)  # “Similarity”
    # apply transfrom to unsegmented data
    p2_reg_ants = ants.apply_transforms(fixed=bar_ant, moving=p2_ant, transformlist=mytx['fwdtransforms'])
    # convert back to numpy
    p2_reg = p2_reg_ants.numpy()
    # save files
    save_feature_name = 'p2_reg'
    np.save(args.localsubjectpath + save_feature_name, p2_reg)
    return reg_2p, reg_barseq


def main(args):
    file_path_bar = args.file_path_barseq  # '/Volumes/Backup5TB/data/d3_array_denoise.npz'
    file_path_2p = args.file_path_2p  # '/Volumes/Backup5TB/data/m25269stacks/25269-2-5.tiff'  # 2-5 is C here we want C here b/c jake said so
    data_bar = load_data(file_path_bar)
    data_all = load_data(file_path_2p)
    # divde data into channels
    data_2p = data_all[1::2, :, :]  # here we want channel 2 data

    # segment Barseq data
    bar_seg = segment_data(data_bar)

    # denoise and segment B2 photon data
    p2_denoise = denoise_2p(data_2p, args.FFT_max_gaussian)
    # for 2P: seg_smooth=1, checkerboard_size=3, seg_interations=10
    p2_seg = segment_data(p2_denoise, args.seg_smooth, args.checkerboard_size, args.seg_interations)

    reg_2p, reg_barseq = reg_2p_barseq(p2_seg, bar_seg)


# run main program with parser
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)

"""
EXTRA:

    def change_prespective(data_2p):
        """
# This warp 2p data into 3d volume. warp barseq data into 2p perspective?
# maybe downsize everythign to farthest away voxel is best approach?
"""

return data_2p_prespective

from PIL import Image
#Channel 1 is green (GCaMP), --> so this is cells,,, not matter for registration.
#
# and channel 2 is red (Texas Red in vessels).
#file_path = '/Volumes/Backup5TB/data/m25269stacks/25269-2-5.tiff' #2-5 is C here

#The main difference between A and C is that the Sindbis - expressing cells are visible in the stack in C.I think
# that would be the best one to focus on for registration since: 1) Its possible to identify the Sindbis cells and
# 2) its taken right before perfusion and is likely to be easier to registerthan A,
# which was taken a few days before, on the day of surgery.There can be non - rigid tissue distortion that happens
# over the intervening days as the swelling from the surgery goes down and the tissue  adapts to the cranial window.
# Thus the closer in time the in vivo stack is to the in vitro tissue, the better the result is likely to be.


#The only possibility for true "ground truth" would be for stacks or scans recorded on the same day, since otherwise
# there is no guarantee the mouse is placed on the microscope at exactly the same position and orientation
# (although wetry to get it close). Even for the same day, we typically just do image-based registration rather
# than trusting the coordinates from the microscope.There can be some slow drift over the five or so hours between
# scan and stack, and we would have to take into account shifts due to movement correction and stack stitching and
# registration.I wouldn't use B for registering anything, it is just to give you a visual overview of the
# field of view. But it's only 10 planes and the spacing is large.


#c and D collected on same day

"""
