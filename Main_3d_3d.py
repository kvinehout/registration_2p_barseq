#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 24th 2020
2d-to-3d
@author: Kaleb Vinehout

This code combines 2D data into 3d volume

Input: folder path to files labels with POS names and type of files within this folder to register (ex: 'IL-A’)

Output: np array of 3D image, calculated image overlap,  and video(— or some other file type ?) of rotating brain


"""


import ants
import os
import argparse
import numpy as np


def parse_args(add_help=True):
    # this adds help to main function
    parser = argparse.ArgumentParser(description='3d to 3D registration', add_help=add_help)
    # these are required inputs arguments
    parser.add_argument("--remotesubjectpath", required=True, type=str, help="This is full path where to files are located eaither locally or on remote server (ex:'/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/'")



    #this adds optional input arguments
    parser.add_argument("--input_overlap", default=None, type=int, help="# number of PIXELS to overlap, if not defined then just use calculated value")



    return parser



def denoise_BARSEQ(A, FFT_max_gaussian):
    """
    Denoise the BARSEQ 3d data and segments data
        Args:
        -	A: Numpy array 3D registered image

        Returns:
        -	A_seg: Numpy array 3D registered image deionised and segmented

    """

    #todo edit this so done in 3D not 2D

    # use band pass filtering to remove shaddow
    filt_A = skimage.filters.difference_of_gaussians(A, 1, FFT_max_gaussian)
    #denoise with wavlet this might not have much of an effect
    denoise = skimage.restoration.denoise_wavelet(filt_A, multichannel=False, rescale_sigma=True)
    #set stuff below mean to zero? --> for some reason this works best
    low_values_flags = denoise < denoise.mean()
    denoise[low_values_flags] = 0
    # segmentation: https://scikit-image.org/docs/0.18.x/auto_examples/segmentation/plot_morphsnakes.html#sphx-glr-auto-examples-segmentation-plot-morphsnakes-py
    # Initial level set
    init_ls = skimage.segmentation.checkerboard_level_set(denoise.shape, 6)  # default is 5
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    # smoothing of 3 works
    A_seg = skimage.segmentation.morphological_chan_vese(denoise, 35, init_level_set=init_ls, smoothing=3,
                                                         iter_callback=callback)  # here 35 is the number of iterations and smoothing=3 is number of times smoothing applied/iteration
    # check to make sure background =0 and segment brain is =1
    if np.count_nonzero(A_seg == 1) > np.count_nonzero(A_seg == 0):
        # inverse image
        A_seg = 1 - A_seg
    return A_seg



def denoise_2p(volume):
    """
    Visualizes the 3D registared brain
        Args:
        -	volume: Numpy array 3D registered image

        Returns:
        -	volume_seg: Numpy array 3D registered image deionised and segmented

    """

    #todo come up with stepps here

    volume_seg = volume

    denoiseA = ants.denoise_image(A_ant)

    return volume_seg




def main(args):
    #todo load 3D barseq data

    # denoise and segment Barseq data
    bar_seg = denoise_BARSEQ(A, FFT_max_gaussian)

    #todo load 3D 2 photon data

    # denoise and segment B2 photon data
    p2_seg = denoise_2p(volume)
    #register 2 datasets
    bar_ant = ants.from_numpy(data=bar_seg)
    p2_ant = ants.from_numpy(data=p2_seg)
    mytx = ants.registration(fixed=bar_ant, moving=p2_ant, str="Rigid", initial_transform=None, outprefix="test", dimension=2)  # “Similarity”
    # apply transfrom to unsegmented data
    p2_reg_ants = ants.apply_transforms(fixed=bar_ant, moving=p2_ant, transformlist=mytx['fwdtransforms'])
    # convert back to numpy
    p2_reg=p2_reg_ants.numpy()
    #save files
    save_feature_name = 'p2_reg'
    np.save(args.localsubjectpath + save_feature_name, p2_reg)



# run main program with parser
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)