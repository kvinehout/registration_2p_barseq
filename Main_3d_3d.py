#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 24th 2020
2d-to-3d
@author: Kaleb Vinehout

This code registeres together 3D reconsturcted confocal imaging and 2P data.

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
import skimage.feature
import skimage.io as skio
import matplotlib.pyplot as plt
# import functions from 2D to 3D, this has to be used before the code
import func_2d_3d as func2d3d
import itk


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
                        help="This is set to True for non-linear registration, otherwise linear registration perfromed"

    parser.add_argument("--apply_transfrom", default=False, type=str2bool,
                        help="This is set to true to apply 2P to confocal slices to other channel, need to provide saved_transforms_path. (ex apply_transfrom = True) (default: False) ")

    parser.add_argument("--saved_transforms_path", default=None, type=str,
                        help="This is used if --apply_transform is set to true. This is the full path to the folder where registration files from another set of images are saved. This needs to included image type for saved files (ex: Il-A)  ex: '/Users/kaleb/data/2d_3D_linear_reg/registration/Il-A'. (default:none)")

    parser.add_argument("--apply_3d_segmentatiomn_confocal", default=False, type=str2bool,
                        help="This is to apply 3D segmentation to reconstructed confocal imaging. Only use if 2D segmentation failed. (ex: --apply_3d_segmentatiomn_confocal=True) (default: False)")

    parser.add_argument("--nonlinear_trans", default=False, type=str2bool,
                        help="This is to apply nonlinear registration between 2p and confocal data, otherwise only linear registration performed. (ex --nonlinear_trans=True)(default = False)")

    parser.add_argument("--Zum_slices_to_2p", nargs=2, type=str, metavar=('Zstart', 'Zend'), required=True,
                        help="Required input. This is the Z values of 2P that roughly correspond to the slices provided. Both Zstart and Zend need to be provided. First number is Z start second is Z end. units are in ??? Ex ---Zum_slices_to-2p 5 10")

    parser.add_argument("--slices_axial_coronal_sagittal", default='axial', choices=['axial', 'coronal', 'sagittal'],
                        help="This defines the slices orientation. The slices are rotated based on this input to match 2p orientation")

    return parser


def make_figure(data, fig_name):
    """
    This makes and saves figures, this scales data for visualization
        Args:
        -	data: Numpy array 3D or 2D to be shown (YXZ --> max projection in  3rd dimension)
        Returns:
        -	fig_name: Full File path and .png name to save fig as
    """
    plt.figure()
    # if 3D
    if data.ndim == 3:
        plt.imshow(np.max(data, axis=2), cmap='magma', vmin=np.percentile(data, 0.1), vmax=np.percentile(data, 99.9))
    # if 2D
    elif data.ndim == 2:
        plt.imshow(data, cmap='magma', vmin=np.percentile(data, 0.1), vmax=np.percentile(data, 99.9))
    plt.show()
    plt.savefig('{}'.format(fig_name), format='png')
    plt.close()
    return 0


def zero_pad(A, B, dim):
    """
    This makes arrays the same size (zero pad) along dim
        Args:
        -	A,B: the A and B array to make the same size
        -   dim: dim overlap only accepts 0 or 1

        Returns:
        -	A_pad,B_pad: arrays zero padded

    """

    # remove extra dimensions
    A = np.squeeze(A)
    B = np.squeeze(B)

    data_dype = A.dtype

    if A.ndim == 2 and B.ndim == 2:
        if A.shape[dim] > B.shape[dim]:
            # B_pad = np.zeros(A.shape)  # * np.mean(srcZ_T_re)
            if dim == 1:
                B_pad = np.zeros([B.shape[0], A.shape[1]], dtype=data_dype)
            elif dim == 0:
                B_pad = np.zeros([A.shape[0], B.shape[1]], dtype=data_dype)
            B_pad[:B.shape[0], :B.shape[1]] = B
        else:
            B_pad = B
        if B.shape[dim] > A.shape[dim]:
            if dim == 1:
                A_pad = np.zeros([A.shape[0], B.shape[1]], dtype=data_dype)
            elif dim == 0:
                A_pad = np.zeros([B.shape[0], A.shape[1]], dtype=data_dype)
            A_pad[:A.shape[0], :A.shape[1]] = A
        else:
            A_pad = A

    elif A.ndim == 3 and B.ndim == 3:
        if A.shape[dim] > B.shape[dim]:
            # B_pad = np.zeros(A.shape)  # * np.mean(srcZ_T_re)
            if dim == 1:
                B_pad = np.zeros([B.shape[0], A.shape[1], B.shape[2]], dtype=data_dype)
            elif dim == 0:
                B_pad = np.zeros([A.shape[0], B.shape[1], B.shape[2]], dtype=data_dype)
            B_pad[:B.shape[0], :B.shape[1], :B.shape[2]] = B
        else:
            B_pad = B
        if B.shape[dim] > A.shape[dim]:
            if dim == 1:
                A_pad = np.zeros([A.shape[0], B.shape[1], A.shape[2]], dtype=data_dype)
            elif dim == 0:
                A_pad = np.zeros([B.shape[0], A.shape[1], A.shape[2]], dtype=data_dype)
            A_pad[:A.shape[0], :A.shape[1], :A.shape[2]] = A
        else:
            A_pad = A
    elif A.ndim == 2 and B.ndim == 3:
        A = np.expand_dims(A, axis=-1)  # expand dim for added unit
        if A.shape[dim] > B.shape[dim]:
            # B_pad = np.zeros(A.shape)  # * np.mean(srcZ_T_re)
            if dim == 1:
                B_pad = np.zeros([B.shape[0], A.shape[1], B.shape[2]], dtype=data_dype)
            elif dim == 0:
                B_pad = np.zeros([A.shape[0], B.shape[1], B.shape[2]], dtype=data_dype)
            B_pad[:B.shape[0], :B.shape[1], :B.shape[2]] = B
        else:
            B_pad = B
        if B.shape[dim] > A.shape[dim]:
            if dim == 1:
                A_pad = np.zeros([A.shape[0], B.shape[1], A.shape[2]], dtype=data_dype)
            elif dim == 0:
                A_pad = np.zeros([B.shape[0], A.shape[1], A.shape[2]], dtype=data_dype)
            A_pad[:A.shape[0], :A.shape[1], :A.shape[2]] = A
        else:
            A_pad = A
    elif A.ndim == 3 and B.ndim == 2:
        B = np.expand_dims(B, axis=-1)  # expand dim for added unit
        if A.shape[dim] > B.shape[dim]:
            # B_pad = np.zeros(A.shape)  # * np.mean(srcZ_T_re)
            if dim == 1:
                B_pad = np.zeros([B.shape[0], A.shape[1], B.shape[2]], dtype=data_dype)
            elif dim == 0:
                B_pad = np.zeros([A.shape[0], B.shape[1], B.shape[2]], dtype=data_dype)
            B_pad[:B.shape[0], :B.shape[1], :B.shape[2]] = B
        else:
            B_pad = B
        if B.shape[dim] > A.shape[dim]:
            if dim == 1:
                A_pad = np.zeros([A.shape[0], B.shape[1], A.shape[2]], dtype=data_dype)
            elif dim == 0:
                A_pad = np.zeros([B.shape[0], A.shape[1], A.shape[2]], dtype=data_dype)
            A_pad[:A.shape[0], :A.shape[1], :A.shape[2]] = A
        else:
            A_pad = A
    else:
        warnings.warn("WARNING: Shape of A is {} and shape of B is {}".format(A.shape, B.shape))
        A_pad = A
        B_pad = B
    del A, B
    return A_pad, B_pad


# todo  we want option here that can find same are in barseq data given same registration. So option to input reconstructed 3D ddata, given registration, and output segment that matches 2P data

def load_data(file_path):
    file_type = file_path.split(".", 1)[1]
    # if file path is numpy array
    if (file_type == 'tiff') or (file_type == 'tif'):
        # if file_path is tiff file
        data = skio.imread(file_path, plugin="tifffile")
    elif file_type == 'npy':
        data = np.load(file_path)
    elif file_type == 'npz':
        dataNPZ = np.load(file_path)
        data = dataNPZ['arr_0']
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

    # todo edit this so done in 3D not 2D  -->  should just need to input  3D volume

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


def apply_trans(transform, data2p, dataconfocal_3d):
    return reg_2p, reg_confocal


def find_part_in_whole_Z(data_2p_all_wide, data_2p_part, Z_start, Z_stop):
    """
    This finds where a given template is located within a given whole image. This only finds Z direction. Assumes XY already fround

    Input:
    whole_image: The whole image (Z,Y,X)
    template: The template image (Z,Y,X)

    Output:
    whole_image_template: The whole image that corresponds to the template
    Z_all = the corrdinates of where the template is located within the whole image
    """
    rigid_zrange = 80  # microns to search above and below estimated z for rigid registration
    px_z = field_z - stack_z + stack.shape[0] / 2 - 0.5
    mini_stack = stack[max(0, int(round(px_z - rigid_zrange))): int(round(px_z + rigid_zrange))]
    corrs = np.stack([feature.match_template(s, field, pad_input=True) for s in mini_stack])
    smooth_corrs = ndimage.gaussian_filter(corrs, 0.7)

    # Get results
    min_z = max(0, int(round(px_z - rigid_zrange)))
    min_y = int(round(0.05 * stack.shape[1]))
    min_x = int(round(0.05 * stack.shape[2]))
    mini_corrs = smooth_corrs[:, min_y:-min_y, min_x:-min_x]
    rig_z, rig_y, rig_x = np.unravel_index(np.argmax(mini_corrs), mini_corrs.shape)

    # Rewrite coordinates with respect to original z
    rig_z = (min_z + rig_z + 0.5) - stack.shape[0] / 2
    rig_y = (min_y + rig_y + 0.5) - stack.shape[1] / 2
    rig_x = (min_x + rig_x + 0.5) - stack.shape[2] / 2

    # todo if Z_start:Z_stop not provided --> then run part in whole

    Slice_data = data[:, :,
                 [Z_start:Z_stop]]  # todo double check this line ... maybe add find_part_in_whole as opption?

    return Slice_data


def find_part_in_whole_XY(data_2p_all_wide, data_2p_part):
    """
    This finds where a given template is located within a given whole image. Works for 2D or 3D images --> but only looks in X and Y direction.

    Input:
    whole_image: The whole image (Z,Y,X)
    template: The template image (Z,Y,X)

    Output:
    whole_image_template: The whole image that corresponds to the template
    X_all, Y_all = the corrdinates of where the template is located within the whole image

    see: https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html

    #Jake pipeline (lines 1302): https://github.com/cajal/pipeline/blob/master/python/pipeline/stack.py
    rigid_zrange = 80  # microns to search above and below estimated z for rigid registration
     px_z = field_z - stack_z + stack.shape[0] / 2 - 0.5
    mini_stack = stack[max(0, int(round(px_z - rigid_zrange))): int(round(px_z + rigid_zrange))]
    corrs = np.stack([feature.match_template(s, field, pad_input=True) for s in mini_stack])
    smooth_corrs = ndimage.gaussian_filter(corrs, 0.7)

        # Get results
        min_z = max(0, int(round(px_z - rigid_zrange)))
        min_y = int(round(0.05 * stack.shape[1]))
        min_x = int(round(0.05 * stack.shape[2]))
        mini_corrs = smooth_corrs[:, min_y:-min_y, min_x:-min_x]
        rig_z, rig_y, rig_x = np.unravel_index(np.argmax(mini_corrs), mini_corrs.shape)

        # Rewrite coordinates with respect to original z
        rig_z = (min_z + rig_z + 0.5) - stack.shape[0] / 2
        rig_y = (min_y + rig_y + 0.5) - stack.shape[1] / 2
        rig_x = (min_x + rig_x + 0.5) - stack.shape[2] / 2
    """
    # machine learning approach? http://proceedings.mlr.press/v80/michaelis18a/michaelis18a.pdf
    # https: // scikit - image.org / docs / dev / auto_examples / features_detection / plot_template.html  # sphx-glr-auto-examples-features-detection-plot-template-py

    if data_2p_all_wide.ndim == 3 and data_2p_part.ndim == 3:
        whole_image = np.max(data_2p_all_wide, axis=0)  # note here the tiff files Z is first
        template = np.max(data_2p_part, axis=0)  # note here the tiff files Z is first
        corrs = skimage.feature.match_template(whole_image, template, pad_input=True)
        corr_max = corrs.max()
        # get max value center point
        ij = np.unravel_index(np.argmax(corrs), corrs.shape)
        x, y = ij[::-1]  # this is center X Y value wrt whole-image --> this is the center point
        Ysize = template.shape[0]
        Xsize = template.shape[1]
        xmin = x - int(Xsize / 2)
        xmax = x + int(Xsize / 2)
        ymin = y - int(Ysize / 2)
        ymax = y + int(Ysize / 2)
        X_all = np.zeros(2)
        Y_all = np.zeros(2)
        X_all[0] = xmin
        X_all[1] = xmax
        Y_all[0] = ymin
        Y_all[1] = ymax
        whole_im_temp = data_2p_all_wide[:, ymin:ymax, xmin:xmax]  # get ALL Z since provided
    elif data_2p_all_wide.ndim == 2 and data_2p_part.ndim == 2:
        whole_image = data_2p_all_wide
        template = data_2p_part
        corrs = skimage.feature.match_template(whole_image, template, pad_input=True)
        corr_max = corrs.max()
        # get max value center point
        ij = np.unravel_index(np.argmax(corrs), corrs.shape)
        x, y = ij[::-1]  # this is center X Y value wrt whole-image --> this is the center point
        Ysize = template.shape[0]
        Xsize = template.shape[1]
        xmin = x - int(Xsize / 2)
        xmax = x + int(Xsize / 2)
        ymin = y - int(Ysize / 2)
        ymax = y + int(Ysize / 2)
        X_all = np.zeros(2)
        Y_all = np.zeros(2)
        X_all[0] = xmin
        X_all[1] = xmax
        Y_all[0] = ymin
        Y_all[1] = ymax
        whole_im_temp = data_2p_all_wide[ymin:ymax, xmin:xmax]  # get ALL Z since provided
    else:
        warnings.warn(message='NEED Template and Whole image to both have same number of dimensions')
        X_all = []
        Y_all = []
        whole_im_temp = []
    return X_all, Y_all, whole_im_temp, corr_max


def register_3d(image_source, image_destination, feature_or_elastix, nonlinear_trans, localsubjectpath):
    """This finds the 3D registration between source and destination. This is done with either feature based or elastix and linear or non-linear.
        Args:
        -	image_source: 3D set of images to overlap source (moving image)
        -   image_destination: 3D set of images to overlap target
        -   feature_or_elastix: this is if run feature or elastix
        -  nonlinear_trans: This is set to true for nonlinear, false otherwise

        Returns:
        -	reg_image_source: the registered source image in 3D
         -	trans:The calculated transformation
    """

    # zero pad the images????
    # The moving image is cropped because it is transformed to the fixed image domain (or "template" image) as you already figured out.
    # You can pad the fixed image with a appropriate number of zeros so that all moving image pixels fall within the fixed image (which is now larger because of the padding). Fx:
    # paddedFixedImage = sitk.ConstantPad(image, (10, 0), (10, 0))
    # np.save('/Users/kaleb/Desktop/source', source)
    # np.save('/Users/kaleb/Desktop/destination', destination)
    A_pad0, B_pad0 = zero_pad(image_source, image_destination, dim=0)
    A_pad1, B_pad1 = zero_pad(A_pad0, B_pad0, dim=1)
    source_pad, destination_pad = zero_pad(A_pad1, B_pad1, dim=2)

    if feature_or_elastix == 'elastix':
        # convert data to work with elastix software package
        destination_feature = destination.astype(np.float32)
        source_feature = source.astype(np.float32)
        fixed_image_feature = itk.image_from_array(destination_feature)
        moving_image_feature = itk.image_from_array(source_feature)
        # define parameter file
        parameter_object = itk.ParameterObject.New()
        if nonlinear_trans:
            default_bspline_parameter_map = parameter_object.GetDefaultParameterMap('bspline')
            parameter_object.AddParameterMap(default_bspline_parameter_map)
            parameter_object.SetParameter("Transform",
                                          "BSplineTransform")  # this is non-linear(use (SplineKernelTransform) for feautres
        else:
            default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
            parameter_object.AddParameterMap(default_rigid_parameter_map)
            parameter_object.SetParameter("Transform",
                                          "SimilarityTransform")  # this is scale, rottation, and translation
        parameter_object.SetParameter("Optimizer",
                                      "StandardGradientDescent")  # full search # this allows me to pick step size what about evolutionary strategy? FullSearch, ConjugateGradient, ConjugateGradientFRPR, QuasiNewtonLBFGS, RSGDEachParameterApart, SimultaneousPerturbation, CMAEvolutionStrategy.
        parameter_object.SetParameter("Registration", "MultiResolutionRegistration")
        parameter_object.SetParameter("Metric",
                                      "AdvancedKappaStatistic")  # "AdvancedNormalizedCorrelation") #AdvancedNormalizedCorrelation") #maybe AdvancedMattesMutualInformation
        parameter_object.SetParameter("FixedInternalImagePixelType", "float")
        parameter_object.SetParameter("MovingInternalImagePixelType", "float")
        parameter_object.SetParameter("FixedImageDimension", "3")
        parameter_object.SetParameter("MovingImageDimension", "3")
        parameter_object.SetParameter("FixedImagePyramid", "FixedRecursiveImagePyramid")  # smooth and downsample
        parameter_object.SetParameter("MovingImagePyramid", "MovingRecursiveImagePyramid")  # smooth and downsample
        parameter_object.SetParameter("NumberOfResolutions", "6")  # high resoltuiions b/c large shifts
        parameter_object.SetParameter("DefaultPixelValue", "0")  # this is b/c background is 0
        parameter_object.SetParameter("ImageSampler",
                                      "Grid")  # want full b/c random sampler would give different metric to compare results so full or grid should work, full is really slow
        parameter_object.SetParameter("NewSamplesEveryIteration",
                                      "false")  # want false b/c random sampler would give different metric to compare results
        # parameter_object.SetParameter("Scales", "1.0")  # this is so rotation and translation treated equally
        parameter_object.SetParameter("UseDirectionCosines", "false")
        parameter_object.SetParameter("AutomaticTransformInitialization", "true")
        parameter_object.SetParameter("AutomaticTransformInitializationMethod", "GeometricalCenter")
        parameter_object.SetParameter("UseComplement", "false")
        parameter_object.SetParameter("WriteResultImage", "true")
        parameter_object.SetParameter("MaximumNumberOfIterations", "500")
        parameter_object.SetParameter("MaximumNumberOfSamplingAttempts", "3")

        # get registration metirc for original data
        result_image, result_transform_parameters = itk.elastix_registration_method(fixed_image=fixed_image_feature,
                                                                                    moving_image=moving_image_feature,
                                                                                    parameter_object=parameter_object,
                                                                                    log_to_console=True,
                                                                                    log_to_file=True,
                                                                                    output_directory=localsubjectpath)

        reg_image_source = result_image
        trans = result_transform_parameters

        if feature_or_elastix == 'feature':

            destination_mean = np.max(destination_raw, axis=2)
            source_mean = np.max(source_raw, axis=2)
            # denoise image
            destination_denoise = denoise(destination_mean, rolling_ball_radius, double_gaussian, high_thres,
                                          Noise2void_or_classical)
            source_denoise = denoise(source_mean, rolling_ball_radius, double_gaussian, high_thres,
                                     Noise2void_or_classical)

            # todo segment images here before feature based and remove gray scale conversion?

            # define denoised data
            destination = destination_denoise
            source = source_denoise
            # convert to grayscale

            # todo remove converstion to gray scale?

            destination = np.interp(destination, (destination.min(), destination.max()), (0, +255))
            source = np.interp(source, (source.min(), source.max()), (0, +255))
            # change dtype to unit 8 --> so get rid of decimals
            destination = destination.astype(np.uint8)
            source = source.astype(np.uint8)
            im_src3d = Image.fromarray(source)
            im_src3d.save(localsubjectpath + "src3d_denoise.jpeg")
            im_des3d = Image.fromarray(destination)
            im_des3d.save(localsubjectpath + "des3d_denoise.jpeg")
            # Read reference imagw
            refFilename = '{}/{}'.format(localsubjectpath, 'des3d_denoise.jpeg')
            imReference = cv2.imread(refFilename, cv2.IMREAD_GRAYSCALE)
            # Read image to be aligned
            imFilename = '{}/{}'.format(localsubjectpath, 'src3d_denoise.jpeg')
            im1 = cv2.imread(imFilename, cv2.IMREAD_GRAYSCALE)
            # preallocate shift
            shift = np.zeros(2)
            MAX_MATCHES = 500
            GOOD_MATCH_PERCENT = 0.05  # this is 5 percent
            # Convert images to grayscale
            im1Gray = im1  # cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2Gray = imReference  # cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)
            # Detect ORB features and compute descriptors.
            orb = cv2.ORB_create(MAX_MATCHES)
            keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
            keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
            if np.any(descriptors1) == None or np.any(descriptors2) == None:
                if input_overlap == None:
                    warnings.warn("ERROR: CAN NOT CALCULATE IMAGE OVERLAP PLEASE PROVIDE --input_overlap VALUE")
                else:
                    warnings.warn("Warning: Can not calculate image overlap, instead using input_overlap value")
                    target_overlap = input_overlap
            else:
                # Match features. # get SAME NUMBER OF points for source and destination??? https://www.youtube.com/watch?v=cA8K8dl-E6k
                # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2)#BRUTEFORCE_HAMMING)
                # matches = matcher.match(descriptors1, descriptors2, None)
                # create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                matches = bf.match(descriptors1, descriptors2)
                # Sort matches by score
                matches.sort(key=lambda x: x.distance, reverse=False)
                # Remove not so good matches
                numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
                # matches = matches[:numGoodMatches]
                # Draw top matches
                imMatches = cv2.drawMatches(im1, keypoints1, imReference, keypoints2, matches, None)
                cv2.imwrite(localsubjectpath + "/matches.jpg", imMatches)
                # Extract location of good matches
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)
                for i, match in enumerate(matches):
                    points1[i, :] = keypoints1[match.queryIdx].pt
                    points2[i, :] = keypoints2[match.trainIdx].pt
                # Find Affine Transformation
                # note swap of order of newpoints here so that image2 is warped to match image1
                # if not enough points match --> set to input_overlap or warning message to add overlap
                if points1.shape[0] < 2 or points1.shape[0] < 2:
                    if input_overlap == None:
                        warnings.warn("ERROR: CAN NOT CALCULATE IMAGE OVERLAP PLEASE PROVIDE --input_overlap VALUE")
                    else:
                        warnings.warn("Warning: Can not calculate image overlap, instead using input_overlap value")
                        target_overlap = input_overlap
                else:
                    m, inliers = cv2.estimateAffinePartial2D(points1, points2)
                    # Use affine transform to warp im2 to match im1
                    height, width = imReference.shape
                    im1Reg = cv2.warpAffine(im1, m, (width, height))
                    shift[0] = m[1, 2]
                    shift[1] = m[0, 2]
                    error = 0  # todo add this?
                    # abs used below so source or destination order doesnt matter and subject from edge to get overlap value
                    if diroverlap == 'up' or diroverlap == 'down':
                        target_overlap = source_raw.shape[0] - abs(int(round(shift[0])))
                    elif diroverlap == 'left' or diroverlap == 'right':
                        target_overlap = source_raw.shape[1] - abs(int(round(shift[1])))
                    else:
                        warnings.warn(
                            "WARNING: diroverlap not defined correctly, Set to down, up, left or right. Currently set to {} ".format(
                                diroverlap))
            reg_image_source =
            trans =

    return reg_image_source


def main(args):
    print("starting 3d to 3d registration program")

    # todo add apply transform to below for other channel data
    # if apply transform is ture this is used to apply to other channel data
    if args.apply_transform:
        print("Apply transform set to true. Loading transform files from {}".format(args.saved_transforms_path))
        # save shift values
        transform = np.load(args.saved_transforms_path + 'transform.npy')
    if not args.apply_transform:
        transform = reg_2p_confocal(p2_seg, bar_seg)
    # apply transform to reconstructed data, find area that matches 2P input
    reg_2p, reg_confocal = apply_trans(transform, data2p, dataconfocal_3d)

    # todo step 0: load data
    file_path_confocal = args.file_path_confocal  # '/Volumes/Backup5TB/data/d3_array_denoise.npz'
    file_path_2p_wide = args.file_path_2p_wide  # '/Volumes/Backup5TB/data/baylor_2P_sample2/25618-2-3.tiff'  # 2-5 is C here we want C here b/c jake said so
    file_path_2p_part = args.file_path_2p_part  # /Volumes/Backup5TB/data/baylor_2P_sample2/25618-2-2.tiff
    # /Volumes/Backup5TB/data/baylor_2P_sample2/25388-2-3.tiff #this is shallow, otherwise 25388-2-2.tif is deep
    data_confocal = load_data(file_path_confocal)
    data_2p_all_wide = load_data(file_path_2p_wide)
    data_2p_all_part = load_data(file_path_2p_part)
    # divde data into channels
    data_2p_wide = data_2p_all_wide[1::2, :, :]  # here we want channel 2 data
    data_2p_part = data_2p_all_part[1::2, :, :]  # here we want channel 2 data

    # todo reorder so X Y Z not Z X Y?

    # rotate slices data if sliced in not 2p orientation
    if args.slices_axial_coronal_sagittal == 'coronal':
        print('Confocal imaging in coronal orientation. Rotation applied')
        data_confocal =  # todo ROTATE DATA
    elif args.slices_axial_coronal_sagittal == 'sagittal':
        print('Confocal imaging in sagittal orientation. Rotation applied')
        data_confocal =  # todo ROTATE DATA
    elif args.slices_axial_coronal_sagittal == 'axial':
        print('Confocal imaging in axial orientation. No rotation applied')
        data_confocal = data_confocal

    # todo segment data if needded
    data_2p_wide = segment_data(data_2p_wide, args.segment)
    data_2p_all_part = segment_data(data_2p_all_part, args.segment)
    data_confocal = segment_data(data_confocal, args.segment)

    # todo step 1: get Z values of 2P that match Z slices
    Z_start = args.Zum_slices_to_2p[0]
    Z_stop = args.Zum_slices_to_2p[1]
    Slice_2p_wide = find_slice_z(data_2p_wide, Z_start, Z_stop)
    Slice_2p_part = find_slice_z(data_2p_all_part, Z_start, Z_stop)

    # todo step 2: use wide field to resister 2P and these Z slices (have option for feature based AFTER elastix)
    data_confocal_reg = register_3d(data_confocal, Slice_2p_wide, args.feature_or_elastix)
    # step 3: get part in whole of 2p target area from 2P wide field
    [X_all, Y_all, whole_im_temp, corr_max] = find_part_in_whole_XY(data_2p_wide, data_2p_part)
    # find part in whole from registered confocal data (this way confocal in 2P space already
    data_confocal_reg_part = data_confocal_reg[:, X_all[0]:X_all[1], Y_all[0]:Y_all[
        1]]  # use X_part,Y_part,Z_part here thiis assumes data_confocal_reg is in Z Y X
    # step 4: use this subset of 2P and subset of Z slices to register together (have option for feature based AFTER elastix)
    data_confocal_reg_part_reg = register_3d(data_confocal_reg_part, data_2p_part, args.feature_or_elastix)
    print("saving registered files")
    save_folder = args.localsubjectpath
    np.savez_compressed(save_folder + args.image_type + '_register_confocal_part', data_confocal_reg_part_reg)
    np.savez_compressed(save_folder + args.image_type + '_slice_data_2p_part', Slice_2p_part)
    np.savez_compressed(save_folder + args.image_type + '_registered_confocal_whole', data_confocal_reg)
    np.savez_compressed(save_folder + args.image_type + '_slice_data_2p_wide', Slice_2p_wide)
    # todo change these so they save registration files from this code
    # save registration  values
    np.save(args.localsubjectpath + '_transform', transform)


# run main program with parser
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)

"""
EXTRA:

def denoise_2p(volume, FFT_max_gaussian):
    Visualizes the 3D registared brain
        Args:
        -	volume: Numpy array 3D registered image

        Returns:
        -	volume_seg: Numpy array 3D registered image deionised and segmented

    # here we want image C
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




def reg_2p_confocal(data2p_part, data2p_whole, dataconfocal_3d):

    This registered the 2P data to the confocal image
        Args:
        -	data2p: Numpy array 3D registered image
        -   dataconfocal_3d: this is the 3D volume from the confocal imaging

        Returns:
        -	reg_2p : Numpy array 3D registered image of 2P data
        -   reg_confocal: Numpy array 3D registered image of confocal data


    # step 0 part in whole registration with temple phase correlation (this gives good initial guess)
    # do this on whole 2P and part 2P data....

    #max project in z the confocal imaging to match style of 2P??? then intailize with 2D to 2d registration ???



    # step 1: also do template matching as initial guess for whole 2P data and confocal data (NO ROtATION HERE ???)

    whole_imaage_template, XYZ_temp = func2d3d.part_in_whole_registration(whole_image, template)




    # step 2: use elastix registation for fine tunnning --> here just to riddgid ?? (thIS GIVES ROTATONI)

    transformation = func2d3d.elastix_2D_registration(destination, source, degree_thres, theta_rad, args.nonlinear_trans)

    # based on 2p whole in part and on 2p whole and slices get part in slices

    # now taake disire 2P part aandd deisred barseq part  --> use affine registration here??




    # feature matching registration???? --> proabbly only use for segmented data? any benifits to template matchnig? --> phase correlation should be more robust

    # https://github.com/slicereg

    # step 1 find the part in whole registration b/c C 2p data is only PART of the barseq data or use 2p whole image to initalize?

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
    return transform

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
