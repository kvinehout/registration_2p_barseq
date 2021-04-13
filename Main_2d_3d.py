#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 24th 2020
2d-to-3d
@author: Kaleb Vinehout

This code combines 2D data into 3d volume

Input: folder path to files labels with POS names and type of files within this folder to register (ex: 'IL-A’)

Step 1: Rigidly Register within each POS folder
	- should have very close initial guess already
	- Here use ICP to register (local registration method)
		- get point cloud with canny edge detection
		- use 03d package for point cloud
		- apply point cloud transform to image with scikit image package


Step 2: Rigidly Stitch together Z plane
	- first calculate percentage image overlap (here we are probably adding a few hours to computational time…. .but i think its worth it)
		- here we take the first overlap and do a full search to find maximum image overlap similarity
		- assume only translation and only in 1 direction ( defined by how images are taken)
	- use calculated percentage image overlap as initial guess to stitch images together
	- to stitch images tougher use ICP (with o3d package)  or phase correlation (scikit - image package) with this initial guess
	- here only do 2D sticking, assuming Z axis is correct between POS folders


Step 3: Affine Register Z planes together
      - here we are assuming that the initial position is close to true alignment (b/c we have lots of data in a Z plane could try to globally register with ICP RANSAC algorithm if assumption not true)
	- use O3D package ICP software package
	- use point to plane ICP methods ( or point to point ICP ?) to register Z planes together
	- get transform and apply to images with scikit software package


Output: np array of 3D image, calculated image overlap,  and video(— or some other file type ?) of rotating brain


"""

print("importing modules")

# Import modules
import paramiko
import os
import skimage.registration
import skimage.transform
import skimage.util
import matplotlib.pyplot as plt
import numpy as np
import warnings
import argparse
import re
# improt from 2d-3d functions
import func_2d_3d as func2d3d

# reload module
# import importlib
# importlib.reload(func2d3d)

print("loading path name and user defined settings")


# HERE WE ASSUME 0,0 IS BOTTOM RIGHT
def parse_args(add_help=True):
    # this adds help to main function
    parser = argparse.ArgumentParser(description='2d to 3D registration', add_help=add_help)

    # make parser function for boolean
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # these are required inputs arguments
    parser.add_argument("--seq_dir", required=True, choices=['top', 'bottom'],
                        help="this is 'top' if POS1 is above POS 0 and 'bottom' if POS 1 is below POS 0")
    parser.add_argument("--image_type", required=True, type=str,
                        help="This is the file name prefex of images to register (ex:'Il-A). Note its assumed optical Z number is after this label in the image name")
    parser.add_argument("--localsubjectpath", required=True, type=str,
                        help="This is full path where to load and save files (ex:'/grid/zador/home/vinehout/code/2d_3D_linear_reg/')")

    parser.add_argument("--remotesubjectpath", required=True, type=str, action='append',
                        help="This is full path where to files are located eaither locally or on remote server (ex:'/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/'")

    parser.add_argument("--input_overlap", required=True, type=float,
                        help="Percent image overlap. 0-1, 0.10 represents 10 percent. (ex:--input_overlap=0.10)")

    # this adds optional input arguments
    parser.add_argument("--opticalZ_dir", default='top', choices=['top', 'bottom'],
                        help="this is 'top' if image 1 is above image 0 and 'bottom' if image 1 is below image 0 (default: 'top'")
    parser.add_argument("--X_dir", default='right', choices=['left', 'right'],
                        help="This is if zero along X is on the 'left' or 'right' of the image (defult:'right'")
    parser.add_argument("--Y_dir", default='top', choices=['top', 'bottom'],
                        help="This is if zero along Y is on the 'top' or 'bottom' of the image (defult:'top'")
    parser.add_argument('--server', default='local', type=str,
                        help="Name of server files are located,(ex:'zadorstorage2.cshl.edu') set to 'local' if files on local computer. (default: 'local')")
    parser.add_argument('--user', default=None, type=str,
                        help="Name of user to access server where filels are located,(ex:'user1'), Not required files on local computer. (default: None)")
    parser.add_argument('--password', default=None, type=str,
                        help="Password to access server where filels are located,(ex:'mypassword'), Not required files on local computer. (default: None)")
    parser.add_argument('--FFT_max_gaussian', default=10, type=int,
                        help="High_Sigma value for band pass filtering to remove low frequencies. See skimage.filters.difference_of_gaussians for details (default: 10)")
    parser.add_argument('--error_overlap', default=0.10, type=float,
                        help="Largest accetable error from input_overlay and actual registration shift. Percentage error repersented as a decimal (ex:0.10) (default 10 percent: 0.10)")
    parser.add_argument('--blank_thres', default=1.5, type=float,
                        help="Threshold to identifiy blank images. Used to see if max more then blank_thres * greater then mean of image (default: 1.5)")
    parser.add_argument('--checkerboard_size', default=6, type=int,
                        help="This is the square size for segmentation, see: skimage.segmentation.checkerboard_level_set (default: 6)")
    parser.add_argument('--seg_interations', default=35, type=int,
                        help="this in the number of segmentation interactions see: skimage.segmentation.morphological_chan_vese (defual:35)")
    parser.add_argument('--seg_smooth', default=3, type=int,
                        help="This is the number of smoothing interations durring segmentation see:  skimage.segmentation.morphological_chan_vese ( defult: 3)")
    parser.add_argument('--POS_folder_Z', nargs=2, type=str, metavar=('Zstart', 'Zend'), default=['0', '-6'],
                        help='Defines the position of the Z value in the folder name (EX: in POSZ(Z)_XXX_YYY the value is [0, -6]) Only numbers are indexed (defualt [0,-6].')
    parser.add_argument('--POS_folder_X', nargs=2, type=str, metavar=('Xstart', 'Xend'), default=['-6', '-3'],
                        help='Defines the position of the X value in the folder name (EX: in POSZ(Z)_XXX_YYY the value is [-6,-3]) Only numbers are indexed (defualt [-6,-3].')
    parser.add_argument('--POS_folder_Y', nargs=2, type=str, metavar=('Ystart', 'Yend'), default=['-2', 'None'],
                        help='Defines the position of the Y value in the folder name (EX: in POSZ(Z)_XXX_YYY the value is [-2,-1]) Only numbers are indexed (defualt [-2,-1].')

    parser.add_argument("--apply_transform", default=False, type=str2bool,
                        help="This is set to true if you want to apply transfomations calculated on a diffferent image channel to the image channel provided here. --saved_transforms_path needs to be defined for this work. (defualt: False)")
    parser.add_argument("--saved_transforms_path", default=None, type=str,
                        help="This is used if --apply_transform is set to true. This is the full path to the folder where registration files from another set of images are saved. This needs to included image type for saved files (ex: Il-A)  ex: '/Users/kaleb/data/2d_3D_linear_reg/registration/Il-A'. (default:none)")
    parser.add_argument("--extra_figs", type=str2bool, default=False,
                        help="This is used to save additional figured useful in troubleshooting, results saved in registration folder. (ex: --saved_transforms_path=True). (Default: Fasle)")

    parser.add_argument("--max_Z_proj_affine", default=False, type=str2bool,
                        help="This is used to use max Z projections instead of once slice for affine Z slice to Z slice calculateions. Set to true if images are  noisy, or false if low noise levels. (ex:--max_Z_proj_affine=True)(defualt:False)")
    parser.add_argument("--high_thres", default=10, type=int,
                        help="This is used to theshold image. Values above high_thres * the mean of the image will be set to zero (ex:--high_thres=10)(default:10)")
    parser.add_argument("--rigid_2d3d", default=False, type=str2bool,
                        help="This is used to only run rigid registratiion on feature map when combining Z planes, otherwise affine registration with optical flow is preformed.(ex:--rigid_2d3d=False)(default:False)")
    parser.add_argument("--output", default='registration', type=str,
                        help="This is used to define the output folder.(ex:--output='registration')(default:'registration')")

    parser.add_argument("--find_rot", default=False, type=str2bool,
                        help="This is set to true to look for 180 degree rotated physical slices.(ex:--find_rot=True)(default:'Fasle')")

    parser.add_argument("--degree_thres", default=10, type=int,
                        help="This is used to define the angle error tolerance from Physical Z slice to Physical Z slice registration, this tolerance is added to a search around zero and 180 degrees. Values are in degrees. (ex:--degree_thres=10)(default:10)")

    parser.add_argument("--denoise_all", default=True, type=str2bool,
                        help="This is used to output a denoise array that is the same size as the full dataset. If set to false the denoise array is only the images used for registration. (ex:--denoise_all=True)(default:True)")

    return parser


def main(args):
    print("starting registration program")
    # if apply transform is ture
    if args.apply_transform:
        print("Apply transform set to true. Loading transform files from {}".format(args.saved_transforms_path))
        # save shift values
        ld_target_overlapX = np.load(args.saved_transforms_path + '_target_overlapX.npy')
        ld_target_overlapY = np.load(args.saved_transforms_path + '_target_overlapY.npy')
        shift_within = np.load(args.saved_transforms_path + '_shift_within.npy')
        shiftX = np.load(args.saved_transforms_path + '_shiftX.npy')
        shiftY = np.load(args.saved_transforms_path + '_shiftY.npy')
        shiftZ = np.load(args.saved_transforms_path + '_shiftZ.npy', allow_pickle=True)
        angleZ = np.load(args.saved_transforms_path + '_angleZ.npy')
    else:
        shift_within = []
        shiftX = []
        shiftY = []
        shiftZ = []
        angleZ = []
    count_shift_within = 0
    count_shiftX = 0
    count_shiftY = 0
    count_shiftZ = 0
    error_all_within = []
    error_allX = []
    error_allY = []
    # todo paraellize these for loops?? thredding and multiprocess package?? --> need child processing?
    # use cprofile and profile to see where most time spend? or Cython --> complie to C?
    # check if multiple folders are input into remotesubjectpath if so combine all this data
    # append values?
    # memorize function? --> only if use same input multipel times... basically makes dictionary
    # use hashing to store variables?
    for pathi in range(int(args.remotesubjectpath.__len__())):  # todo change this line to test one Z plane
        remotesubjectpath_one = args.remotesubjectpath[pathi]
        print(remotesubjectpath_one)
        # get all file images to load in 3D cube
        # find number of Z axis --> this is number of imaging slices
        # connect to server
        if args.server != 'local':
            ssh_con = paramiko.SSHClient()
            ssh_con.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_con.connect(hostname=args.server, username=args.user, password=args.password)
            sftp_con = ssh_con.open_sftp()
            # get list of folders cubes in data path
            all_files_in_path = sftp_con.listdir(path=remotesubjectpath_one)
            r = re.compile('Pos')  # this gets all folders with POS in name
            cubefolder = list(filter(r.search, all_files_in_path))
            sftp_con.close()
            ssh_con.close()
        else:
            all_files_in_path = os.listdir(path=remotesubjectpath_one)
            r = re.compile('Pos')  # this gets all folders with POS in name
            cubefolder = list(filter(r.search, all_files_in_path))
        # make folder we save evertyhing to and folder we copy over
        if not os.path.isdir(args.localsubjectpath + '/' + args.output + '/'):
            os.mkdir(args.localsubjectpath + '/' + args.output + '/')
        # redefine local path to new folder
        args.localsubjectpath = args.localsubjectpath + '/' + args.output + '/'
        print("reading file name for file size")
        # get maximum Z
        ALLCubeZ = np.zeros(len(cubefolder))
        ALLCubeX = np.zeros(len(cubefolder))
        ALLCubeY = np.zeros(len(cubefolder))

        # look for none values and if not then convert to int
        POS_folder_X0 = None if args.POS_folder_X[0] == 'None' else int(args.POS_folder_X[0])
        POS_folder_X1 = None if args.POS_folder_X[1] == 'None' else int(args.POS_folder_X[1])
        POS_folder_Y0 = None if args.POS_folder_Y[0] == 'None' else int(args.POS_folder_Y[0])
        POS_folder_Y1 = None if args.POS_folder_Y[1] == 'None' else int(args.POS_folder_Y[1])
        POS_folder_Z0 = None if args.POS_folder_Z[0] == 'None' else int(args.POS_folder_Z[0])
        POS_folder_Z1 = None if args.POS_folder_Z[1] == 'None' else int(args.POS_folder_Z[1])

        # get maximum Z value
        for i in range(len(cubefolder)):
            allnum = re.sub("[^0-9]", "", cubefolder[i][:])
            ALLCubeZ[i] = int(allnum[POS_folder_Z0:POS_folder_Z1])
            # for each Z axis (NOT really each 3x3 grid)
            ALLCubeX[i] = int(allnum[POS_folder_X0:POS_folder_X1])
            ALLCubeY[i] = int(allnum[POS_folder_Y0:POS_folder_Y1])  # exec(foo + " = 'something else'")
        MaxCubeX = max(ALLCubeX)
        MaxCubeY = max(ALLCubeY)
        MaxCubeZ = max(ALLCubeZ)
        del ALLCubeZ
        # for each Z value
        # todo
        for Z_one in range(int(MaxCubeZ) + 1):
            Z = Z_one + (pathi * (int(MaxCubeZ) + 1))
            print("Z = {}".format(Z))
            # get maximum X and Y in case X/Y plane different size for each Z
            ALLCubeX = np.zeros(len(cubefolder))
            ALLCubeY = np.zeros(len(cubefolder))
            for i in range(len(cubefolder)):
                allnum = re.sub("[^0-9]", "", cubefolder[i][:])
                ALLCubeX[i] = int(allnum[POS_folder_X0:POS_folder_X1])
                ALLCubeY[i] = int(allnum[POS_folder_Y0:POS_folder_Y1])
            MaxCubeX = max(ALLCubeX)
            MaxCubeY = max(ALLCubeY)
            del ALLCubeX, ALLCubeY
            # for Y
            for Y in range(int(MaxCubeY) + 1):
                print("Y = {}".format(Y))
                # for X
                for X in range(int(MaxCubeX) + 1):
                    print("X = {}".format(X))
                    Zstring = str(Z_one)
                    Ystring = '00' + str(Y)
                    Xstring = '00' + str(X)
                    cubename = '/Pos' + Zstring + '_' + Xstring + '_' + Ystring + '/'
                    # download files in this X Y Z folder --> this is ONE cube
                    remotefilepath = remotesubjectpath_one + cubename
                    localfilepath = args.localsubjectpath  # dont need folder path b/c images removed after downloaded
                    # download and define 3D variable for files in this folder
                    imarray3D = func2d3d.sshfoldertransfer(args.server, args.user, args.password, remotefilepath,
                                                           localfilepath, args.image_type, args.opticalZ_dir)
                    print("Registration of optical slices within folder {}.".format(remotefilepath))
                    # STEP 1 --> ALIGN SLICES in IMARRAY 3D
                    for i in range(len(imarray3D[1, 1, :]) - 1):
                        A = imarray3D[:, :, (i + 1)]  # this is the source
                        # for first image I use this as destination
                        if i == 0:
                            B = imarray3D[:, :, i]  # this is the  destination
                        else:
                            # if any other image get destination from prior AB_img
                            B = AB_img[:, :, i]
                            # define B_whole as the destination already registered
                            AB_whole = AB_img
                        if args.apply_transform:
                            error = 0  # not calculated
                            shift_reg = shift_within[count_shift_within]
                            Within_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                                                 translation=(
                                                                                     shift_reg[1], shift_reg[0]))
                            count_shift_within = count_shift_within + 1
                        else:
                            error, shift_reg, Within_Trans = func2d3d.registration_within(A, B, args.FFT_max_gaussian,
                                                                                          args.error_overlap, X, Y, Z,
                                                                                          i, args.localsubjectpath,
                                                                                          args.extra_figs,
                                                                                          args.high_thres)
                            # apply value to shift ONLY if calculating shift
                            shift_within.append(shift_reg)
                        error_all_within.append(error)
                        A_T = skimage.transform.warp(A, Within_Trans._inv_matrix, mode='edge')
                        if A_T.max() < 1:  # only if max less then one convert to unit
                            A_T = skimage.img_as_uint(A_T, force_copy=False)
                        # combine A and B into one matrix
                        if i == 0:
                            # here we need to add Z value so source and destiantion values given Z
                            AB_img = np.stack((B, A_T), axis=-1)  # add axis to the end
                        else:
                            # for other images add B_whole + B_xy + A_TrC
                            A_T = np.expand_dims(A_T, axis=-1)
                            AB_img = np.concatenate((AB_whole, A_T), axis=2)
                            del AB_whole
                        del A_T, Within_Trans, A, B
                    # STEP 2 Y line: Stitch together images with first X initial guess and O3D point to point ICP registration (1st make Y lines, then combine Y lines
                    srcY = AB_img  # define source image
                    del imarray3D
                    if X == 0:
                        # DON'T STITCH Because nothing to stitch
                        AB_img_old = AB_img
                        del AB_img
                    else:
                        print("Stitching file X = {} and X= {} together.".format(X, (X - 1)))
                        desY = AB_img_old  # define destination image
                        if args.apply_transform:
                            error = 0  # not calculated
                            shift = shiftX[count_shiftX]
                            target_overlapX = ld_target_overlapX
                            count_shiftX = count_shiftX + 1
                            blank_overlap = False
                        else:
                            if X == 1 and Y == 0 and Z == 0:  # calculate the initial stitch level ONLY for first overlap
                                error, shift, target_overlapX, blank_overlap = func2d3d.registration_X(srcY, desY,
                                                                                                       args.X_dir,
                                                                                                       args.FFT_max_gaussian,
                                                                                                       args.error_overlap,
                                                                                                       X, Y, Z,
                                                                                                       args.blank_thres,
                                                                                                       args.localsubjectpath,
                                                                                                       args.extra_figs,
                                                                                                       args.high_thres,
                                                                                                       args.input_overlap)
                            else:
                                error, shift, target_overlapX, blank_overlap = func2d3d.registration_X(srcY, desY,
                                                                                                       args.X_dir,
                                                                                                       args.FFT_max_gaussian,
                                                                                                       args.error_overlap,
                                                                                                       X, Y, Z,
                                                                                                       args.blank_thres,
                                                                                                       args.localsubjectpath,
                                                                                                       args.extra_figs,
                                                                                                       args.high_thres,
                                                                                                       args.input_overlap,
                                                                                                       target_overlapX)
                            shiftX.append(shift)
                        error_allX.append(error)
                        # calculate OVERALL SHIFT from inital guess + phase correlations + error?
                        SKI_Trans_all = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                                              translation=(
                                                                                  (target_overlapX + shift[1]),
                                                                                  shift[0]))
                        # APPLY SHIFT TO WHOLE 3D volume with  LOOP COMPREHENSION
                        srcY_T = [skimage.transform.warp(srcY[:, :, i], SKI_Trans_all._inv_matrix, mode='edge') for i in
                                  range(srcY.shape[2])]
                        if np.max(srcY_T) < 1:  # only if max less then one convert to unit
                            srcY_T = skimage.img_as_uint(srcY_T, force_copy=False)
                        # rearanage
                        srcY_T_re = np.transpose(srcY_T, axes=[1, 2, 0])
                        # todo calculate features here and add to feature map
                        if args.max_Z_proj_affine:
                            srcY_T_re_one = np.max(srcY_T_re, axis=2)
                            desY_one = np.max(desY, axis=2)
                        else:
                            if args.seq_dir == 'top':
                                srcY_T_re_one = srcY_T_re[:, :, 0]
                                desY_one = desY[:, :, -1]
                            elif args.seq_dir == 'bottom':
                                srcY_T_re_one = srcY_T_re[:, :, -1]
                                desY_one = desY[:, :, 0]
                            else:
                                warnings.warn("opticalZ_dir variable not defined correctly")
                        # check if whole image is blank, if so then set to zero and skip feature detection?
                        if blank_overlap and srcY_T_re_one.max() < args.blank_thres * srcY_T_re_one.mean():
                            if args.denoise_all:
                                srcY_T_re_denoise = np.zeros(srcY_T_re.shape)
                            else:
                                srcY_T_re_denoise = np.zeros(srcY_T_re_one.shape)
                            print('Blank image at X {} Y {} Z{}, Skipping denoise, setting values to zero'.format(X, Y,
                                                                                                                  Z))
                        else:
                            if args.denoise_all:
                                srcY_T_re_denoise = [
                                    func2d3d.denoise(srcY_T_re[:, :, i], args.FFT_max_gaussian, args.high_thres) for i
                                    in range(srcY_T_re.shape[2])]
                                srcY_T_re_denoise = np.transpose(srcY_T_re_denoise, axes=[1, 2, 0])
                            else:
                                srcY_T_re_denoise = func2d3d.denoise(srcY_T_re_one, args.FFT_max_gaussian,
                                                                     args.high_thres)
                        if args.extra_figs:
                            plt.figure()
                            plt.imshow(srcY_T_re_one)
                            name = 'X={}_Y={}_Z={}'.format(X, Y, Z)
                            plt.savefig(args.localsubjectpath + name + '_raw.png', format='png')
                            plt.close()
                            plt.figure()
                            if args.denoise_all:
                                plt.imshow(np.max(srcY_T_re_denoise, axis=2))
                            else:
                                plt.imshow(srcY_T_re_denoise)
                            name = 'X={}_Y={}_Z={}'.format(X, Y, Z)
                            plt.savefig(args.localsubjectpath + name + '_denoised.png', format='png')
                            plt.close()
                        del srcY_T, SKI_Trans_all, srcY, srcY_T_re_one
                        if X == 1:
                            if args.denoise_all:
                                desY_denoise = [func2d3d.denoise(desY[:, :, i], args.FFT_max_gaussian, args.high_thres)
                                                for i in range(desY.shape[2])]
                                desY_denoise = np.transpose(desY_denoise, axes=[1, 2, 0])
                            else:
                                desY_denoise = func2d3d.denoise(desY_one, args.FFT_max_gaussian, args.high_thres)
                            # concat 3D volumes together
                            if args.X_dir == 'right':
                                Y_img = np.concatenate((srcY_T_re, desY), axis=1)
                                Y_img_denoise = np.concatenate((srcY_T_re_denoise, desY_denoise), axis=1)
                            elif args.X_dir == 'left':
                                Y_img = np.concatenate((desY, srcY_T_re), axis=1)
                                Y_img_denoise = np.concatenate((desY_denoise, srcY_T_re_denoise), axis=1)
                            if args.extra_figs:
                                plt.figure()
                                if args.denoise_all:
                                    plt.imshow(np.max(Y_img_denoise, axis=2))
                                else:
                                    plt.imshow(Y_img_denoise)
                                name = 'X={}_Y={}_Z={}'.format(X, Y, Z)
                                plt.savefig(args.localsubjectpath + name + 'Y_img_denoise.png', format='png')
                                plt.close()
                        else:
                            if args.X_dir == 'right':
                                Y_img = np.concatenate((srcY_T_re, Yline),
                                                       axis=1)  # need to match in other directions.. so keep Zeros in Y direction
                                Y_img_denoise = np.concatenate((srcY_T_re_denoise, Yline_denoise), axis=1)
                            elif args.X_dir == 'left':
                                Y_img = np.concatenate((Yline, srcY_T_re), axis=1)
                                Y_img_denoise = np.concatenate((Yline_denoise, srcY_T_re_denoise), axis=1)
                        # define YLine as Y_img without the shifted zero values
                        if args.X_dir == 'right':
                            Yline = Y_img[:, int(abs(target_overlapX + shift[
                                1])):]  # need to match in other directions.. so keep Zeros in Y direction
                            Yline_denoise = Y_img_denoise[:, int(abs(target_overlapX + shift[1])):]
                        elif args.X_dir == 'left':
                            Yline = Y_img[:, :-int(abs(target_overlapX + shift[
                                1]))]  # need to match in other directions.. so keep Zeros in Y direction
                            Yline_denoise = Y_img_denoise[:, :-int(abs(target_overlapX + shift[1]))]
                        AB_img_old = AB_img
                        del AB_img, desY, srcY_T_re, shift, error, Y_img, Y_img_denoise, srcY_T_re_denoise
                        if args.extra_figs:
                            plt.figure()
                            if args.denoise_all:
                                plt.imshow(np.max(Yline_denoise, axis=2))
                            else:
                                plt.imshow(Yline_denoise)
                            name = 'X={}_Y={}_Z={}'.format(X, Y, Z)
                            plt.savefig(args.localsubjectpath + name + 'Yline_denoise.png', format='png')
                            plt.close()
                # STEP 2 Z plane: Stitch together images with first X initial guess and O3D point to point ICP registration (1st make Y lines, then combine Y lines
                srcZ = Yline  # this is the Y to add to rest
                srcZ_denoise = Yline_denoise
                if Y == 0:
                    # DON'T STITCH Because nothing to stitch
                    Yline_old = Yline
                    Yline_old_denoise = Yline_denoise
                    del Yline, Yline_denoise
                else:
                    print("Stitching file Y = {} and Y= {} together.".format(Y, (Y - 1)))
                    desZ = Yline_old  # define destination image
                    desZ_denoise = Yline_old_denoise
                    if args.apply_transform:
                        error = 0  # not calculated
                        shift = shiftY[count_shiftY]
                        target_overlapY = ld_target_overlapY
                        count_shiftY = count_shiftY + 1
                    else:
                        if Y == 1 and Z == 0:  # calculate the initial stitch level ONLY for first overlap
                            error, shift, target_overlapY = func2d3d.registration_Y(srcZ, desZ, args.Y_dir,
                                                                                    args.FFT_max_gaussian,
                                                                                    args.error_overlap, X, Y, Z,
                                                                                    args.blank_thres,
                                                                                    args.localsubjectpath,
                                                                                    args.extra_figs, args.high_thres,
                                                                                    args.input_overlap)
                        else:
                            error, shift, target_overlapY = func2d3d.registration_Y(srcZ, desZ, args.Y_dir,
                                                                                    args.FFT_max_gaussian,
                                                                                    args.error_overlap, X, Y, Z,
                                                                                    args.blank_thres,
                                                                                    args.localsubjectpath,
                                                                                    args.extra_figs, args.high_thres,
                                                                                    args.input_overlap, target_overlapY)
                        shiftY.append(shift)
                    error_allY.append(error)
                    del AB_img_old
                    # calculate OVERALL SHIFT from inital guess + phase correlations
                    SKI_Trans_all = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                                          translation=(
                                                                              (shift[1], (target_overlapY + shift[0]))))
                    # APPLY SHIFT TO WHOLE 3D volume with  LOOP COMPREHENSION
                    srcZ_T = [skimage.transform.warp(srcZ[:, :, i], SKI_Trans_all._inv_matrix, mode='edge') for i in
                              range(srcZ.shape[2])]
                    if np.max(srcZ_T) < 1:  # only if max less then one convert to unit
                        srcZ_T = skimage.img_as_uint(srcZ_T, force_copy=False)
                    else:
                        srcZ_T = np.array(srcZ_T, dtype=srcZ.dtype)
                    # rearrange
                    srcZ_T_re = np.transpose(srcZ_T, axes=[1, 2, 0])
                    if args.denoise_all:
                        srcZ_T_denoise = [
                            skimage.transform.warp(srcZ_denoise[:, :, i], SKI_Trans_all._inv_matrix, mode='edge') for i
                            in range(srcZ_denoise.shape[2])]
                        if np.max(srcZ_T_denoise) < 1:  # only if max less then one convert to unit
                            srcZ_T_denoise = skimage.img_as_uint(srcZ_T_denoise, force_copy=False)
                        else:
                            srcZ_T_denoise = np.array(srcZ_T_denoise, dtype=srcZ_denoise.dtype)
                        # rearrange
                        srcZ_T_re_denoise = np.transpose(srcZ_T_denoise, axes=[1, 2, 0])
                    else:
                        # APPLY SHIFT to denoise value
                        srcZ_T_re_denoise = skimage.transform.warp(srcZ_denoise, SKI_Trans_all._inv_matrix, order=0,
                                                                   mode='edge', preserve_range=True)
                        srcZ_T_re_denoise = srcZ_T_re_denoise.astype(srcZ_denoise.dtype)
                    if args.extra_figs:
                        plt.figure()
                        if args.denoise_all:
                            plt.imshow(np.max(srcZ_T_re_denoise, axis=2))
                        else:
                            plt.imshow(srcZ_T_re_denoise)
                        name = 'X={}_Y={}_Z={}'.format(X, Y, Z)
                        plt.savefig(args.localsubjectpath + name + 'srcZ_T_re_denoise.png', format='png')
                        plt.close()
                    del srcZ_T
                    if Y == 1:
                        # pad the Zplane in the X
                        dim = 1
                        [srcZ_T_re, desZ] = func2d3d.zero_pad(srcZ_T_re, desZ, dim)
                        [srcZ_T_re_denoise, desZ_denoise] = func2d3d.zero_pad(srcZ_T_re_denoise, desZ_denoise, dim)
                        if args.Y_dir == 'top':
                            Z_img = np.concatenate((srcZ_T_re, desZ), axis=0)
                            Z_img_denoise = np.concatenate((srcZ_T_re_denoise, desZ_denoise), axis=0)
                        elif args.Y_dir == 'bottom':
                            Z_img = np.concatenate((desZ, srcZ_T_re), axis=0)
                            Z_img_denoise = np.concatenate((desZ_denoise, srcZ_T_re_denoise), axis=0)
                    else:
                        # pad the Zplane in the X
                        dim = 1
                        [Zplane, srcZ_T_re] = func2d3d.zero_pad(Zplane, srcZ_T_re, dim)
                        [Zplane_denoise, srcZ_T_re_denoise] = func2d3d.zero_pad(Zplane_denoise, srcZ_T_re_denoise, dim)
                        if args.Y_dir == 'top':
                            Z_img = np.concatenate((srcZ_T_re, Zplane), axis=0)
                            Z_img_denoise = np.concatenate((srcZ_T_re_denoise, Zplane_denoise), axis=0)
                        elif args.Y_dir == 'bottom':
                            Z_img = np.concatenate((Zplane, srcZ_T_re), axis=0)
                            Z_img_denoise = np.concatenate((Zplane_denoise, srcZ_T_re_denoise), axis=0)
                    if args.Y_dir == 'top':
                        Zplane = Z_img[int(abs(target_overlapY + shift[0])):,
                                 :]  # this removes Zeros that are due to shift
                        Zplane_denoise = Z_img_denoise[int(abs(target_overlapY + shift[0])):,
                                         :]  # this removes Zeros that are due to shift
                    elif args.Y_dir == 'bottom':
                        Zplane = Z_img[:-int(abs(target_overlapY + shift[0])),
                                 :]  # this removes Zeros that are due to shift
                        Zplane_denoise = Z_img_denoise[:-int(abs(target_overlapY + shift[0])), :]
                    Yline_old = Yline
                    Yline_old_denoise = Yline_denoise
                    if args.extra_figs:
                        plt.figure()
                        if args.denoise_all:
                            plt.imshow(np.max(Z_img_denoise, axis=2))
                        else:
                            plt.imshow(Z_img_denoise)
                        name = 'X={}_Y={}_Z={}'.format(X, Y, Z)
                        plt.savefig(args.localsubjectpath + name + 'Z_img_denoise.png', format='png')
                        plt.close()
                    del Yline, Yline_denoise, Z_img, Z_img_denoise, desZ, desZ_denoise, srcZ, srcZ_denoise, srcZ_T_re, srcZ_T_re_denoise, error, shift
            # Step 3 combine Z plane ---> use ICP without initial guess (maybe globally registration? )
            # WHAT If foR z we take mean in optical Z to get feautre map
            # we use this feautre map to calcualte translation
            # apply trnaslation to orginal images so we done loose any resolution but mean out noise (both salt/peper and low freq shift)
            src3d = Zplane  # this is the Y to add to rest
            src3d_denoise = Zplane_denoise  # this is the Y to add to rest
            if Z == 0:
                # DON'T STITCH Because nothing to stitch
                Zplane_old = Zplane
                Zplane_old_denoise = Zplane_denoise
                del Zplane, Yline_old, Zplane_denoise, Yline_old_denoise
            else:
                print("Registration of  Z = {} and Z= {}.".format(Z, (Z - 1)))
                des3d = Zplane_old  # define destination image
                des3d_denoise = Zplane_old_denoise  # define destination image
                # zero pad
                dim = 0
                [src3d, des3d] = func2d3d.zero_pad(src3d, des3d, dim)
                [src3d_denoise, des3d_denoise] = func2d3d.zero_pad(src3d_denoise, des3d_denoise, dim)
                dim = 1
                [src3d, des3d] = func2d3d.zero_pad(src3d, des3d, dim)
                [src3d_denoise, des3d_denoise] = func2d3d.zero_pad(src3d_denoise, des3d_denoise, dim)
                # get feature map for each optical slice with segmentation
                src3d_feature = func2d3d.segmentation(src3d_denoise, args.checkerboard_size, args.seg_interations,
                                                      args.seg_smooth)
                des3d_feature = func2d3d.segmentation(des3d_denoise, args.checkerboard_size, args.seg_interations,
                                                      args.seg_smooth)
                # preform Z registration
                src3d_T, src3d_T_feature, src3d_T_denoise, count_shiftZ, shiftZ, error_allZ = func2d3d.registration_Z(
                    src3d, src3d_denoise,
                    des3d_denoise, src3d_feature,
                    des3d_feature, count_shiftZ,
                    shiftZ, angleZ, args.apply_transform,
                    args.rigid_2d3d, args.error_overlap, args.find_rot, args.degree_thres, args.denoise_all,
                    args.max_Z_proj_affine, args.seq_dir)
                if args.extra_figs:
                    # make plots of Z plane
                    plt.figure()
                    plt.imshow(src3d_T_feature)
                    plt.savefig(args.localsubjectpath + '/Z=' + str(Z) + '_Zplane_feature_shift.png', format='png')
                    plt.close()
                    plt.figure()
                    if args.denoise_all:
                        plt.imshow(np.max(src3d_denoise, axis=2))
                    else:
                        plt.imshow(src3d_denoise)
                    plt.savefig(args.localsubjectpath + '/Z=' + str(Z) + '_Zplane_denoise.png', format='png')
                    plt.close()
                    plt.figure()
                    plt.imshow(src3d_feature)
                    plt.savefig(args.localsubjectpath + '/Z=' + str(Z) + '_Zplane_feature.png', format='png')
                    plt.close()
                if Z == 1:
                    # pad in the Y
                    dim = 0
                    [des3d, src3d_T] = func2d3d.zero_pad(des3d, src3d_T, dim)
                    [des3d_feature, src3d_T_feature] = func2d3d.zero_pad(des3d_feature, src3d_T_feature, dim)
                    [des3d_denoise, src3d_T_denoise] = func2d3d.zero_pad(des3d_denoise, src3d_T_denoise, dim)
                    # pad the 3D image in the X
                    dim = 1
                    [des3d, src3d_T] = func2d3d.zero_pad(des3d, src3d_T, dim)
                    [des3d_denoise, src3d_T_denoise] = func2d3d.zero_pad(des3d_denoise, src3d_T_denoise, dim)
                    [des3d_feature, src3d_T_feature] = func2d3d.zero_pad(des3d_feature, src3d_T_feature, dim)
                    # expand dimentions of ndim = 2 data
                    src3d_T_feature = np.expand_dims(src3d_T_feature, axis=-1)
                    des3d_feature = np.expand_dims(des3d_feature, axis=-1)
                    #this is done if denoise_all is not selected then ndim==2
                    if des3d_denoise.ndim == 2:
                        des3d_denoise = np.expand_dims(des3d_denoise, axis=-1)
                    if src3d_T_denoise.ndim == 2:
                        src3d_T_denoise = np.expand_dims(src3d_T_denoise, axis=-1)
                    if args.seq_dir == 'top':  # this is top if POS1 is above POS 0 and 'bottom' if POS 1 is below POS 0
                        d3_img = np.concatenate((src3d_T, des3d), axis=2)
                        print(src3d_T.shape())
                        print(des3d.shape())
                        print(src3d_T_denoise.shape())
                        print(des3d_denoise.shape())
                        print(src3d_T_feature.shape())
                        print(des3d_feature.shape())
                        d3_img_denoise = np.concatenate((src3d_T_denoise, des3d_denoise), axis=2)
                        d3_img_feature = np.concatenate((src3d_T_feature, des3d_feature), axis=2)
                    elif args.seq_dir == 'bottom':
                        d3_img = np.concatenate((des3d, src3d_T), axis=2)
                        d3_img_denoise = np.concatenate((des3d_denoise, src3d_T_denoise), axis=2)
                        d3_img_feature = np.concatenate((des3d_feature, src3d_T_feature), axis=2)
                    else:
                        warnings.warn("WARNING: seq_dir variable not defined properly: Use 'top' or 'bottom'")
                    del des3d, src3d_T, src3d_T_feature, des3d_feature
                else:
                    # expand feature map dimensions if needed
                    if src3d_T_feature.ndim == 2:
                        src3d_T_feature = np.expand_dims(src3d_T_feature, axis=-1)  # expand dim for added unit
                    if d3_array_feature.ndim == 2:
                        d3_array_feature = np.expand_dims(d3_array_feature, axis=-1)  # expand dim for added unit
                    # pad the 3D image in the Y
                    # pad in the Y
                    dim = 0
                    [d3_array, src3d_T] = func2d3d.zero_pad(d3_array, src3d_T, dim)
                    [d3_array_denoise, src3d_T_denoise] = func2d3d.zero_pad(d3_array_denoise, src3d_T_denoise, dim)
                    [d3_array_feature, src3d_T_feature] = func2d3d.zero_pad(d3_array_feature, src3d_T_feature, dim)
                    # pad the 3D image in the X
                    dim = 1
                    [d3_array, src3d_T] = func2d3d.zero_pad(d3_array, src3d_T, dim)
                    [d3_array_denoise, src3d_T_denoise] = func2d3d.zero_pad(d3_array_denoise, src3d_T_denoise, dim)
                    [d3_array_feature, src3d_T_feature] = func2d3d.zero_pad(d3_array_feature, src3d_T_feature, dim)
                    if args.seq_dir == 'top':  # this is top if POS1 is above POS 0 and 'bottom' if POS 1 is below POS 0
                        d3_img = np.concatenate((src3d_T, d3_array), axis=2)
                        d3_img_denoise = np.concatenate((src3d_T_denoise, d3_array_denoise), axis=2)
                        d3_img_feature = np.concatenate((src3d_T_feature, d3_array_feature), axis=2)
                    elif args.seq_dir == 'bottom':
                        d3_img = np.concatenate((d3_array, src3d_T), axis=2)
                        d3_img_denoise = np.concatenate((d3_array_denoise, src3d_T_denoise), axis=2)
                        d3_img_feature = np.concatenate((d3_array_feature, src3d_T_feature), axis=2)
                    else:
                        warnings.warn("WARNING: seq_dir variable not defined properly: Use 'top' or 'bottom'")
                    del src3d_T, src3d_T_feature
                d3_array = d3_img  # Z_img[int(abs(target_overlapY + shift[0])):, :]  # this removes Zeros that are due to shift
                d3_array_denoise = d3_img_denoise  # Z_img[int(abs(target_overlapY + shift[0])):, :]  # this removes Zeros that are due to shift
                d3_array_feature = d3_img_feature
                Zplane_old = Zplane
                del Zplane, d3_img, d3_img_feature, d3_img_denoise
    # DO THIS AFTER ALL folders
    del Zplane_old
    # print out error and shift to terminal
    print("Error in the X direction. {}".format(error_allX))
    print("Error in the Y direction. {}".format(error_allY))
    print("Error in the Z direction. {}".format(error_allZ))
    print("Estimation of overlap in X. {}".format(target_overlapX))
    print("Estimation of overlap in Y. {}".format(target_overlapY))
    print("Shift for each cube in the optical Z direction. {}".format(shift_within))
    print("Shift for each cube in the X direction. {}".format(shiftX))
    print("Shift for each cube in the Y direction. {}".format(shiftY))
    print("Shift for each cube in the Z direction. {}".format(shiftZ))

    # save shift values
    np.save(args.localsubjectpath + args.image_type + '_target_overlapX', target_overlapX)
    np.save(args.localsubjectpath + args.image_type + '_target_overlapY', target_overlapY)
    np.save(args.localsubjectpath + args.image_type + '_shift_within', shift_within)
    np.save(args.localsubjectpath + args.image_type + '_shiftX', shiftX)
    np.save(args.localsubjectpath + args.image_type + '_shiftY', shiftY)
    np.save(args.localsubjectpath + args.image_type + '_shiftZ', shiftZ)
    np.save(args.localsubjectpath + args.image_type + '_angleZ', angleZ)

    # STEP 4: visualize and save 3d array
    # save array for image and feature map
    print("saving files and making .gifs")
    save_folder = args.localsubjectpath
    np.savez_compressed(save_folder + args.image_type + '_d3_array_denoise', d3_array_denoise)
    np.savez_compressed(save_folder + args.image_type + '_d3_array_feature', d3_array_feature)
    np.savez_compressed(save_folder + args.image_type + '_d3_array', d3_array)

    # make videos/images

    # make and save video of the result
    # import imageio

    # images = []
    # d3_array = np.uint8(d3_array)
    # for i in range(d3_array.shape[2]):
    #    images.append(Image.fromarray(d3_array[:, :, i]))
    # gif_filename_slide = '3D_brain_slide.gif'
    # gif_slide_full_path = args.localsubjectpath + gif_filename_slide
    # imageio.mimsave(gif_slide_full_path, images)

    """
        This example shows how to render volumetric (i.e. organized in voxel)
        data in brainrender. The data used are is the localized expression of 
        'Gpr161' from the Allen Atlas database, downloaded with brainrender
        and saved to a numpy file

    import brainrender
    from bg_space import AnatomicalSpace
    from vedo import Volume
    from brainrender import Scene
    from pathlib import Path
    from myterial import blue_grey
    from rich import print
    from myterial import orange
    import imio
    import bg_space as bg

    
    #see: https://github.com/brainglobe/brainrender/blob/master/examples/user_volumetric_data.py
    # 1. load the data
    print("Loading data")


    datafile = Path("/Users/kaleb/Downloads/T_AVG_brn3c_GFP.tif")
    data = imio.load.load_any(datafile)

    #crop data? --> is this offset in visualization code?
    #here find egde of non-zero values in 3D-space of feature map and set to offset?

    #the data file should be gray scale 0-255
    #data = Volume(np.load('data.npy')) #  this will work



    # 2. aligned the data to the scene's atlas' axes
    print("Transforming data")




    #we want mouse instead of Zfish...
    #The Allen Mouse Brain Atlas at 10, 25, 50 and 100 micron resolutions
    scene = Scene(atlas_name="allen_mouse_10um") #options here are 10 (1320,800,1140), 25 (528,320,456), 50 (264,160,228), or 100 (132,80,114) um
    source_space = AnatomicalSpace(
        "ira"
    )  # for more info: https://docs.brainglobe.info/bg-space/usage # I=inferior, R=right, A=ante (359, 974, 597)

    #see: https://github.com/brainglobe/bg-space
    #need to correct for shape differnces, atlas shape 974,359,597
    #source_space = bgs.SpaceConvention("asl", resolution=(2, 1, 2), offset=(1, 0, 0)) #offset is cropping?
    #target_space = bgs.SpaceConvention("sal", resolution=(1, 1, 1), offset=(0, 0, 2))



    target_space = scene.atlas.space
    transformed_stack = source_space.map_stack_to(target_space, data)
    # 3. create a Volume vedo actor and smooth
    print("Creating volume")
    vol = Volume(transformed_stack, origin=scene.root.origin()).medianSmooth()
    # 4. Extract a surface mesh from the volume actor
    print("Extracting surface")
    SHIFT = [-20, 15, 30]  # fine tune mesh position
    mesh = (vol.isosurface(threshold=20).c(blue_grey).decimate().clean().addPos(*SHIFT))
    # 5. render
    print("Rendering")
    scene.add(mesh)
    #scene.render(zoom=13)
    # 6. save?
    # Create an instance of video maker
    vm = VideoMaker(scene, "/Users/kaleb/Documents/CSHL/2d_3D_linear_reg/", "vid1")
    # make a video with the custom make frame function
    # this just rotates the scene
    vm.make_video(elevation=2, duration=2, fps=15)
    """


# run main program with parser
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)

"""
    #VISUALIZE WITH THIS: https://www.biorxiv.org/content/10.1101/2020.02.23.961748v2.full
    #https://elifesciences.org/articles/65751#content

    # make rotating brain use feature map : d3_array_feature

    #/usr/local/lib/python3.7/site-packages/numpy/core/_asarray.py:136: VisibleDeprecationWarning:
    #Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
    #MovieWriter ffmpeg unavailable; using Pillow instead.


    gif_filename = '3D_brain_rotate'
    # convert d3_array_feature to X Y Z data
    # d3_array_feature=np.transpose(d3_array_feature)
    [X, Y, Z] = np.nonzero(d3_array_feature)

    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D

    # Create a figure and a 3D Axes
    fig = plt.figure()
    ax = Axes3D(fig)

    def init():
        # Plot the surface.
        ax.scatter(X, Y, Z)
        return fig,

    def animate(i):
        # azimuth angle : 0 deg to 360 deg
        ax.view_init(elev=10, azim=i * 4)
        return fig,

    # Animate
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=None, interval=50, blit=True)  # frames = 90
    fn = args.localsubjectpath + gif_filename
    ani.save(fn + '.mp4', writer='ffmpeg', fps=int(1000 / 50))
    # ani.save(fn+'.gif', writer='imagemagick', fps=int(1000/50))



            dim = 0
            [src3d, des3d] = func2d3d.zero_pad(src3d, des3d, dim)
            dim = 1
            [src3d, des3d] = func2d3d.zero_pad(src3d, des3d, dim)




            if args.opticalZ_dir == 'top':
                src3d_edge=src3d[:,:,?]
                des3d_edge=des3d[:,:,?]
            elif args.opticalZ_dir == 'bottom':
                src3d_edge=src3d[:,:,?]
                des3d_edge=des3d[:,:,?]
            else:
                warnings.warn("opticalZ_dir variable not defined correctly")

            # take mean of THIS Zplane and Target Zplane
            A_mean_Z = np.mean(src3d, axis=2, dtype=src3d.dtype)
            B_mean_Z = np.mean(des3d, axis=2, dtype=des3d.dtype)





            # filter and segment
            src3d_feature = func2d3d.preproc(A_mean_Z, args.FFT_max_gaussian)
            des3d_feature = func2d3d.preproc(B_mean_Z, args.FFT_max_gaussian)
            del A_mean_Z, B_mean_Z
            #make plots of Z plane
            plt.figure()
            plt.imshow(src3d[:, :, 1])
            plt.savefig('Z=' + str(Z) + '_Zplane.png', format='png')
            plt.close()
            plt.figure()
            plt.imshow(src3d_feature)
            plt.savefig('Z=' + str(Z) + '_Zplane_feature.png', format='png')
            plt.close()
            # --- Compute the optical flow
            v, u = skimage.registration.optical_flow_tvl1(src3d_feature, des3d_feature)
            # --- Use the estimated optical flow for registration
            nr, nc = src3d_feature.shape
            row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
            #loop comprehension to apply to all images in src3d
            src3d_T = [skimage.transform.warp(src3d[:, :, i], np.array([row_coords + v, col_coords + u]), mode='nearest')
                       for i in range(src3d.shape[2])]
            if np.max(src3d_T) < 1:  # only if max less then one convert to unit
                src3d_T = skimage.img_as_uint(src3d_T, force_copy=False)
            else:
                src3d_T = np.array(src3d_T, dtype=src3d.dtype)
            src3d_T = np.transpose(src3d_T, axes=[1, 2, 0])
            # only transform one feature b/c comrpessed
            src3d_T_feature = skimage.transform.warp(src3d_feature, np.array([row_coords + v, col_coords + u]),
                                                     mode='nearest')
            src3d_T_feature = src3d_T_feature / (np.max(src3d_T_feature))
            src3d_T_feature[src3d_T_feature < 1] = 0
            src3d_T_feature = np.array(src3d_T_feature, dtype=src3d_feature.dtype)
            # error_allZ.append([])
            shiftZ.append([v, u])
            # todo try ants here instead of TvL1 optical flow?
            ## remove?
            if ???
                # TODO try phase corelation
                # shorten variables
                A_mean_Z_Short = A_mean_Z[:5562, :]
                B_mean_Z_Short = B_mean_Z[:, :9092]
                shift_reg, error, diffphase = skimage.registration.phase_cross_correlation(A_mean_Z_padX, B_mean_Z_padX)
                SKI_Trans_allZ = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                                       translation=((shift_reg[1], (shift_reg[0]))))
                # APPLY SHIFT TO WHOLE 3D volume with  LOOP COMPREHENSION
                A_mean_Z_Short_T = skimage.transform.warp(A_mean_Z_Short, SKI_Trans_allZ._inv_matrix)
                # todo try ANTS optical flow method (Gaussian-Regularized Elastic Deformation) this is ElasticSyN
                [A_seg] = func2d3d.preproc(A, args.FFT_max_gaussian)
                [A_seg] = func2d3d.preproc(A, args.FFT_max_gaussian)
                #save feautre map for visualization
                reg_3Dimage_features=
            ## remove above?
       
            if Z == 1:
                # pad in the Y
                dim = 0
                [des3d, src3d_T] = func2d3d.zero_pad(des3d, src3d_T, dim)
                [des3d_feature, src3d_T_feature] = func2d3d.zero_pad(des3d_feature, src3d_T_feature, dim)
                # pad the 3D image in the X
                dim = 1
                [des3d, src3d_T] = func2d3d.zero_pad(des3d, src3d_T, dim)
                [des3d_feature, src3d_T_feature] = func2d3d.zero_pad(des3d_feature, src3d_T_feature, dim)
                src3d_T_feature = np.expand_dims(src3d_T_feature, axis=-1)
                des3d_feature = np.expand_dims(des3d_feature, axis=-1)
                if args.seq_dir == 'top':  # this is top if POS1 is above POS 0 and 'bottom' if POS 1 is below POS 0
                    d3_img = np.concatenate((src3d_T, des3d), axis=2)
                    d3_img_feature = np.concatenate((src3d_T_feature, des3d_feature), axis=2)
                elif args.seq_dir == 'bottom':
                    d3_img = np.concatenate((des3d, src3d_T), axis=2)
                    d3_img_feature = np.concatenate((des3d_feature, src3d_T_feature), axis=2)
                else:
                    warnings.warn("WARNING: seq_dir variable not defined properly: Use 'top' or 'bottom'")
                del des3d, src3d_T, des3d_feature, src3d_T_feature
            else:
                # expand feature map dimensions if needed
                if src3d_T_feature.ndim == 2:
                    src3d_T_feature = np.expand_dims(src3d_T_feature, axis=-1) #expand dim for added unit
                if d3_array_feature.ndim == 2:
                    d3_array_feature = np.expand_dims(d3_array_feature, axis=-1)  # expand dim for added unit
                # pad the 3D image in the Y
                # pad in the Y
                dim = 0
                [d3_array, src3d_T] = func2d3d.zero_pad(d3_array, src3d_T, dim)
                [d3_array_feature, src3d_T_feature] = func2d3d.zero_pad(d3_array_feature, src3d_T_feature, dim)
                # pad the 3D image in the X
                dim = 1
                [d3_array, src3d_T] = func2d3d.zero_pad(d3_array, src3d_T, dim)
                [d3_array_feature, src3d_T_feature] = func2d3d.zero_pad(d3_array_feature, src3d_T_feature, dim)
                if args.seq_dir == 'top':  # this is top if POS1 is above POS 0 and 'bottom' if POS 1 is below POS 0
                    d3_img = np.concatenate((src3d_T, d3_array), axis=2)
                    d3_img_feature = np.concatenate((src3d_T_feature, d3_array_feature), axis=2)
                elif args.seq_dir == 'bottom':
                    d3_img = np.concatenate((d3_array, src3d_T), axis=2)
                    d3_img_feature = np.concatenate((d3_array_feature, src3d_T_feature), axis=2)
                else:
                    warnings.warn("WARNING: seq_dir variable not defined properly: Use 'top' or 'bottom'")
                del src3d_T, src3d_T_feature
            d3_array = d3_img  # Z_img[int(abs(target_overlapY + shift[0])):, :]  # this removes Zeros that are due to shift
            d3_array_feature = d3_img_feature
            Zplane_old = Zplane
            del Zplane, d3_img, d3_img_feature
# from skimage import data
# import plotly
# import plotly.express as px
# import glob
# import IPython.display as IPdisplay
# import pynamical
# import pyglet
import ants
from shutil import copyfile
from PIL import Image

                    # todo try ants ridgid registration? or maks then phase correlation?

                    A_ant = ants.from_numpy(data=filt_A)
                    maskA = ants.get_mask(A_ant) #this does threhsold at mean to max and then does mophaologicalstuff
                    mytx = ants.registration(fixed=maskA, moving=maskB, str="Rigid", initial_transform=None,
                                             outprefix="test", dimension=2)  # “Similarity”
                    # apply transfrom to unsegmented data
                    mywarpedimage = ants.apply_transforms(fixed=A_ant, moving=B_ant,
                                                          transformlist=mytx['fwdtransforms'])
                    # todo try ants ridgid registration here: error: AttributeError: 'numpy.ndarray' object has no attribute 'dimension'
                    # alueError: image must have at least 3 dimensions
                    # convert back to numpy
                    # B_reg=mywarpedimage.numpy()

                    srcY_Ti_mean_overlap_seg = func2d3d.preproc(srcY_Ti_mean_overlap, FFT_max_gaussian)
                    desY_mean_overlap_seg = func2d3d.preproc(desY_mean_overlap, FFT_max_gaussian)
                    shift, error, diffphase = skimage.registration.phase_cross_correlation(desY_mean_overlap_seg,
                                                                                           srcY_Ti_mean_overlap_seg)
                    del srcY_Ti_mean_overlap_seg, desY_mean_overlap_seg
                    
                   

#todo remove after testing
if Z == 0:
    d3_array = Zplane_old
    Zplane_old_mean = np.mean(Zplane_old, axis=2)
    src3d_feature = func2d3d.preproc(Zplane_old_mean, FFT_max_gaussian)
    src3d_feature = np.expand_dims(src3d_feature, axis=2)
    src3d_feature = np.array(src3d_feature)
    src3d_feature = np.concatenate((src3d_feature, src3d_feature), axis=2)
    d3_array_feature = src3d_feature



# STEP 5: move all files in localsubjectpath back to remote area
# redefinie new path
remotesubjectpath_reg = remotesubjectpath + '/registration/'
if server != 'local':
    ssh_con = paramiko.SSHClient()
    ssh_con.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_con.connect(hostname=server, username=user, password=args.password)
    sftp_con = ssh_con.open_sftp()
    sftp_con = ssh.open_sftp()
    #make new folder on remote area
    sftp_con.mkdir(remotesubjectpath_reg)

    #todo add condition if dont have make directory access



    #get local files to copy over
    all_files_in_path = os.listdir(path=localsubjectpath)
    # for all images to save
    for file in all_files_in_path:
        localfilepath3D = localsubjectpath + file
        remotefilepath3D = remotesubjectpath_reg + file
        sftp_con.get(localfilepath3D, remotefilepath3D)
        # delete file locally
        os.remove(localfilepath3D)
        sftp_con.close()
        ssh_con.close()
else:
    if not os.path.isdir(remotesubjectpath_reg):
        os.mkdir(remotesubjectpath_reg)
    all_files_in_path = os.listdir(path=localsubjectpath)
    # for all images to save
    for file in all_files_in_path:
        localfilepath3D = localsubjectpath + file
        remotefilepath3D = remotesubjectpath_reg + file
        copyfile(localfilepath3D, remotefilepath3D)
        # delete folder locally
        os.remove(localfilepath3D)

                    # what if we use https://scikit-image.org/docs/0.18.x/user_guide/tutorial_segmentation.html
                    # then we apply registration?
                    #

                    # todo does this even work? .... we are getting ONLY zero,zero shift here.... for every case. Should be a few pixels right?
                    # todo ALSO for Z we are gettiing zero error? ---> ICP algorithum not working? not really shifting when it should be.... crap
                    # todo i guess set up test case for ICP algoriithum and make sure that works?????

                    # this is less sensative to noise in images..... maybe better here???? \https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html?highlight=lucas%20kanade
                    # todo does this work better??? https://scikit-image.org/docs/stable/auto_examples/registration/plot_opticalflow.html#sphx-glr-auto-examples-registration-plot-opticalflow-py

                    # todo: try this here/ https://simpleelastix.readthedocs.io/RigidRegistration.html

                    # todo https://simpleelastix.readthedocs.io/ParameterMaps.html

                    # todo try: https://pypi.org/project/pystackreg/


#http://blog.mahler83.net/2019/10/rotating-3d-t-sne-animated-gif-scatterplot-with-matplotlib/


X = data.iloc[:,0:-1]
Y = data.iloc[:,-1].astype('int')
no_of_balls = 25
RS=22 #set reproducable random?
X = [random.triangular() for i in range(no_of_balls)]
Y = [random.gauss(0.5, 0.25) for i in range(no_of_balls)]

from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, random_state=RS, perplexity=10)
tsne_fit = tsne.fit_transform(X)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
colors = 'b', 'r'
labels = 'Group1', 'Group2'

for i, c, label in zip(range(len(labels)), colors, labels):
    ax.scatter(tsne_fit[data['Group']==i, 0], tsne_fit[data['Group']==i, 1], tsne_fit[data['Group']==i, 2], s=30, c=c, label=label, alpha=0.5)
fig.legend()


def rotate(angle):
    ax.view_init(azim=angle)


angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save('inhadr_tsne1.gif', writer=animation.PillowWriter(fps=20))












fig = px.imshow(reg_3Dimage_features, animation_frame=0)

fig.layout.annotations[0]['text'] = 'Cell membranes'
fig.layout.annotations[1]['text'] = 'Nuclei'
plotly.io.show(fig)

# visualize the registered image
title_font = pynamical.get_title_font()
label_font = pynamical.get_label_font()

import pynamical
from pynamical import simulate, phase_diagram_3d
import pandas as pd, numpy as np, matplotlib.pyplot as plt, random, glob, os, IPython.display as IPdisplay
from PIL import Image

working_folder = '{}/{}'.format(save_folder, gif_filename)
if not os.path.exists(working_folder):
    os.makedirs(working_folder)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(reg_3D_ptcl[:, 0], reg_3D_ptcl[:, 1], reg_3D_ptcl[:, 2])
# ax.plot3D(reg_3D_ptcl[:,0], reg_3D_ptcl[:,1], reg_3D_ptcl[:,2], 'gray')
# pops = simulate(num_gens=1000, rate_min=3.99, num_rates=1)
# fig, ax = phase_diagram_3d(pops, remove_ticks=False, show=False, save=False)

# create 36 frames for the animated gif
steps = 36
# a viewing perspective is composed of an elevation, distance, and azimuth
# define the range of values we'll cycle through for the distance of the viewing perspective
min_dist = 7.
max_dist = 10.
dist_range = np.arange(min_dist, max_dist, (max_dist - min_dist) / steps)
# define the range of values we'll cycle through for the elevation of the viewing perspective
min_elev = 10.
max_elev = 60.

elev_range = np.arange(max_elev, min_elev, (min_elev - max_elev) / steps)
# now create the individual frames that will be combined later into the animation
for azimuth in range(0, 360, int(360 / steps)):
    # pan down, rotate around, and zoom out
    ax.azim = float(azimuth / 3.)
    ax.elev = elev_range[int(azimuth / (360. / steps))]
    ax.dist = dist_range[int(azimuth / (360. / steps))]
    # set the figure title to the viewing perspective, and save each figure as a .png
    #fig.suptitle('elev={:.1f}, azim={:.1f}, dist={:.1f}'.format(ax.elev, ax.azim, ax.dist))
    plt.savefig('{}/{}/img{:03d}.png'.format(save_folder, gif_filename, azimuth))

# don't display the static plot...
plt.close()









# load all the static images into a list then save as an animated gif
gif_filepath = '{}/{}.gif'.format(save_folder, gif_filename)
images = [Image.open(image) for image in sorted(glob.glob('{}/*.png'.format(working_folder)))]
gif = images[0]
gif.info['duration'] = 75  # milliseconds per frame
gif.info['loop'] = 0  # how many times to loop (0=infinite)
gif.save(fp=gif_filepath, format='gif', save_all=True, append_images=images[1:])
IPdisplay.Image(url=gif_filepath)
# load and show an animated gif file using module pyglet
# pick an animated gif file you have in the working directory
ag_file = gif_filepath
animation = pyglet.resource.animation(ag_file)
sprite = pyglet.sprite.Sprite(animation)
# create a window and set it to the image size
win = pyglet.window.Window(width=sprite.width, height=sprite.height)
# set window background color = r, g, b, alpha
# each value goes from 0.0 to 1.0
green = 0, 1, 0, 1
pyglet.gl.glClearColor(*green)


@win.event
def on_draw():
    win.clear()
    sprite.draw()


pyglet.app.run()





                Aptcl, Bptcl = func2d3d.preproc(A, B, FFT_max_gaussian,median_size)  # this filters data and transfroms to point cloud
                # add Z value here for transformation
                AptclZ = np.ones([Aptcl.shape[0], (Aptcl.shape[1] + 1)])
                AptclZ[:, :-1] = Aptcl
                BptclZ = (np.ones([Bptcl.shape[0], (Bptcl.shape[1] + 1)]))
                BptclZ[:, :-1] = Bptcl
                # STEP 1: register within VOLUME
                pcdA = o3d.geometry.PointCloud()
                pcdA.points = o3d.utility.Vector3dVector(AptclZ)
                pcdB = o3d.geometry.PointCloud()
                pcdB.points = o3d.utility.Vector3dVector(BptclZ)
                threshold = 1  #before we had at most 1.0# Movement range threshold at most move a pixel here???
                # here we are assuming no overlap as initial point
                trans_init_array = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
                # Run icp point to point
                reg_p2p = o3d.pipelines.registration.registration_icp(pcdA, pcdB, threshold, trans_init_array,o3d.pipelines.registration.TransformationEstimationPointToPoint())


# preallocate 3D volume ??? this will not be exacct b/c 15% overlap could be off
# reg_3Dimage=
# get maximum Z
ALLCubeZ = np.zeros(len(cubefolder))
ALLCubeX = np.zeros(len(cubefolder))
ALLCubeY = np.zeros(len(cubefolder))
# get maximum Z value
for i in range(len(cubefolder)):
    allnum = re.sub("[^0-9]", "", cubefolder[i][:])
    ALLCubeZ[i] = int(allnum[:-6])  # HARD CODE Z LOCATION IN FILE NAME
    # for each Z axis (NOT really each 3x3 grid)
    ALLCubeX[i] = int(allnum[-6:-3])  # HARD CODE X LOCATION IN FILE NAME
    ALLCubeY[i] = int(allnum[-2:])  # HARD CODE Y LOCATION IN FILE NAME
MaxCubeX = max(ALLCubeX)
MaxCubeY = max(ALLCubeY)
MaxCubeZ = max(ALLCubeZ)

error_allX = []
error_allY = []
error_allZ = []
# for each Z value
# remove 7 after testing
for Z in range(int(MaxCubeZ) + 1):  # # add 1 b/c range(number) doesnt up though bumber
    # get maximum X and Y in case X/Y plane different size for each Z ---> is this possible?
    ALLCubeX = np.zeros(len(cubefolder))
    ALLCubeY = np.zeros(len(cubefolder))
    for i in range(len(cubefolder)):
        allnum = re.sub("[^0-9]", "", cubefolder[i][:])
        ALLCubeX[i] = int(allnum[-6:-3])  # HARD CODE X LOCATION IN FILE NAME
        ALLCubeY[i] = int(allnum[-2:])  # HARD CODE Y LOCATION IN FILE NAME
    MaxCubeX = max(ALLCubeX)
    MaxCubeY = max(ALLCubeY)
    # for Y
    for Y in range(int(MaxCubeY) + 1):  # step 3: Add another cube to this block until we have 3x3 grid
        # for X
        for X in range(int(MaxCubeX) + 1):
            Zstring = str(Z)
            # here we want to start counting at zero and add zero in front of numbers less then 9
            Ystring = '00' + str(Y)
            Xstring = '00' + str(X)
            cubename = '/Pos' + Zstring + '_' + Xstring + '_' + Ystring + '/'
            # download files in this X Y Z folder --> this is ONE cube
            remotefilepath = remotesubjectpath + cubename
            localfilepath = localsubjectpath  # dont need folder path b/c images removed after downloaded
            # download and define 3D variable for files in this folder
            imarray3D = func2d3d.sshfoldertransfer(server, user, password, remotefilepath, localfilepath, image_type)
            # STEP 1 --> ALIGN SLICES in IMARRAY 3D --> here use ICP point to point with O3D to register
            for i in range(len(imarray3D[1, 1, :]) - 1):
                A = imarray3D[:, :, (i + 1)]  # this is the source
                # for first image I use this as destination
                if i == 0:
                    B = imarray3D[:, :, i]  # this is the  destination
                else:
                    # if any other image get destination from prior AB_img
                    B = AB_img[:, :, i]
                    # define B_whole as the destination already registered
                    AB_whole = AB_img
                Aptcl, Bptcl = func2d3d.preproc(A, B, FFT_max_gaussian,
                                                median_size)  # this filters data and transfroms to point cloud
                # add Z value here for transformation
                AptclZ = np.ones([Aptcl.shape[0], (Aptcl.shape[1] + 1)])
                AptclZ[:, :-1] = Aptcl
                BptclZ = (np.ones([Bptcl.shape[0], (Bptcl.shape[1] + 1)])) * 2
                BptclZ[:, :-1] = Bptcl
                # STEP 1: register within VOLUME
                pcdA = o3d.geometry.PointCloud()
                pcdA.points = o3d.utility.Vector3dVector(AptclZ)
                pcdB = o3d.geometry.PointCloud()
                pcdB.points = o3d.utility.Vector3dVector(BptclZ)
                threshold = 1.0  # Movement range threshold --> chnage to image size???
                # here we are assuming no overlap as initial point
                trans_init_array = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
                # Run icp point to point
                reg_p2p = o3d.pipelines.registration.registration_icp(pcdA, pcdB, threshold, trans_init_array,
                                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
                del pcdA, pcdB
                # this transforms data from A into B space (USE IMAGES NOT point clouds) --> only use 2D part
                A_T = skimage.transform.warp(A, reg_p2p.transformation[:3, :3])
                print(reg_p2p.transformation[:3, :3])
                del reg_p2p
                # combine A and B into one matrix
                if i == 0:
                    # here we need to add Z value so source and destiantion values given Z
                    AB_img = np.stack((B, A_T), axis=-1)  # add axis to the end
                else:
                    # for other images add B_whole + B_xy + A_TrC
                    A_T = np.expand_dims(A_T, axis=-1)
                    AB_img = np.concatenate((AB_whole, A_T), axis=2)
            # STEP 2 Y line: Stitch together images with first X initial guess and O3D point to point ICP registration (1st make Y lines, then combine Y lines
            srcY = AB_img  # define source image
            if X == 0:
                # DON'T STITCH Because nothing to stitch
                AB_img_old = AB_img
                del AB_img
            else:
                desY = AB_img_old  # define destination image
                if X == 1 and Y == 0 and Z == 0:  # calculate the initial stitch level ONLY for first overlap
                    diroverlap = 'right'
                    [target_overlapX] = func2d3d.findoverlay(srcY, desY, diroverlap)
                    # calculate initial transformation from overlap
                    trans_init_stitchX = func2d3d.initial_transform(target_overlapX, diroverlap)
                # calculate shift on MEAN image --> apply to whole image this helps with noisy images
                srcY_mean = np.mean(srcY, axis=2)
                desY_mean = np.mean(desY, axis=2)
                shiftXi = -(srcY.shape[1] - abs(target_overlapX))
                SKI_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                                  translation=(shiftXi, 0))
                srcY_Ti = skimage.transform.warp(srcY_mean, SKI_Trans._inv_matrix)
                # get phase correlation for small local change
                desY_mean_overlap = desY_mean[:, :(target_overlapX)]
                srcY_Ti_mean_overlap = srcY_Ti[:, :(target_overlapX)]
                shift, error, diffphase = skimage.registration.phase_cross_correlation(desY_mean_overlap,
                                                                                       srcY_Ti_mean_overlap)
                error_allX.append(error)

                # calculate OVERALL SHIFT from inital guess + phase correlations
                SKI_Trans_all = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0, translation=(
                    (target_overlapX + shift[1]), shift[0]))
                # APPLY SHIFT TO WHOLE 3D volume with  LOOP COMPREHENSION
                srcY_T = [skimage.transform.warp(srcY[:, :, i], SKI_Trans_all._inv_matrix) for i in
                          range(srcY.shape[2])]
                srcY_T = np.array(srcY_T)
                # rearanage
                srcY_T_re = np.transpose(srcY_T, axes=[1, 2, 0])
                del srcY_T, SKI_Trans, srcY_Ti
                if X == 1:
                    # concat 3D volumes together
                    Y_img = np.concatenate((srcY_T_re, desY), axis=1)
                else:
                    Y_img = np.concatenate((srcY_T_re, Yline),
                                           axis=1)  # need to match in other directions.. so keep Zeros in Y direction
                # define YLine as Y_img without the shifted zero values
                Yline = Y_img[:, int(abs(
                    target_overlapX + shift[1])):]  # need to match in other directions.. so keep Zeros in Y direction
                testABOLD = AB_img_old  #todo used for testing
                testAB = AB_img  #todo used for testing
                AB_img_old = AB_img
                del AB_img

        # STEP 2 Z plane: Stitch together images with first X initial guess and O3D point to point ICP registration (1st make Y lines, then combine Y lines
        srcZ = Yline  # this is the Y to add to rest
        if Y == 0:
            # DON'T STITCH Because nothing to stitch
            Yline_old = Yline
            del Yline
        else:
            desZ = Yline_old  # define destination image
            # pad in the Y
            dim = 1
            [srcZ, desZ] = func2d3d.zero_pad(srcZ, desZ, dim)
            if Y == 1 and Z == 0:  # calculate the initial stitch level ONLY for first overlap
                diroverlap = 'down'
                [target_overlapY] = func2d3d.findoverlay(srcZ, desZ, diroverlap)
                # calculate initial transformation from overlap
                trans_init_stitchY = func2d3d.initial_transform(target_overlapY, diroverlap)

                #NOTE WHEN INDIVIDUAL added we get 200 something as target_overlapY .....
                # TODO rearrage so bulid out 1 cube at a time --> this way each cube has its own Y and X registration
                # TOdo above is worth is b/c X error is 0.05 and Y error is 0.1 (error 0f 0.8 due to mistake in code.....
                # TODO why is Y error 0.1 to 0.2 while X error is 0.05 ????
                # here we haave AB_im111 aanad ABim112 to test
                #note when i do one cube Y makes sence BUT NOT for Y plane.....
                #UNSURE WHY this would happen.... i guess if X is wrong?
                #if X iss wrong then Y line --> all crap b/c OFF in X (on a cube by cube level? or on a whole level?)


                # i guess could add each cube one at a time? instead of in Y line? ---> but this shouldnt make a difference
                # maybe b/c...... "overlap" images are too large? .... check images?
                # the target_overlapY value is 76?? ... should be 200 something.....
            # calculate shift on MEAN image --> apply to whole image this helps with noisy images
            srcZ_mean = np.mean(srcZ, axis=2)
            desZ_mean = np.mean(desZ, axis=2)
            shiftYi = -(srcY.shape[0] - abs(target_overlapY))
            SKI_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                              translation=(0, shiftYi))
            srcZ_Ti = skimage.transform.warp(srcZ_mean, SKI_Trans._inv_matrix)
            # get phase correlation for small local change
            desZ_mean_overlap = desZ_mean[:target_overlapY, :]
            srcZ_Ti_mean_overlap = srcZ_Ti[:target_overlapY, :]
            shift, error, diffphase = skimage.registration.phase_cross_correlation(desZ_mean_overlap,
                                                                                   srcZ_Ti_mean_overlap)
            error_allY.append(error)
            # calculate OVERALL SHIFT from inital guess + phase correlations
            SKI_Trans_all = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0, translation=(
                (shift[0], target_overlapY + shift[1])))
            # APPLY SHIFT TO WHOLE 3D volume with  LOOP COMPREHENSION
            srcZ_T = [skimage.transform.warp(srcZ[:, :, i], SKI_Trans_all._inv_matrix) for i in
                      range(srcZ.shape[2])]
            srcZ_T = np.array(srcZ_T)
            # rearanage
            srcZ_T_re = np.transpose(srcZ_T, axes=[1, 2, 0])
            del srcZ_T, SKI_Trans, srcZ_Ti
            if Y == 1:
                Z_img = np.concatenate((srcZ_T_re, desZ), axis=0)
            else:
                # pad the Zplane in the X
                dim = 1
                [Zplane_pad, srcZ_T_re_pad] = func2d3d.zero_pad(Zplane, srcZ_T_re, dim)
                Z_img = np.concatenate((srcZ_T_re_pad, Zplane_pad), axis=0)  # need to match in other directions
            Zplane = Z_img[int(abs(target_overlapY + shift[1])):, :]  # this removes Zeros that are due to shift
            Yline_old = Yline
            del Yline
    # Step 3 combine Z plane ---> use ICP without initial guess (maybe globally registration? )
    # WHAT If foR z we take mean in optical Z to get feautre map
    # we use this feautre map to calcualte translation
    # apply trnaslation to orginal images so we done loose any resolution but mean out noise (both salt/peper and low freq shift)
    src3d = Zplane  # this is the Y to add to rest
    if Z == 0:
        # DON'T STITCH Because nothing to stitch
        Zplane_old = Zplane
        del Zplane
    else:
        des3d = Zplane_old  # define destination image
        # take mean of THIS Zplane and Target Zplane
        A_mean_Z = np.mean(src3d, axis=2)
        B_mean_Z = np.mean(des3d, axis=2)
        Aimg = np.uint8(A_mean_Z)
        edgesA3 = func2d3d.auto_canny(Aimg)
        Bimg = np.uint8(B_mean_Z)
        edgesB3 = func2d3d.auto_canny(Bimg)
        # convert to point cloud --> really this list of points by xyz for nonzero data
        Aptcl3d = func2d3d.image_to_XYZ(edgesA3)
        Bptcl3d = func2d3d.image_to_XYZ(edgesB3)
        # add Z value here for transformation
        Aptcl3dZ = np.ones([Aptcl3d.shape[0], (Aptcl3d.shape[1] + 1)])
        Aptcl3dZ[:, :-1] = Aptcl3d
        Bptcl3dZ = (np.ones([Bptcl3d.shape[0], (Bptcl3d.shape[1] + 1)])) * 2
        Bptcl3dZ[:, :-1] = Bptcl3d
        pcdA = o3d.geometry.PointCloud()
        pcdA.points = o3d.utility.Vector3dVector(Aptcl3dZ)
        pcdB = o3d.geometry.PointCloud()
        pcdB.points = o3d.utility.Vector3dVector(Bptcl3dZ)
        # global register Z plane (with ICP) --> do we want this?
        trans_init_3d = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])  # SO NO initial transform
        # Local register Z plane (with ICP point to plane) --> point to plane helps flat regions slide along each other
        threshold = 1.0  # Movement range threshold
        # Run icp point to poiint? --> point to plane needs normals and only increased speed not accuracy
        reg_p2p = o3d.pipelines.registration.registration_icp(pcdA, pcdB, threshold, trans_init_3d,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint())
        error_allZ.append(reg_p2p.inlier_rmse)
        del pcdA, pcdB
        # this transforms data from A into B space (USE IMAGES NOT point clouds) --> this applies 2D to all 3D images
        src3d_T = skimage.transform.warp(src3d, reg_p2p.transformation[:3, :3])
        del reg_p2p
        if Z == 1:
            # pad in the Y
            dim = 0
            [des3d_padY, src3d_T_padY] = func2d3d.zero_pad(des3d, src3d_T, dim)
            # pad the 3D image in the X
            dim = 1
            [des3d_padYX, src3d_T_padYX] = func2d3d.zero_pad(des3d_padY, src3d_T_padY, dim)
            d3_img = np.concatenate((src3d_T_padYX, des3d_padYX), axis=2)
        else:
            src3d_T = np.expand_dims(src3d_T, axis=-1)
            # pad the 3D image in the Y
            # pad in the Y
            dim = 0
            [d3_arrayY, src3d_T_padY] = func2d3d.zero_pad(d3_array, src3d_T, dim)
            # pad the 3D image in the X
            dim = 1
            [d3_arrayYX, src3d_T_padYX] = func2d3d.zero_pad(d3_arrayY, src3d_T_padY, dim)
            d3_img = np.concatenate((src3d_T_padYX, d3_arrayYX), axis=2)
        d3_array = d3_img
        Zplane_old = Zplane
        del Zplane





                if Zplane.shape[1] > srcZ_T_re.shape[1]:
                    srcZ_T_re_pad = np.zeros(
                        Zplane.shape)  # * np.mean(srcZ_T_re) # use zero not mean b/c this is what X shift uses when translated
                    srcZ_T_re_pad[:srcZ_T_re.shape[0], (Zplane.shape[1] - srcZ_T_re.shape[1]):] = srcZ_T_re
                    srcZ_T_re = srcZ_T_re_pad
                if srcZ_T_re.shape[1] > Zplane.shape[1]:
                    Zplane_pad = np.zeros([Zplane.shape[0], srcZ_T_re.shape[1], Zplane.shape[2]])  # * np.mean(Zplane)  # use zero not mean b/c this is what X shift uses when translated
                    Zplane_pad[:Zplane.shape[0], (srcZ_T_re.shape[1] - Zplane.shape[1]):] = Zplane
                    Zplane = Zplane_pad
                    




            # zero padd shorter Yline --> from differences in X shifts
            if srcZ.shape[1] > desZ.shape[1]:
                desZ_pad = np.ones(srcZ.shape) * np.mean(desZ)
                desZ_pad[:desZ.shape[0], (srcZ.shape[1] - desZ.shape[1]):] = desZ
                desZ = desZ_pad
            if desZ.shape[1] > srcZ.shape[1]:
                srcZ_pad = np.ones(desZ.shape) * np.mean(srcZ)
                srcZ_pad[:srcZ.shape[0], (desZ.shape[1] - srcZ.shape[1]):] = srcZ
                srcZ = srcZ_pad


        #for testing:
        #imarray3D_ALL=np.zeros([MaxCubeX,MaxCubeY,MaxCubeZ,2048, 2048, 41])
        imarray3D_ALLXYZ=np.zeros([(int(MaxCubeX+1) * 2048), (int(MaxCubeY+1) * 2048),(int(MaxCubeZ+1) * 41)])
        Z_start = 0
        # EXTRA
        # STEP TO CCONVERT TO POINT CLOUD (number pooints, number of dimensions)
        # MAYBE FOR Z REGISTRATION USE THIS RIDGID regsion method... HOW IS THIS DONE??? --> KEEP ICP for OVERLAP stitching ??? (still need point cloud here)
        # mov = A
        # ref = B
        # Rigid Body transformation https://pypi.org/project/pystackreg/
        # sr = StackReg(StackReg.RIGID_BODY)
        # out_rot = sr.register_transform(ref, mov)

                        # imarray3D_ALL[X,Y,Z,:,:,:]=imarray3D  #here try and get matrix of ALL
                        # or what if we shift X by 2048*X, Y by 2048*Y and Z by Z*41
                        imarray3D_ALLXYZ[X_start:(X_start + 2048), Y_start: (Y_start + 2048), Z_start: (
                            Z_start + 41), ] = imarray3D
                X_start = X_start + 2048
                del imarray3D
                Y_start = Y_start + 2048
                Z_start = Z_start + 41

# GLOBAL registration between two or more volumes

#for first and only first image get % overlap and apply this overlap to rest of images as first guess
#find image overlap by full search in X direction (sift +1 pixel in a for loop and find max image similarity)
#do this for whole stack of 41 images.... should be same overlap +/- error ---> if not return an error


Aptcl, Bptcl = func2d3d.preproc(A, B, window_size)
pcdA = o3d.geometry.PointCloud()
pcdA.points = o3d.utility.Vector3dVector(Aptcl)
pcdB = o3d.geometry.PointCloud()
pcdB.points = o3d.utility.Vector3dVector(Bptcl)
# global registration
#what if we make are own algorithum, maximize the image similairty and only shift in direction that makes sence --> would be slow and proabbly full search of solution sapce in this limited space



#global registration --> SVD?? https://www.programmersought.com/article/17864660558/

import cv2
import open3d as o3d
from matplotlib import pyplot as pyplot
import numpy as np
import copy
import scipy
from scipy import spatial
import random
import sys
import math


# Kabsch Algorithm
def compute_transformation(source, target):
    # Normalization
    number = len(source)
    # the centroid of source points
    cs = np.zeros((3, 1))
    # the centroid of target points
    ct = copy.deepcopy(cs)
    cs[0] = np.mean(source[:][0]);
    cs[1] = np.mean(source[:][1]);
    cs[2] = np.mean(source[:][2])
    ct[0] = np.mean(target[:][0]);
    cs[1] = np.mean(target[:][1]);
    cs[2] = np.mean(target[:][2])
    # covariance matrix
    cov = np.zeros((3, 3))
    # translate the centroids of both models to the origin of the coordinate system (0,0,0)
    # subtract from each point coordinates the coordinates of its corresponding centroid
    for i in range(number):
        sources = source[i].reshape(-1, 1) - cs
        targets = target[i].reshape(-1, 1) - ct
        cov = cov + np.dot(sources, np.transpose(targets))
    # SVD (singular values decomposition)
    u, w, v = np.linalg.svd(cov)
    # rotation matrix
    R = np.dot(u, np.transpose(v))
    # Transformation vector
    T = ct - np.dot(R, cs)
    return R, T


# compute the transformed points from source to target based on the R/T found in Kabsch Algorithm
def _transform(source, R, T):
    points = []
    for point in source:
        points.append(np.dot(R, point.reshape(-1, 1) + T))
    return points


# compute the root mean square error between source and target
def compute_rmse(source, target, R, T):
    rmse = 0
    number = len(target)
    points = _transform(source, R, T)
    for i in range(number):
        error = target[i].reshape(-1, 1) - points[i]
        rmse = rmse + math.sqrt(error[0] ** 2 + error[1] ** 2 + error[2] ** 2)
    return rmse


def draw_registrations(source, target, transformation=None, recolor=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if (recolor):  # recolor the points
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if (transformation is not None):  # transforma source to targets
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pc2array(pointcloud):
    return np.asarray(pointcloud.points)


def registration_RANSAC(source, target, source_feature, target_feature, ransac_n=3,
                        max_iteration=10000, max_validation=100):
    # the intention of RANSAC is to get the optimal transformation between the source and target point cloud
    s = pc2array(source)  # (4760,3)
    t = pc2array(target)
    # source features (33,4760)
    sf = np.transpose(source_feature.data)
    tf = np.transpose(target_feature.data)
    # create a KD tree
    tree = spatial.KDTree(tf)
    corres_stock = tree.query(sf)[1]
    for i in range(max_iteration):
        # take ransac_n points randomly
        idx = [random.randint(0, s.shape[0] - 1) for j in range(ransac_n)]
        corres_idx = corres_stock[idx]
        source_point = s[idx, ...]
        target_point = t[corres_idx, ...]
        # estimate transformation
        # use Kabsch Algorithm
        R, T = compute_transformation(source_point, target_point)
        # calculate rmse for all points
        source_point = s
        target_point = t[corres_stock, ...]
        rmse = compute_rmse(source_point, target_point, R, T)
        # compare rmse and optimal rmse and then store the smaller one as optimal values
        if not i:
            opt_rmse = rmse
            opt_R = R
            opt_T = T
        else:
            if rmse < opt_rmse:
                opt_rmse = rmse
                opt_R = R
                opt_T = T
    return opt_R, opt_T


# used for downsampling
voxel_size = 0.05


# this is to get the fpfh features, just call the library
def get_fpfh(cp):
    cp = cp.voxel_down_sample(voxel_size)
    cp.estimate_normals()
    return cp, o3d.pipelines.registration.compute_fpfh_feature(cp,
                                                     o3d.geometry.KDTreeSearchParamHybrid(radius=5,
                                                                                          max_nn=100))


#GLOBAL registration: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
voxel_size=10
pcdA.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=(voxel_size * 2), max_nn=30))
pcdB.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=(voxel_size * 2), max_nn=30))
pcd_fpfhA = o3d.pipelines.registration.compute_fpfh_feature(pcdA,o3d.geometry.KDTreeSearchParamHybrid(radius=(voxel_size * 5), max_nn=100))
pcd_fpfhB = o3d.pipelines.registration.compute_fpfh_feature(pcdB,o3d.geometry.KDTreeSearchParamHybrid(radius=(voxel_size * 5),max_nn=100))
distance_threshold = voxel_size * 1.5

result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcdA, pcdB, pcd_fpfhA, pcd_fpfhB, True,distance_threshold,o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
#problem here is that correspondence_set size of 0
func2d3d.draw_registration_result(pcdA, pcdB, result.transformation)

        draw_registration_result(source_down, target_down, result_ransac.transformation)


                # convert between point cloud and image: http://www.open3d.org/docs/0.9.0/tutorial/Basic/working_with_numpy.html

                # MAKE SURE WE TRANSFROM IMAGE NOT POINT CLOUD BC DONT WANT ERROR FROM EDGE DETECTION

                # this has ICP registration and is FAST b/c its in C

                # this fins the MSE oof overlapped area and % overlap: open3d.pipelines.registration.RegistrationResult
                # open3d.pipelines.registration.evaluate_registration

                # robust kernal help with outlier detection (might help werid results)

                # first use global regisatration --> then use local registraion ICP (or just give starting position?)  RANSAC

                # what if we SKIP feature detection ans use COLOR registratioin?



                T, distances, i = func2d3d.icp(Aptcl, Bptcl, init_pose=None, max_iterations=200,tolerance=0.001)  # source =A, destination =B, init_pose appied to source (A)
            #take x,y data and covert to x,y,color
            # convert format into XY color format --> this is poiint cloud format
            A_xy = func2d3d.img_to_XYC(A)
            # convert B into XY (this is destination so this defines xyz used in A_T
            B_xy = func2d3d.img_to_XYC(B)
            #Apply transform to A and B (before preprocessing) --> see https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/
            dim_num = A.ndim
            A_TrC=func2d3d.transform_data(A_xy, T, dim_num)
            #combine A and B into one matrix
            #for first image only B_xy + A_TrC
            if i == 1:

                #here we need to add Z value so source and destiantion values given Z

                AB_xy=np.concatenate((B_xy, A_TrC))
            else:
            #for other images add B_whole + B_xy + A_TrC
                AB_xy_part = np.concatenate((B_xy, A_TrC))
                AB_xy= np.concatenate((AB_xy_part, B_whole))


            # todo before line below add Z value if concat 2D values

            #find the overlap and get average of the combined image
            [AB_xy_avg,overlap_perc,image_sim] = func2d3d.image_overlap(AB_xy)


            # convert combined matrix back into an image --> done here so same XY or XYZ across image matrix?
            [AB_img] =func2d3d.XYC_to_img(AB_xy_avg)

            np_image = np.array(o3d_image) #https://github.com/intel-isl/Open3D/issues/957

        #now start to combine 3D slices and get overlap for each X in a given Y
        #if X =1 #this is first set in row

        #else

            #align in 3D

            #align in 2D

    # now start to combine 3D slices and get overlap for each Y in a given Z
    # if Y =1

    # else
            # align in 3D only

# now start to combine 3D slices and get overlap for each Z in whole image
# if Z =1 #here

# else
            # align in 3D only


                A_T = np.dot(T, A)
                #rows,cols = img.shape
                #dst = cv2.warpAffine(img,M,(cols,rows))
                #define new variable with transformed data
                #keras.preprocessing.image.apply_transform
                ImageDataGenerator.apply_transform(A,T,channel_axis=0,fill_mode='nearest',cval=0.0)
                tf.keras.preprocessing.image.apply_transform(

                #this transfrom is 2049 by 2049, while image is 2048 by 2048
                #issure Here is what if we SHIFT by a pixel, so now
                #redefine imarray with registared data
                imarray3D[:,:,i]=B

                # Make C a homogeneous representation of B.... is this basically zero paddiing??? for transformatiion shiift?
                newddata = np.ones((len(A), 4)) #4 here b/c x,y,z plus homogenous represention
                newddata[:, 0:3] = A #this is saying x,y,z is A, the 4th demsion is ones for the homegenous repersentation


                newddata=np.dot(T, newddata.T).T  # this transforms A... issues is A and T are not equal 2048 vs 2049




                imarray3D[:,:,(i+1)]=newddata
                T_all[i]=T
                multi_slice_viewer(imarray3D)
                del A, B, T
            #Step 1.5 for this slice to volume registration no overlap --> measure of sucess ? --> Dice?


            #want to ADD new cube to already registarted set of cubes --> this way new cube is registared in all diredctioins while set of cube registration is held constant
            #for each pair --> 3D registration then 2D registration




           # step 1: align cube 3D volume with neighbor 3D volume --> this should make sure Z axis aligned within cube


            # if first cube of this Z plane
            if X==1 && Y== 1
              #skip registration here, just define plane with loaded data
              imarray3D_whole=imarray3D
            else
                # src = np.dot(init_pose, src)
                A= imarray3D # source
                B= imarray3D_whole  # distination
                src_init_pose =   # here is the transfrom to apply to source to get to desitination, for this is sould be none
                T,distances,i=func2d3d.icp(A, B, init_pose=src_init_pose, max_iterations=200, tolerance=0.001)

                #apply transfrom to source and add to whole
                np.dot(T, A.T).T  # this transforms A



                #Step 2: for each Z within this cube align to neighbor --> this should align within cube non-Linearities
      
                for ? in imarray3D
                    A= imarray3D and
                    B=

                    #here we DO have a rough inital guess.... ccan we incorperate this here?
                    #inital guess should be so located next to image.. NOT 15% mark

                    T, distances, i = func2d3d.icp(A, B, init_pose=None, max_iterations=200, tolerance=0.001)
                    regIM1 =
                    regIM2 =


                    #Step 2.5: combine transfomred data into one grid

                    # step 5 find %overlap and overlap image similairty --> if not 15% overlap or dissimilar image then error in registration

                    im1_overlap, percent_overlap, image_similarity = func2d3d.measurements_of_se(im1, im2, T)

                    #define overlap area .....


                    # remove overlap from 3d array

                    #append transformed data to 3d array
                    reg_3Dimage = array.extend(iterable)



                    #do we want to visualize the im1, im2 and registraed image for each regisitration?


           # Step 2: for each Z layer grid align to neighbor Z layer grid ---> again no overlap here? (could there be?)

            #IF first Z skip this step and just define
            if Z ==1
            #skip registratiion

            else




#EXAMPLE:
# Define empty matrices of shape no_of_matches * 2.
#p1 = np.zeros((no_of_matches, 2))
#p2 = np.zeros((no_of_matches, 2))

#for i in range(len(matches)):
#    p1[i, :] = kp1[matches[i].queryIdx].pt
#    p2[i, :] = kp2[matches[i].trainIdx].pt








"""
