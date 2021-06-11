# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 24th 2020
2d-to-3d
@author: Kaleb Vinehout

This code combines 2D data into 3d volume, functions for this code
"""
import os
import re
import warnings
from shutil import copyfile
# import ants
import matplotlib.pyplot as plt
import numpy as np
# import modules
import paramiko
import skimage
import skimage.registration
import skimage.segmentation
import skimage.transform
from PIL import Image, ImageOps, ImageChops
from skimage import filters
from skimage import restoration
import cv2
import imreg_dft as ird
import math
from skimage.restoration import (calibrate_denoiser, denoise_wavelet, denoise_tv_chambolle, denoise_nl_means,
                                 estimate_sigma)
from skimage.util import img_as_float, random_noise
from skimage.restoration.j_invariant import _invariant_denoise
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, binary_dilation, disk, square, area_opening, opening
from skimage.color import label2rgb
import itk
from skimage.filters import threshold_multiotsu
from PIL import Image
import sys

import matplotlib
import scipy
import operator
from skimage.color import rgb2gray
from functools import partial
from matplotlib import gridspec
from skimage.data import chelsea, hubble_deep_field
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.neighbors import NearestNeighbors


# import ICP


# todo add unit tests after each function???


# define file load function
def sshfoldertransfer(server, user, password, remotefilepath, localfilepath, image_type, opticalZ_dir):
    """Connects to host and searches for files matching file_pattern
    in remote_path. Downloads all matches to 'local_path'"""
    if server != 'local':
        # Opening ssh and ftp
        ssh_con = paramiko.SSHClient()
        ssh_con.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_con.connect(hostname=server, username=user, password=password)
        sftp_con = ssh_con.open_sftp()
        # Finding files
        all_files_in_path_raw = sftp_con.listdir(path=remotefilepath)
    # make option if NOT on server...aka on same computer...
    else:
        all_files_in_path_raw = os.listdir(path=remotefilepath)
    # sort files
    all_files_in_path = sorted(all_files_in_path_raw)
    r = re.compile(image_type)  # this limits to only files with Il-A in the name, so EXCLUDING P-DIC imaging
    files = list(filter(r.search, all_files_in_path))
    # preallocate
    # load one file to get size and data type
    file_remote = remotefilepath + files[0]
    file_local = localfilepath + files[0]
    if server != 'local':
        sftp_con.get(file_remote, file_local)
    else:
        copyfile(file_remote, file_local)
    im = Image.open(file_local)
    imarray = np.array(im)
    imarray3D = np.zeros((imarray.shape[0], imarray.shape[1], len(files)),
                         dtype=imarray.dtype)  # HARD CODE SIZE OF IMAGES
    # Download files
    print("Loading {} files from {}.".format(len(files), remotefilepath))
    countZ = 0
    for file in files:
        file_remote = remotefilepath + file
        file_local = localfilepath + file
        if server != 'local':
            sftp_con.get(file_remote, file_local)
        else:
            copyfile(file_remote, file_local)
        # Open the tiff image and save as 3D image
        im = Image.open(file_local)
        imarray = np.array(im)
        # get numbers after image_type and only numbers after this are Zvalue
        Zvalue = re.findall("\d+", file[file.index(image_type) + len(image_type):])[0]
        # remove zeros at begning of string number here.... and convert string to decimal
        ZvalueNum = int(Zvalue)
        # check Zvalue from label match sorted count
        if countZ != ZvalueNum:
            warnings.warn(
                message="WARNING: Check .tif file named correctly, from name we get {}, while this is the {} image".format(
                    countZ, Zvalue))
        if opticalZ_dir == 'top':
            ZvalueNum = ZvalueNum
        elif opticalZ_dir == 'bottom':
            ZvalueNum = -(ZvalueNum + 1)
        else:
            warnings.warn("opticalZ_dir variable not defined correctly")
        imarray3D[:, :, ZvalueNum] = imarray
        del imarray
        # after image is loaded in variable delete image?
        os.remove(file_local)
        countZ = countZ + 1
    if server != 'local':
        sftp_con.close()
        ssh_con.close()
    # return the copied then loaded image this is a 3D cube
    return imarray3D


# helper function for multi_slice_viewer
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


# helper function for multi_slice_viewer
def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


# helper function for multi_slice_viewer
def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


# helper function for multi_slice_viewer
def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def multi_slice_viewer(volume):
    """
    Visualizes the 3D registared brain
        Args:
        -	reg_3dImage: Numpy array 3D registered image

        Returns:
        -	reg_3d_movie: movie of 3D registered image

    """
    # transpose data so in Z, X , Y
    volumeT = volume.T
    remove_keymap_conflicts({'j', 'k'})  # so the j and k keys move forward and back in slices
    fig, ax = plt.subplots()
    ax.volume = volumeT
    ax.index = volumeT.shape[0] // 2
    ax.imshow(volumeT[ax.index], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)

    return


def remove_large_obj(A, area_thres, int_thres, FFT_max_gaussian, high_thres_dec):
    """
        This removes large and high intensity objects from image
        Args:
        -A Numpy array to be searched for large high intensity regions
        - area_thres: this the square root of the area minimum to look for
        -int_thres:  this the intensity to look for
        -FFT_max_gaussian: to filter low frequency out of image
        -high_thres_dec: the amount to increase boundary by to insure full inclosure of large object
        Returns:
        -mask_A: This is the image with the large birght area masked out

    """
    ori_A = np.copy(A)
    filt_A = skimage.filters.difference_of_gaussians(A, 1, FFT_max_gaussian)
    # denoise with wavlet this might not have much of an effect
    denoise = skimage.restoration.denoise_wavelet(filt_A, multichannel=False, rescale_sigma=True)
    # set stuff below mean to zero? --> for some reason this works best
    low_values_flags = denoise < np.median(denoise)
    denoise[low_values_flags] = 0
    image = denoise
    # apply threshold
    thresh = 2 * np.median(denoise)  # 1000 #threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    dialted = binary_dilation(cleared, selem=None)
    # dialted = binary_dilation(dialted1, selem=square(300))
    # label image regions
    label_image = label(dialted)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=A, bg_label=0)
    for region in regionprops(label_image, intensity_image=A):
        # take regions with large enough areas
        if region.area >= area_thres ** 2 and region.mean_intensity >= int_thres:  # mean intensity not max intensity makes more sence here
            warnings.warn(message='Large high intensity artifact found, removing artifact')
            # define this as a mask to REMOVE from image
            # for this box set value to zero? mean? median?
            (min_row, min_col, max_row, max_col) = region.bbox
            high_values_flags = np.array(np.zeros(A.shape), dtype=bool)
            # add 10% to boundry box
            buffer_row = (max_row - min_row) * (high_thres_dec * 2)
            buffer_col = (max_col - min_col) * (high_thres_dec * 2)
            min_row_shift = round(min_row - buffer_row)
            max_row_shift = round(max_row + buffer_row)
            min_col_shift = round(min_col - buffer_col)
            max_col_shift = round(max_col + buffer_col)
            if min_col_shift < 0:
                min_col_shift = 0
            if min_row_shift < 0:
                min_row_shift = 0
            if max_col_shift > A.shape[1]:
                max_col_shift = A.shape[1]
            if max_row_shift > A.shape[0]:
                max_row_shift = A.shape[0]
            high_values_flags[min_row_shift:max_row_shift, min_col_shift:max_col_shift] = True
            A[high_values_flags] = np.median(A)
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    mask_A_raw = A
    mask_A = np.array(mask_A_raw)
    # remove extra dimensions
    mask_A = np.squeeze(mask_A)
    if mask_A.ndim > 2:
        warnings.warn(message='wrong dimensions in high intensity image. Size found of {}'.format(mask_A.shape))
    # make data type same as input
    mask_A = mask_A.astype(ori_A.dtype)
    return mask_A


def denoise_noise2self(A, FFT_max_gaussian, high_thres):
    # https://github.com/NVlabs/noise2noise
    # try noise2noise traiing? https://github.com/czbiohub/noise2self or https://github.com/NVlabs/selfsupervised-denoising
    """
    noisy_image = imarray3D_mean_512
    import sys
    sys.path.append("..")
    from util import show, plot_images, plot_tensors
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import Dataset
    mnist_train = MNIST('../data/MNIST', download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]), train=True)
    mnist_test = MNIST('../data/MNIST', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), train=False)
    from torch import randn
    def add_noise(img):
        return img + randn(img.size()) * 0.4

    class SyntheticNoiseDataset(Dataset):
        def __init__(self, data, mode='train'):
            self.mode = mode
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            img = self.data[index][0]
            return add_noise(img), img

    noisy_mnist_train = SyntheticNoiseDataset(mnist_train, 'train')
    noisy_mnist_test = SyntheticNoiseDataset(mnist_test, 'test')

    noisy, clean = noisy_mnist_train[0]
    plot_tensors([noisy[0], clean[0]], ['Noisy Image', 'Clean Image'])
    from mask import Masker
    masker = Masker(width=4, mode='interpolate')
    net_input, mask = masker.mask(noisy.unsqueeze(0), 0)
    plot_tensors([mask, noisy[0], net_input[0], net_input[0] - noisy[0]],
                 ["Mask", "Noisy Image", "Neural Net Input", "Difference"])
    from models.babyunet import BabyUnet
    model = BabyUnet()

    from torch.nn import MSELoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    data_loader = DataLoader(noisy_mnist_train, batch_size=32, shuffle=True)



    for i, batch in enumerate(data_loader):
        noisy_images, clean_images = batch

        net_input, mask = masker.mask(noisy_images, i)
        net_output = model(net_input)

        loss = loss_function(net_output * mask, noisy_images * mask)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            print("Loss (", i, "): \t", round(loss.item(), 4))

        if i == 100:
            break

    test_data_loader = DataLoader(noisy_mnist_test,
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=3)
    i, test_batch = next(enumerate(test_data_loader))
    noisy, clean = test_batch

    simple_output = model(noisy)
    invariant_output = masker.infer_full_image(noisy, model)

    idx = 18
    plot_tensors([clean[idx], noisy[idx], simple_output[idx], invariant_output[idx]],
                 ["Ground Truth", "Noisy Image", "Single Pass Inference", "J-Invariant Inference"])

    print("Test loss, single pass: ", round(loss_function(clean, simple_output).item(), 3))
    print("Test loss, J-invariant: ", round(loss_function(clean, invariant_output).item(), 3))



    #this is self with only one data.... doesnt learn

    import sys
    sys.path.append("..")
    from util import show, plot_images, plot_tensors
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.morphology import disk
    from skimage.filters import gaussian, median,
    from skimage import data, img_as_float, img_as_ubyte
    from skimage.color import gray2rgb
    from skimage.util import random_noise
    #from skimage.measure import compare_psnr
    #plt.rc('figure', figsize=(5, 5))
    #image = img_as_float(data.camera())
    #show(image)
    #np.random.seed(3)
    #noisy_image = random_noise(image, mode='gaussian', var=0.01)
    #show(noisy_image)
    noisy_image = imarray3D_mean  # _512



    import torch
    noisy = torch.Tensor(noisy_image[np.newaxis, np.newaxis])
    from util import getbestgpu
    #device = getbestgpu()
    from mask import Masker
    masker = Masker(width=4, mode='interpolate')

    torch.manual_seed(3)
    from models.dncnn import DnCNN
    model = DnCNN(1, num_of_layers=8)
    sum(p.numel() for p in model.parameters() if p.requires_grad)
    image.ravel().shape
    from torch.nn import MSELoss, L1Loss
    from torch.optim import Adam

    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    model = model.to()#model = model.to(device)
    noisy = noisy.to()#noisy = noisy.to(device)

    losses = []
    val_losses = []
    best_images = []
    best_val_loss = 1



    #this is modified for my image????
    from mask import Masker
    masker = Masker()
    for i, batch in enumerate(data_loader):
        noisy_images, _ = batch
        input, mask = masker.mask(noisy_images, i)
        output = model(input)
        loss = loss_function(output * mask, noisy_images * mask)




    for i in range(500):
        model.train()

        net_input, mask = masker.mask(noisy, i % (masker.n_masks - 1))
        net_output = model(net_input)

        loss = loss_function(net_output * mask, noisy * mask)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            losses.append(loss.item())
            model.eval()

            net_input, mask = masker.mask(noisy, masker.n_masks - 1)
            net_output = model(net_input)

            val_loss = loss_function(net_output * mask, noisy * mask)

            val_losses.append(val_loss.item())

            print("(", i, ") Loss: \t", round(loss.item(), 5), "\tVal Loss: \t", round(val_loss.item(), 5))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                denoised = np.clip(model(noisy).detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64)
                best_images.append(denoised)
    denoised = best_images[-1]
    """
    return denoised


def denoise(A, FFT_max_gaussian, high_thres):
    """
    This denoises the data with the optimal of wavlet, TV norm, or non-local means
        Args:
        -A Numpy array to be denoised
        - high_thres: this is the threshold to very high values
        -FFT_max_gaussian: this if high sigma for difference of gausian
        Returns:
        -	denoised: denoised image

    """
    sigma_est_ori = np.max(skimage.restoration.estimate_sigma(A, multichannel=False))
    print(f"estimated noise standard deviation = {sigma_est_ori}")
    # remove extremely high values --> remove hair
    if (A > (high_thres * A.mean())).any() and A.ndim == 2:
        warnings.warn(message='High intensity Image detected')
        high_thres_dec = high_thres / 100
        # remove whole connected image to this
        int_thres = (high_thres * A.mean())
        area_thres = A.shape[1] * (high_thres_dec / 2)
        A = remove_large_obj(A, area_thres, int_thres, FFT_max_gaussian, high_thres_dec)

    # remove salt and peper with TV norm, wavlet or non-local means
    # calibrate TV norm
    parameter_ranges_tv = {'weight': np.arange(0.1, 0.3, 0.5), 'eps': np.arange(0.0002, 0.001, 0.04),
                           'n_iter_max': np.arange(200, 300, 100), 'multichannel': [False]}
    _, (parameters_tested_tv, losses_tv) = calibrate_denoiser(A, denoise_tv_chambolle,
                                                              denoise_parameters=parameter_ranges_tv, extra_output=True)
    best_parameters_tv = parameters_tested_tv[np.argmin(losses_tv)]
    denoised_calibrated_tv = _invariant_denoise(A, denoise_tv_chambolle, denoiser_kwargs=best_parameters_tv)
    print(f'Minimum self-supervised loss TV: {np.min(losses_tv):.4f}')
    # calibrate wavelet
    parameter_ranges_wavelet = {'wavelet': ['db1', 'db2', 'haar'], 'convert2ycbcr': [False], 'multichannel': [False],
                                'rescale_sigma': [True], 'mode': ['soft'], 'method': ['BayesShrink']}
    _, (parameters_tested_wavelet, losses_wavelet) = calibrate_denoiser(A, denoise_wavelet,
                                                                        parameter_ranges_wavelet, extra_output=True)
    best_parameters_wavelet = parameters_tested_wavelet[np.argmin(losses_wavelet)]
    denoised_calibrated_wavelet = _invariant_denoise(A, denoise_wavelet, denoiser_kwargs=best_parameters_wavelet)
    print(f'Minimum self-supervised loss wavelet: {np.min(losses_wavelet):.4f}')
    # calibrate nl means
    sigma_est = estimate_sigma(A)
    parameter_ranges_nl = {'h': np.arange(1.15, 2, 5) * sigma_est, 'patch_size': np.arange(5, 10, 5),
                           'patch_distance': np.arange(6, 10, 5), 'multichannel': [False], 'fast_mode': [True],
                           'preserve_range': [True]}
    _, (parameters_tested_nl, losses_nl) = calibrate_denoiser(A, denoise_nl_means, parameter_ranges_nl,
                                                              extra_output=True)
    best_parameters_nl = parameters_tested_nl[np.argmin(losses_nl)]
    denoised_calibrated_nl = _invariant_denoise(A, denoise_nl_means, denoiser_kwargs=best_parameters_nl)
    print(f'Minimum self-supervised loss NL means: {np.min(losses_nl):.4f}')
    Low_methods = [np.min(losses_tv), np.min(losses_wavelet), np.min(losses_nl)]
    if Low_methods.index(min(Low_methods)) == 0:
        print(f'Method Minimum self-supervised loss is for TV: {np.min(losses_tv):.4f}')
        denoised1 = denoised_calibrated_tv
    elif Low_methods.index(min(Low_methods)) == 1:
        print(f'Method Minimum self-supervised loss is for wavelet: {np.min(losses_wavelet):.4f}')
        denoised1 = denoised_calibrated_wavelet
    elif Low_methods.index(min(Low_methods)) == 2:
        print(f'Method Minimum self-supervised loss is for NL means: {np.min(losses_nl):.4f}')
        denoised1 = denoised_calibrated_nl
    else:
        warnings.warn(message='Optimal denoise failed')
        denoised1 = A

    denoised1 = skimage.util.img_as_float(denoised1)

    # after salt and peper removed with above then use rolling-ball b/c rolling ball sensitive to salt/pepper.
    # remove lighting effects with rolling ball -- this removes too much of the data
    # background = skimage.restoration.rolling_ball(denoised1)
    # denoised = denoised1 - background
    # replace with difference of gausian?

    #todo change this? or use rolling ball that works better???
    denoised = skimage.filters.difference_of_gaussians(denoised1, 1, FFT_max_gaussian)

    # if all values in image are the same value set blank image to zero (should be the case already)
    if denoised.all() == denoised.max():
        warnings.warn(message='Blank denoise image')

    # if less then one --> #identify blank image?? does it matter??? why?
    if np.max(denoised) < 1:
        denoised = skimage.img_as_uint(denoised, force_copy=False)

    return denoised


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list. THIS IS USED FOR SEGMENTATION
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def remove_zero_segmntation(A_seg, A, Z):
    """Find if one segmented image has all zeros, if so then replace. only do this if NOT segmented into ONLY zero and ones
    """
    A_seg_ori = np.copy(A_seg)
    # find if one segmentation has all zeros, if so then replace zero segmentation with lowest segmenation
    seg_values_max = np.ones(A_seg.max() + 1)
    seg_values_min = np.ones(A_seg.max() + 1)
    all_zero_seg = []
    for count in range(A_seg.max() + 1):
        seg_values_max[count] = np.max(A[np.where(A_seg == count)])
        seg_values_min[count] = np.min(A[np.where(A_seg == count)])
        if seg_values_max[count] == 0 and seg_values_min[count] == 0:
            all_zero_seg.append(count)
    for i in range(len(all_zero_seg)):
        count = all_zero_seg[i]
        # change seg_values_min to doesnt have count value
        seg_values_min_new = np.delete(seg_values_min, count, axis=0)
        replace_value = np.where(seg_values_min == (seg_values_min_new[seg_values_min_new.argmin()]))
        A_seg[np.where(A_seg == count)] = replace_value
        if i > 0:
            warnings.warn(
                "Warning: Z image {} segmentation detection of multiple segmentation with zero value, this should not hapen ".format(
                    str(Z)))
    return A_seg, all_zero_seg


def segmentation(A, checkerboard_size, seg_interations, seg_smooth, localsubjectpath, Z, segment_otsu, extra_figs):
    """
    This segments the image with chan vase
        Args:
        - A Numpy array to be segmented
        - seg_interations: number of interations for chan_vese segmentation
        - checkerboard_size:  size of checkerboard for chan_vese segmentatioin
        -seg_smooth: number of smoothing/iterations for chan_vese segmentation
        -localsubjectpath: this is where to save figures
        -Z: Current Z value for figure names
        -segment_otsu ture to run otsu segmentation, false to run chan_vaase segmentation
        -extra_figs true to create extra figures
        Returns:
        -   A_seg: segmented image

    """

    np.save(localsubjectpath + 'Z=' + str(Z) + '_denoise_for_segmentation', A)

    # do this before segmentaion???
    # image = cv2.GaussianBlur(image, (5, 5), 0)

    # local otsu? https://scikit-image.org/docs/0.12.x/auto_examples/segmentation/plot_local_otsu.html
    # issue with local is that most area would just be noise....
    # img = np.interp(img, (img.min(), img.max()), (0, +255))
    # img = img.astype(np.uint8)
    # img = img_as_ubyte(img)
    # radius = 15
    # selem = disk(radius)
    # local_otsu = rank.otsu(img, selem)
    # A_seg=img >= local_otsu
    # imshow(img >= local_otsu, cmap=plt.cm.gray)
    # threshold_global_otsu = threshold_otsu(img)
    # global_otsu = img >= threshold_global_otsu

    if segment_otsu:
        # get threshold values for binary image (b/c only 2 classes)
        thresholds = threshold_multiotsu(A, classes=2)
        # Using the threshold values, we generate the three regions.
        A_seg2 = np.digitize(A, bins=thresholds)
        A_seg2, all_zero_seg = remove_zero_segmntation(A_seg2, A, Z)

        # todo why not make this a for loop that combines all zero groups until increaseing number of classes doesnt give addditional zero group???
        # todo this is basically trying to find ideal number of clusters????
        # zero_classes=0
        # numclass = 3
        # while zero_classes < 1:
        # thresholds = threshold_multiotsu(A, classes=numclass) #todo why does this give classes without any data????
        # A_seg = np.digitize(A, bins=thresholds)
        # remove empty classes
        # todo remove empty classes np.histogram(A_seg)

        # if empty classes exits
        # zero_classes=1

        # numclass = numclass + 1
        # combine zero segmented into other classes if needed
        # A_seg3, all_zero_seg = remove_zero_segmntation(A_seg3, A, Z)
        # set all but highest class value to zero
        # result = np.where(A_seg3<numclass, 0, A_seg3)

        if len(all_zero_seg):  # use 3 classes if segmentation with all zeros exists
            thresholds = threshold_multiotsu(A, classes=3)
            A_seg3 = np.digitize(A, bins=thresholds)
            # make sure len(np.unique(A_seg)) >= 3
            if len(np.unique(A_seg3)) >= 3:
                A_seg3, all_zero_seg = remove_zero_segmntation(A_seg3, A, Z)
                A_seg = A_seg3
            else:
                # todo figure this part out???? --> why is this werid??? ddue to remove large object zeros???
                print('segmentation has only 2 classes with 3 classes set')
                thresholds = threshold_multiotsu(A, classes=2)
                # Using the threshold values, we generate the three regions.
                A_seg2_raw = np.digitize(A, bins=thresholds)
                A_seg = A_seg2_raw
        else:
            A_seg = A_seg2
    else:
        # Initial level set
        init_ls = skimage.segmentation.checkerboard_level_set(A.shape, checkerboard_size)  # default is 5
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = store_evolution_in(evolution)
        # smoothing of 3 works
        A_seg = skimage.segmentation.morphological_chan_vese(A, seg_interations, init_level_set=init_ls,
                                                             smoothing=seg_smooth,
                                                             iter_callback=callback)  # here 35 is the number of iterations and smoothing=3 is number of times smoothing applied/iteration
        # remove segmentation if all zeros
        A_seg, all_zero_seg = remove_zero_segmntation(A_seg, A, Z)
        if len(all_zero_seg):  # use otsu 3 classes if zero segmentation exisits
            warnings.warn(
                "Warning: Z image {} segmentation of all zeros deteccted, using otsu segmentation with 3 classes ".format(
                    str(Z)))
            thresholds = threshold_multiotsu(A, classes=3)
            A_seg = np.digitize(A, bins=thresholds)
            A_seg, all_zero_seg = remove_zero_segmntation(A_seg, A, Z)

    # check to make sure segmented values start at zero
    if A_seg.min() != 0:
        A_seg = A_seg - A_seg.min()
        warnings.warn(
            "Warning: Z image {} segmentation starts at nonzero, shifting to zero segmentation ".format(str(Z)))

    # check to make sure background =0 and segment brain is =1
    if np.count_nonzero(A_seg == A_seg.max()) > np.count_nonzero(A_seg == 0):
        # inverse image
        A_seg = A_seg.max() - A_seg
        warnings.warn(
            "Warning: Z image {} segmentation of background labeled high, inverting segmentation ".format(str(Z)))

    # todo remove grid lines in segmented image
    # A_seg_gray = np.interp(A_seg, (A_seg.min(), A_seg.max()), (0, +255))
    # change dtype to unit 8 --> so get rid of decimals
    # A_seg_gray = A_seg_gray.astype(np.uint8)
    # img_gauss = cv2.GaussianBlur(A_seg_gray, ksize=(13, 13), sigmaX=10)
    # ret, img_gauss_th = cv2.threshold(img_gauss, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
    # edges = cv2.Canny(image=img_gauss_th, threshold1=240, threshold2=255)
    # edges_smoothed = cv2.GaussianBlur(edges, ksize=(5, 5), sigmaX=10)
    # lines = cv2.HoughLinesP(edges_smoothed, rho=1, theta=1 * np.pi / 180, threshold=40, minLineLength=30, maxLineGap=25)
    # img_hough_lines_contours = np.ones(A_seg_gray.shape)
    # line_nos = lines.shape[0]
    # for i in range(line_nos):
    #    x_1 = lines[i][0][0]
    #    y_1 = lines[i][0][1]
    #    x_2 = lines[i][0][2]
    #    y_2 = lines[i][0][3]
    #    cv2.line(A_seg, pt1=(x_1, y_1), pt2=(x_2, y_2), color=(255, 0, 0), thickness=2)
    # img_hough_lines = A_seg_gray.copy()
    # img_hough_lines[img_hough_lines_contours == 0] = 255

    if extra_figs:
        plt.figure()
        plt.imshow(A)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
        plt.savefig(localsubjectpath + '/Z=' + str(Z) + '_Zplane_before_seg.png', format='png')
        plt.close()
        plt.figure()
        plt.imshow(A_seg)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
        plt.savefig(localsubjectpath + '/Z=' + str(Z) + '_Zplane_after_segmented.png', format='png')
        plt.close()
    return A_seg


def feature_CV2(destination_raw, source_raw, diroverlap, FFT_max_gaussian, high_thres, localsubjectpath, input_overlap):
    # todo try https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_brief.html#sphx-glr-auto-examples-features-detection-plot-brief-py
    # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html#sphx-glr-auto-examples-features-detection-plot-orb-py
    """This finds the overlay given A and B with feature based registration
        Args:
        -	source_raw: 3D set of images to overlap source (moving image)
        -   destination_raw: 3D set of images to overlap target
        -   diroverlap: the direction of overlap, the direction the source image moves --> up, down, left or right
        -  FFT_max_gaussian: used for denoise
        -  high_thres: used for denoise
        - localsubjectpath: path to save intermediate figures
        -input_overlap: provided overlap value

        Returns:
        -	target_overlap: pixel number of overlay
    """
    destination_mean = np.max(destination_raw, axis=2)
    source_mean = np.max(source_raw, axis=2)
    # denoise image
    destination_denoise = denoise(destination_mean, FFT_max_gaussian, high_thres)
    source_denoise = denoise(source_mean, FFT_max_gaussian, high_thres)
    # define denoised data
    destination = destination_denoise
    source = source_denoise
    # convert to grayscale
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
    return target_overlap,


# todo rework this function? maybe replace with combo of other functions?
# todo if we move denoise to BEFORE registration for X and Y then only need to segment here if fails for X and Y, although denoise still useful if WITHIN registration (add input for this ???)
# this is used once in registration within, once in registration X, once in registration Y --> this is used when not in range

def Seg_reg_phase(A, B, FFT_max_gaussian, name, extra_figs, high_thres):
    """
    This denoise and segments data then finds phase correlation of segmented data. This is used if registration is outside of range when first performed.
        Args:
        -	A,B: Numpy array 3D image
        - FFT_max_gausian: this is for removing low frequency sugested value = 10
        - name: file name to save
        - extra_figs: This is true to make extra figures
        - high_thres:
        Returns:
        -	shift_reg: shift of transfrom
        -	error: error of transfrom
        -	Within_Trans: Transform

    filt_A = skimage.filters.difference_of_gaussians(A, 1, FFT_max_gaussian)
    filt_B = skimage.filters.difference_of_gaussians(B, 1, FFT_max_gaussian)
    A_ant = ants.from_numpy(data=filt_A)
    B_ant = ants.from_numpy(data=filt_B)
    # threshold?
    maskA = ants.get_mask(A_ant)
    maskB = ants.get_mask(B_ant)
    """
    # use band pass filtering to remove shaddow
    # filt_A = skimage.filters.difference_of_gaussians(A, 1, FFT_max_gaussian)
    # denoise with wavlet this might not have much of an effect
    # denoise_A = skimage.restoration.denoise_wavelet(filt_A, multichannel=False, rescale_sigma=True)
    # set stuff below mean to zero?
    # low_values_flags = denoise_A < denoise_A.mean()
    # denoise_A[low_values_flags] = 0
    # high_values_flags = denoise_A > high_thres * denoise_A.mean()
    # denoise_A[high_values_flags] = 0
    # maskA = denoise_A
    # use band pass filtering to remove shaddow
    # filt_B = skimage.filters.difference_of_gaussians(B, 1, FFT_max_gaussian)
    # denoise_B = skimage.restoration.denoise_wavelet(filt_B, multichannel=False, rescale_sigma=True)
    # set stuff below mean to zero?
    # low_values_flags = denoise_B < denoise_B.mean()
    # denoise_B[low_values_flags] = 0
    # high_values_flags = denoise_B > high_thres * denoise_B.mean()
    # denoise_B[high_values_flags] = 0
    # maskB = denoise_B

    # denoise data
    maskA = denoise(A, FFT_max_gaussian, high_thres)
    maskB = denoise(B, FFT_max_gaussian, high_thres)

    # segment data --> issue is chan vase segmnetation fails with little data.... does ostu?
    # what about log trnasform? and postive value shift?
    # A_seg=segmentation(A, checkerboard_size, seg_interations, seg_smooth, localsubjectpath, Z, segment_otsu, extra_figs)
    # A_seg = segmentation(A, checkerboard_size, seg_interations, seg_smooth, localsubjectpath, Z, segment_otsu, extra_figs)

    # shift segmented data
    shift_reg, error, diffphase = skimage.registration.phase_cross_correlation(maskA, maskB)
    print('Filtered diff phase of {}'.format(diffphase))
    Within_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                         translation=(shift_reg[1], shift_reg[0]))
    if extra_figs:
        plt.figure()
        plt.imshow(maskA)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
        plt.savefig(name + 'overlay_mask_A.png', format='png')
        plt.close()
        plt.figure()
        plt.imshow(maskB)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
        plt.savefig(name + 'overlay_mask_B.png', format='png')
        plt.close()

    return shift_reg, error, Within_Trans


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


def registration_within(A, B, FFT_max_gaussian, error_overlap, X, Y, Z, i, localsubjectpath, extra_figs, high_thres):
    """
    This registers within a given X Y Z value. THis registered the optical zoom in the confocal imaging
        Args:
        -	A,B: the A and B array to register
        -   FFT_max_gaussian: The fft sigma max to input Seg_reg_phase
        -   error_overlap: The acceptable error limts
        -   X, Y, Z, i: The values of this slice to use for naming and warnings
        - localsubjectpath: path to save extra files
        - extra_figs: set to true to save figures
        - high_thres: this is for denoising if needed used in Seg_reg_phase

        Returns:
        -	error : registration error
        -   shift_reg : registration shift
        -   Within_Trans : registration transformation

    """

    shift_reg, error, diffphase = skimage.registration.phase_cross_correlation(A, B)
    Within_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                         translation=(shift_reg[1], shift_reg[0]))
    trans_init_array = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # here if NOT identity matrix output a warning
    trans_init_array_np = np.array(trans_init_array)
    if not np.array_equal(Within_Trans._inv_matrix, trans_init_array_np):
        print(
            "Non identify matrix found shift of {} and {} found for cube X {} Y {} Z {} and file {}, preprocessing file and running registration again".format(
                shift_reg[0], shift_reg[1], X, Y, Z, i, ))
        # try segmentation then regisstraataion with phasse correlation
        name = '{}/Z_{}_X_{}_Y{}_i{}'.format(localsubjectpath, Z, X, Y, i)
        [shift_reg, error, Within_Trans] = Seg_reg_phase(A, B, FFT_max_gaussian, name, extra_figs, high_thres)
        reg_range_shift = range(round((-1 * (A.shape[0] / 10 * error_overlap))),
                                round((A.shape[0] / 10 * error_overlap)))
        if shift_reg[0] not in reg_range_shift or shift_reg[1] not in reg_range_shift:
            warnings.warn(
                "WARNING:for cube X {} Y {} Z{} and file {} we get non identity large transform of {}.Setting to idenity".format(
                    X, Y, Z, i, Within_Trans._inv_matrix))
            Within_Trans._inv_matrix = trans_init_array
    del trans_init_array_np, trans_init_array
    return error, shift_reg, Within_Trans


def registration_X(srcY, desY, X_dir, FFT_max_gaussian, error_overlap, X, Y, Z, blank_thres, localsubjectpath,
                   extra_figs, high_thres, input_overlap,
                   target_overlapX=None):
    """
    This registers stiches along X axis
        Args:
        -	srcY, desY: the source and destination  array to register
        -   X_dir: the direction of overlap (left or right)
        -   FFT_max_gaussian: The fft sigma max to input Seg_reg_phase
        -   error_overlap: The acceptable error limts
        -   X, Y, Z: The values of this slice to use for naming and warnings
        -  blank_thres: the threhold for ID blank images
        - localsubjectpath:location to save output figures
        - extra_figs: Set to true to make additional figures
        - high_thres: used in feature based ID for denoise

        Optional args:
        - target_overlapX: calculated overlap
        - input_overlap: inital guess of overlap

        Returns:
        -	error: registration error
        -   shift : registration shift
        -   target_overlapX  : initial registration guess along X

    """

    if input_overlap is None:
        if X == 1 and Y == 0 and Z == 0:  # calculate the initial stitch level ONLY for first overlap
            diroverlap = X_dir  # 'right'
            [target_overlapX] = feature_CV2(desY, srcY, diroverlap, FFT_max_gaussian, high_thres, localsubjectpath,
                                            input_overlap)
    else:
        # convert percent in image overlap to number of pixels
        input_overlap = round(srcY.shape[1] * input_overlap)
        if X == 1 and Y == 0 and Z == 0:  # calculate the initial stitch level ONLY for first overlap
            diroverlap = X_dir  # 'right'
            [target_overlapX] = feature_CV2(desY, srcY, diroverlap, FFT_max_gaussian, high_thres, localsubjectpath,
                                            input_overlap)
            # make sure user defined input of input_overlap, else just use calculated
            try:
                input_overlap
            except NameError:
                input_overlap = target_overlapX
            # print warning if user defined overlap and calculated overlap different
            if target_overlapX in range(round((input_overlap - (input_overlap * error_overlap))),
                                        round((input_overlap + (input_overlap * error_overlap)))):
                print("X overlap within {} % of user defined values.".format(error_overlap * 100))
            else:
                # print warning
                warnings.warn(
                    "WARNING: calculated X overlap of {} not within {} % of user defined overlap of {}. Using user defined overlap.".format(
                        target_overlapX, (error_overlap * 100), input_overlap))
                # use user defined value instead
                target_overlapX = input_overlap
    # calculate shift on MEAN image --> apply to whole image this helps with noisy images
    srcY_mean = np.max(srcY, axis=2)  # , dtype=srcY.dtype)
    desY_mean = np.max(desY, axis=2)  # , dtype=desY.dtype)
    if X_dir == 'right':
        shiftXi = -(srcY.shape[1] - abs(target_overlapX))
    elif X_dir == 'left':
        shiftXi = abs(target_overlapX)
    SKI_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                      translation=(shiftXi, 0))
    srcY_Ti = skimage.transform.warp(srcY_mean, SKI_Trans._inv_matrix, mode='edge')
    if srcY_Ti.max() < 1:  # only if max less then one convert to unit
        srcY_Ti = skimage.img_as_uint(srcY_Ti, force_copy=False)
    # srcY_Ti = np.array(srcY_Ti, dtype=srcY_mean.dtype)
    # get phase correlation for small local change
    print(target_overlapX)
    if X_dir == 'right':
        desY_mean_overlap = desY_mean[:, :(target_overlapX)]
        srcY_Ti_mean_overlap = srcY_Ti[:, :(target_overlapX)]
    elif X_dir == 'left':
        desY_mean_overlap = desY_mean[:, -target_overlapX:]
        srcY_Ti_mean_overlap = srcY_Ti[:, -target_overlapX:]

    del desY_mean, srcY_mean
    shift, error, diffphase = skimage.registration.phase_cross_correlation(desY_mean_overlap,
                                                                           srcY_Ti_mean_overlap)
    print('X original error of {}'.format(error))
    print('X diff phase of {}'.format(diffphase))
    # find if BLANK squares added together, if so redefine shift to zero
    if desY_mean_overlap.max() < blank_thres * desY_mean_overlap.mean() and srcY_Ti_mean_overlap.max() < blank_thres * srcY_Ti_mean_overlap.mean():
        print("Blank image overlap found at Y {} Z {} for X {} and X {}, setting to default overlap".format(Y, Z, X,
                                                                                                            (X - 1)))
        blank_overlap = True
        shift = np.array([0, 0])
        if extra_figs:
            plt.figure()
            plt.imshow(srcY_Ti_mean_overlap)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(
                localsubjectpath + 'X=' + str(X) + 'Y=' + str(Y) + 'Z=' + str(Z) + '_source_overlap_X_blank.png',
                format='png')
            plt.close()
            plt.figure()
            plt.imshow(desY_mean_overlap)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(
                localsubjectpath + 'X=' + str(X) + 'Y=' + str(Y) + 'Z=' + str(Z) + '_destination_overlap_X_blank.png',
                format='png')
            plt.close()
    else:
        blank_overlap = False
    # give warning if shift is more then error_overlap of target_overlapX
    # print warning if user defined overlap and calculated overlap different
    X_range_shift = range(round((-1 * (target_overlapX * error_overlap))), round((target_overlapX * error_overlap)))
    if shift[0] not in X_range_shift or shift[1] not in X_range_shift:
        print("Sparse image found with shift of {} and {}. Using filtered phase shift for overlap registration.".format(
            shift[1], shift[0]))
        if extra_figs:
            plt.figure()
            plt.imshow(srcY_Ti_mean_overlap)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(
                localsubjectpath + 'X=' + str(X) + 'Y=' + str(Y) + 'Z=' + str(Z) + '_source_overlap_X_sparse.png',
                format='png')
            plt.close()
            plt.figure()
            plt.imshow(desY_mean_overlap)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(
                localsubjectpath + 'X=' + str(X) + 'Y=' + str(Y) + 'Z=' + str(Z) + '_destination_overlap_X_sparse.png',
                format='png')
            plt.close()
        # try segmentation followed by phase correlation
        name = '{}/Z_{}_X_{}_Y{}'.format(localsubjectpath, Z, X, Y)
        [shift, error, Within_Trans] = Seg_reg_phase(desY_mean_overlap, srcY_Ti_mean_overlap, FFT_max_gaussian, name,
                                                     extra_figs, high_thres)
        print('X segmented data error of {}'.format(error))
        # NOW IF STILL NOT WITHIN 10%... use defalt
        if shift[0] not in X_range_shift or shift[1] not in X_range_shift:
            # print warning
            warnings.warn(
                "WARNING: image overlap of Z {} Y {} and X {} and X {} not within error. Overlap varry by {} and {} pixels. changing to defult overlaap".format(
                    Z, Y, (X - 1), X, shift[1], shift[0]))
            shift = np.array([0, 0])
    del srcY_Ti, SKI_Trans, desY_mean_overlap, srcY_Ti_mean_overlap
    return error, shift, target_overlapX, blank_overlap


def registration_Y(srcZ, desZ, Y_dir, FFT_max_gaussian, error_overlap, X, Y, Z, blank_thres, localsubjectpath,
                   extra_figs, high_thres, input_overlap, target_overlapY=None):
    """
    This registers stiches along X axis
        Args:
        -	srcZ, desZ: the source and destination  array to register
        -   Y_dir: the direction of overlap (top or bottom)
        -   FFT_max_gaussian: The fft sigma max to input Seg_reg_phase
        -   error_overlap: The acceptable error limts
        -   X, Y, Z: The values of this slice to use for naming and warnings
        -  blank_thres: the threhold for ID blank images
        - localsubjectpath:location to save output figures
        - extra_figs: Set to true to make additional figures
        - high_thres: used in feature based ID for denoise
        Optional args:
        - target_overlapY: calculated overlap
        - input_overlap: input guess of overlap

        Returns:
        -	error: registration error
        -   shift : registration shift
        -   target_overlapY  : initial registration guess along Y

    """
    if input_overlap is None:
        if Y == 1 and Z == 0:  # calculate the initial stitch level ONLY for first overlap
            if Y_dir == 'top':
                diroverlap = 'down'
            elif Y_dir == 'bottom':
                diroverlap = 'up'
            [target_overlapY] = feature_CV2(desZ, srcZ, diroverlap, FFT_max_gaussian, high_thres, localsubjectpath,
                                            input_overlap)
    else:
        # convert percent in image overlap to number of pixels
        input_overlap = round(srcZ.shape[0] * input_overlap)
        # pad in the Y
        dim = 1
        [srcZ, desZ] = zero_pad(srcZ, desZ, dim)
        if Y == 1 and Z == 0:  # calculate the initial stitch level ONLY for first overlap
            if Y_dir == 'top':
                diroverlap = 'down'
            elif Y_dir == 'bottom':
                diroverlap = 'up'
            [target_overlapY] = feature_CV2(desZ, srcZ, diroverlap, FFT_max_gaussian, high_thres, localsubjectpath,
                                            input_overlap)
            # print warning if user defined overlap and calculated overlap different
            if target_overlapY in range(round((input_overlap - (input_overlap * error_overlap))),
                                        round((input_overlap + (input_overlap * error_overlap)))):
                print("Y overlap within {} of user defined values.".format(error_overlap))
            else:
                # print warning
                warnings.warn(
                    "WARNING: calculated Y overlap of {} not within {} % of user defined overlap of {}. Using user defined overlap.".format(
                        target_overlapY, (error_overlap * 100), input_overlap))
                target_overlapY = input_overlap
    # calculate shift on MEAN image --> apply to whole image this helps with noisy images
    srcZ_mean = np.max(srcZ, axis=2)  # , dtype=srcZ.dtype)
    desZ_mean = np.max(desZ, axis=2)  # , dtype=desZ.dtype)
    if Y_dir == 'top':
        shiftYi = -(srcZ.shape[0] - abs(target_overlapY))
    elif Y_dir == 'bottom':
        shiftYi = abs(target_overlapY)
    SKI_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                      translation=(0, shiftYi))
    srcZ_Ti = skimage.transform.warp(srcZ_mean, SKI_Trans._inv_matrix, mode='edge')
    if np.max(srcZ_Ti) < 1:  # only if max less then one convert to unit
        srcZ_Ti = skimage.img_as_uint(srcZ_Ti, force_copy=False)
    else:
        srcZ_Ti = np.array(srcZ_Ti, dtype=srcZ_mean.dtype)
    # get phase correlation for small local change
    if Y_dir == 'top':
        desZ_mean_overlap = desZ_mean[:(target_overlapY), :]
        srcZ_Ti_mean_overlap = srcZ_Ti[:(target_overlapY), :]
    elif Y_dir == 'bottom':
        desZ_mean_overlap = desZ_mean[-(target_overlapY):, :]
        srcZ_Ti_mean_overlap = srcZ_Ti[-(target_overlapY):, :]
    del desZ_mean, srcZ_mean
    shift, error, diffphase = skimage.registration.phase_cross_correlation(desZ_mean_overlap, srcZ_Ti_mean_overlap)
    print('Y original error of {}'.format(error))
    print('Y original diff phase of {}'.format(diffphase))
    # find if BLANK squares added together, if so redefine shift to zero
    if desZ_mean_overlap.max() < blank_thres * desZ_mean_overlap.mean() and srcZ_Ti_mean_overlap.max() < blank_thres * srcZ_Ti_mean_overlap.mean():
        print("Blank image overlap found at Z {} for y {} and Y {}, setting to default overlap".format(Z, Y,
                                                                                                       (Y - 1)))
        shift = np.array([0, 0])
        if extra_figs:
            plt.figure()
            plt.imshow(srcZ_Ti_mean_overlap)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(localsubjectpath + 'Y=' + str(Y) + 'Z=' + str(Z) + '_source_overlap_Y_blank.png', format='png')
            plt.close()
            plt.figure()
            plt.imshow(desZ_mean_overlap)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(localsubjectpath + 'Y=' + str(Y) + 'Z=' + str(Z) + '_destination_overlap_Y_blank.png',
                        format='png')
            plt.close()
    Y_range_shift = range(round((-1 * (target_overlapY * error_overlap))),
                          round((target_overlapY * error_overlap)))
    if shift[0] not in Y_range_shift or shift[1] not in Y_range_shift:
        print("Sparse image found with shift of {} and {}. Using filtered phase shift for overlap registration.".format(
            shift[0], shift[1]))
        if extra_figs:
            plt.figure()
            plt.imshow(srcZ_Ti_mean_overlap)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(localsubjectpath + 'Y=' + str(Y) + 'Z=' + str(Z) + '_source_overlap_Y_sparse.png', format='png')
            plt.close()
            plt.figure()
            plt.imshow(desZ_mean_overlap)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(localsubjectpath + 'Y=' + str(Y) + 'Z=' + str(Z) + '_destination_overlap_Y_sparse.png',
                        format='png')
            plt.close()
        # try segmentation and regisstration
        name = '{}/Z_{}_X_{}_Y{}'.format(localsubjectpath, Z, X, Y)
        [shift, error, Within_Trans] = Seg_reg_phase(desZ_mean_overlap, srcZ_Ti_mean_overlap, FFT_max_gaussian, name,
                                                     extra_figs, high_thres)
        print('Y segmented data error of {}'.format(error))
        # NOW IF STILL NOT WITHIN 10%... use defalt
        if shift[0] not in Y_range_shift or shift[1] not in Y_range_shift:
            # print warning
            warnings.warn(
                "WARNING: image overlap of Z {} for Y {} and Y {} not within error. Overlap varry by {} and {} pixels. Changing to default overlap".format(
                    Z, Y, (Y - 1), shift[0], shift[1]))
            shift = np.array([0, 0])
    del desZ_mean_overlap, srcZ_Ti_mean_overlap, SKI_Trans, srcZ_Ti
    return error, shift, target_overlapY


# In case of boolaen segmentation, the dice loss is equal to 2 x dot product of the flattened arrays,
# divided by the total volume of both masks.
def dice_loss(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = y_true.dot(y_pred)
    union = sum(y_true) + sum(y_pred)
    dice = 2 * intersection / union
    return dice


def rmsdiff(source, source_transformed, destination):
    import ants
    source_ant = ants.from_numpy(data=source)
    destination_ant = ants.from_numpy(data=destination)
    # register 2 datasets
    mytx = ants.registration(fixed=destination_ant, moving=source_ant, initial_transform=None, outprefix="test",
                             dimension=2)  # “Similarity”
    # apply transfrom to unsegmented data
    p2_reg_ants = ants.apply_transforms(fixed=bar_ant, moving=p2_ant, transformlist=mytx['fwdtransforms'])
    # convert back to numpy
    p2_reg = p2_reg_ants.numpy()

    ants.image_mutual_information(source_ant, destination_ant)

    # todo .... find a way to compapre Z matching methods....

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    rms = sqrt(mean_squared_error(im1, im2))

    "Calculate the root-mean-square difference between two images"
    from PIL import Image
    from matplotlib import cm
    im1_im = Image.fromarray(np.uint8(cm.gist_earth(im1) * 255))
    im2_im = Image.fromarray(np.uint8(cm.gist_earth(im2) * 255))

    diff = ImageChops.difference(im1, im2)

    diff = np.abs(im1 - im2)

    h = np.histogram(diff)  # diff.histogram()
    sq = (value * (idx ** 2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares / float(im1.size[0] * im1.size[1]))

    # or does this evaluate?
    # or what if we try: sum of squared intensity differences (SSD): -->

    return rms


def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # [M x M]
    return ZCAMatrix

def auto_detect_180(destination_feature, source_feature, localsubjectpath):
    """
    This finds if 180 is found or not. This uses elastix.
    Documentation of elastix at: https://elastix.lumc.nl/doxygen/index.html and http://elastix.bigr.nl/wiki/index.php/Par0013
    input:
    destination_feature: destination image
    source_feature: source image
    localsubjectpath: path to save files

    output:
    new_angle_rad: angle either 180 or 0 if 180 rotation detected
    """
    # rotate image by theta_rad, set to 180 or 0 in Z registration code
    # source_feature = skimage.transform.rotate(source_feature, math.degrees(theta_rad)) #here we want to use segmented image b/c intensity of blood vessesl isnt meaninful
    # convert data to work with elastix software package
    destination_feature = destination_feature.astype(np.float32)
    source_feature = source_feature.astype(np.float32)
    fixed_image_feature = itk.image_from_array(destination_feature)
    moving_image_feature = itk.image_from_array(source_feature)
    # define parameter file
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    parameter_object.SetParameter("Transform", "EulerTransform")  # this is rigid
    parameter_object.SetParameter("Optimizer",
                                  "StandardGradientDescent")  # full search # this allows me to pick step size what about evolutionary strategy? FullSearch, ConjugateGradient, ConjugateGradientFRPR, QuasiNewtonLBFGS, RSGDEachParameterApart, SimultaneousPerturbation, CMAEvolutionStrategy.
    parameter_object.SetParameter("Registration", "MultiResolutionRegistration")
    parameter_object.SetParameter("Metric",
                                  "AdvancedKappaStatistic")  # "AdvancedNormalizedCorrelation") #AdvancedNormalizedCorrelation") #maybe AdvancedMattesMutualInformation
    parameter_object.SetParameter("FixedInternalImagePixelType", "float")
    parameter_object.SetParameter("MovingInternalImagePixelType", "float")
    parameter_object.SetParameter("FixedImageDimension", "2")
    parameter_object.SetParameter("MovingImageDimension", "2")
    parameter_object.SetParameter("FixedImagePyramid", "FixedRecursiveImagePyramid")  # smooth and downsample
    parameter_object.SetParameter("MovingImagePyramid", "MovingRecursiveImagePyramid")  # smooth and downsample
    parameter_object.SetParameter("NumberOfResolutions", "6")  # high resoltuiions b/c large shifts
    parameter_object.SetParameter("DefaultPixelValue", "0")  # this is b/c background is 0
    parameter_object.SetParameter("ImageSampler",
                                  "Grid")  # want full b/c random sampler would give different metric to compare results so full or grid should work, full is really slow
    parameter_object.SetParameter("NewSamplesEveryIteration",
                                  "false")  # want false b/c random sampler would give different metric to compare results
    parameter_object.SetParameter("Scales", "1.0")  # this is so rotation and translation treated equally
    parameter_object.SetParameter("UseDirectionCosines", "false")
    parameter_object.SetParameter("AutomaticTransformInitialization", "true")
    parameter_object.SetParameter("AutomaticTransformInitializationMethod", "GeometricalCenter")
    parameter_object.SetParameter("UseComplement", "false")
    parameter_object.SetParameter("WriteResultImage", "true")
    parameter_object.SetParameter("MaximumNumberOfIterations", "500")
    parameter_object.SetParameter("MaximumNumberOfSamplingAttempts", "3")

    # note: This is something specific to our scope and probably would not generalize to data from other people. I would just rotate them manually for each dataset.
    # rotate image 180 degrees
    # source_denoise_180 = skimage.transform.rotate(source_denoise, 180)
    # source_denoise_180 = source_denoise_180.astype(np.float32)
    # moving_image_denoise_180 = itk.image_from_array(source_denoise_180)
    source_feature_180 = skimage.transform.rotate(source_feature, 180)
    source_feature_180 = source_feature_180.astype(np.float32)
    moving_image_feature_180 = itk.image_from_array(source_feature_180)
    # get registration metirc for 180 and original data
    result_image, result_transform_parameters = itk.elastix_registration_method(fixed_image=fixed_image_feature,
                                                                                moving_image=moving_image_feature,
                                                                                parameter_object=parameter_object,
                                                                                log_to_console=True, log_to_file=True,
                                                                                output_directory=localsubjectpath)  # '/Users/kaleb/Documents/CSHL/2d_3D_linear_reg/')
    # load localsubjectpath /elastix.log file
    file1 = open(localsubjectpath + "elastix.log", "r")
    string = file1.read()
    m1 = re.search('Final metric value  = (\d+.\d+)', string)
    Final_metric_value = float(m1.group(1))
    file1.close()
    # get denoise (with feaure-- multimodal?) 180 rotation registration
    result_image_180, result_transform_parameters_180 = itk.elastix_registration_method(fixed_image=fixed_image_feature,
                                                                                        moving_image=moving_image_feature_180,
                                                                                        parameter_object=parameter_object,
                                                                                        log_to_console=True,
                                                                                        log_to_file=True,
                                                                                        output_directory=localsubjectpath)  # '/Users/kaleb/Documents/CSHL/2d_3D_linear_reg/')
    file2 = open(localsubjectpath + "elastix.log", "r")
    string = file2.read()
    m2 = re.search('Final metric value  = (\d+.\d+)', string)
    Final_metric_value_180 = float(m2.group(1))
    file2.close()
    # if 180 rotaoin better:
    if Final_metric_value_180 > Final_metric_value:  # for AdvancedKappaStatistic higher value is better (1.0 = prefect)
        result_image_good = result_image_180
        result_transform_parameters_good = result_transform_parameters_180
        warnings.warn(message="WARNING: 180 degree rotation detected")
        add_theta = 180
    else:
        result_image_good = result_image
        result_transform_parameters_good = result_transform_parameters
        add_theta = 0
    # now get rotation and translations measurements
    # pm0 = result_transform_parameters_good.GetParameterMap(0)
    # center = [float(p) for p in pm0['CenterOfRotationPoint']]  # todo take this as center of rotatiion
    # elx_parameters = [float(p) for p in pm0['TransformParameters']]  # todo take this as translation and degree values?
    # Y_shift =   #these aare in number of pixels (sould be -150 for test case)
    # X_shift =   #these aare in number of pixels (shoould be -100 for test case)
    # elx_angle=  #this  is in radians (should be 5 for test case)
    # new_angle_rad = elx_angle + math.radians(add_theta) #this  is in radians both 180 or not roation and elastix rotation
    # shift[0] = Y_shift  #Y direction, axis 0
    # shift[1] = X_shift  #X direction, axis 1
    new_angle_rad = math.radians(add_theta)

    return new_angle_rad


def phase_corr_rotation(destination, source, degree_thres, theta_rad):
    """
    This finds the rotation angle if small

    Input:
    destination: destination image
    source: source image
    degree_thres: degree threshold for warning messages
    theta_rad: amount to initially rotate if given

    Output:
    new_angle_rad: output angle to rotation in radians
    """
    # rotate source by theta
    source_theta = skimage.transform.rotate(source, math.degrees(theta_rad))

    myconstraints = {
        "scale": [1, 0]  # this is a scale of 1 without and stardev
        # todo add  degree_thres as a constriant here???
    }
    result = ird.imreg.similarity(destination, source_theta,
                                  constraints=myconstraints)
    theta = abs(result["angle"])

    lst = [0, 180]  # this is list of angles to return closest one to measured valve
    idx = (np.abs(lst - theta)).argmin()
    new_angle_180_0 = lst[idx]
    if new_angle_180_0 == 180:
        warnings.warn(message="WARNING: 180 degree rotation detected")
    # if outside threshold
    if np.abs((new_angle_180_0 - theta)) > degree_thres:
        # if outside threshold set theta to 0, b/c setting to 180 or 0 might lead to false postive if not strong evidance of 180 rotation
        new_angle = 0
        warnings.warn(
            message="WARNING: Z rotation degree shift larger then {} from {}, angle of {} detected. Setting angle to {}".format(
                degree_thres, new_angle_180_0, np.abs((new_angle_180_0 - theta)), new_angle))
    else:
        new_angle = theta
    print('Z Theta is {}'.format(new_angle))
    # convert angle to radians
    new_angle_rad = math.radians(new_angle)
    return new_angle_rad


def elastix_2D_registration(destination, source, nonlinear_trans):
    """
    This fund non-linear (bspline) or linear (similarity) transform between source and destination data. THis uses elastix registtration

    Input:
    destination: destination image, ideally feature based image
    source: source image, ideally feature based image
    nonlinear_trans: This is set to true to find on-linear transformation, false for linear transformation

    Output:
    transformation: elastix tranfomation matrix
    """

    # todo try elastic on failed 2d to 2d registratoin also try optical flow, feature based???? try phase correaltion on raw data? try prewhitten b4 denosie phase correaltion?
    # _deps/elx-src/Common/OpenCL/ITKimprovements/itkOpenCLContext.cxx(387): itkOpenCL warning.
    # Warning: in function: Create; Name: OpenCLContext (0x555557873e00)
    # Details: OpenCLContext::Create(method:4):CL_INVALID_PLATFORM
    # this gets tranformation parameters: GetTransformParameterMap()
    # (TransformParameters θ_x , θ_y , θ_z , t_x , t_y , t_z) --> in mm???

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
        parameter_object.SetParameter("Transform", "SimilarityTransform")  # this is scale, rottation, and translation
    parameter_object.SetParameter("Optimizer",
                                  "StandardGradientDescent")  # full search # this allows me to pick step size what about evolutionary strategy? FullSearch, ConjugateGradient, ConjugateGradientFRPR, QuasiNewtonLBFGS, RSGDEachParameterApart, SimultaneousPerturbation, CMAEvolutionStrategy.
    parameter_object.SetParameter("Registration", "MultiResolutionRegistration")
    parameter_object.SetParameter("Metric",
                                  "AdvancedKappaStatistic")  # "AdvancedNormalizedCorrelation") #AdvancedNormalizedCorrelation") #maybe AdvancedMattesMutualInformation
    parameter_object.SetParameter("FixedInternalImagePixelType", "float")
    parameter_object.SetParameter("MovingInternalImagePixelType", "float")
    parameter_object.SetParameter("FixedImageDimension", "2")
    parameter_object.SetParameter("MovingImageDimension", "2")
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

    # get registration metirc for 180 and original data
    result_image, result_transform_parameters = itk.elastix_registration_method(fixed_image=fixed_image_feature,
                                                                                moving_image=moving_image_feature,
                                                                                parameter_object=parameter_object,
                                                                                log_to_console=True, log_to_file=True,
                                                                                output_directory=localsubjectpath)

    return result_transform_parameters



def part_in_whole_registration(whole_image, template):
    """
    This finds where a given template is located within a given whole image. Works for 2D or 3D images

    Input:
    whole_image: The whole image
    template: The template image

    Output:
    whole_image_template: The whole image that corresponds to the template
    XYZ_temp = the corrdinates of where the template is located within the whole image

    see: https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
    """

    # https: // scikit - image.org / docs / dev / auto_examples / features_detection / plot_template.html  # sphx-glr-auto-examples-features-detection-plot-template-py

    # get max value center point
    # ij = np.unravel_index(np.argmax(result), result.shape)
    # x, y = ij[::-1] #this is center X Y value
    # htemp, wtemp = template.shape
    # rect = plt.Rectangle((x, y), wtemp, htemp, edgecolor='r', facecolor='none')

    # Run registration with no rotations
    px_z = field_z - stack_z + stack.shape[0] / 2 - 0.5
    mini_stack = stack[max(0, int(round(px_z - rigid_zrange))): int(round(
        px_z + rigid_zrange))]
    corrs = np.stack([skimage.feature.match_template(whole_image, template, pad_input=True) for s in
                      mini_stack])
    smooth_corrs = scipy.ndimage.gaussian_filter(corrs, 0.7)

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

    # XYZ_temp =

    # whole_imaage_template =

    return whole_imaage_template, XYZ_temp


# todo prewhitten data before phase correatlion?


def crop_Z(image, list_manual_Z_crop, Z, extra_figs, localsubjectpath):
    """
    This crops a given image with defined y_start, y_end, x_start, x_end values. These are read in from list_manual_Z_crop.Can only crop along X and Y
        Args:
        -	image: the image to apply cropping to this can be 2D or 3D
        -   list_manual_Z_crop: the path to the excel file with cropping parameters for a given Z
        -   Z: Z value to search for given cropping parameters
        -   extra_figs: this is true to make figures, false to not make figures
        - localsubjectpath: this is the path location to save figures

        Returns:
        -	image_crop: this is the cropped image. After cropping image is zero padded. this will match 2d or 3D dimensions in image
    """
    # if list_manual_Z_crop is given then crop image before registration below
    if np.any(list_manual_Z_crop) is not None:
        # load data
        npcsv = np.genfromtxt(list_manual_Z_crop, delimiter=',', encoding='utf8', dtype=np.str)
        # get Z values --> find Z in first row somewhere  --> add .lower to can look at captial or not
        namaes = np.char.lower(npcsv[0])  # names is first row
        values = np.zeros([npcsv.shape[0] - 1, npcsv.shape[1]])
        for i in range(1, npcsv.shape[0]):
            vi = i - 1
            values[vi][:] = list(map(int, npcsv[i]))
        # find row with Z, 'Y_start', 'Y_end', 'X_start', 'X_end'
        zstr = 'z'
        Zrow = np.flatnonzero(np.core.defchararray.find(namaes, zstr) != -1)
        if Zrow.size <= 0:
            warnings.warn(message="WARNING:No Z colum found in list_manual_Z_crop csv file")
        YSstr = 'y_start'
        y_start_row = np.flatnonzero(np.core.defchararray.find(namaes, YSstr) != -1)
        if y_start_row.size <= 0:
            warnings.warn(message="WARNING:No y_start colum found in list_manual_Z_crop csv file")
        YEstr = 'y_end'
        y_end_row = np.flatnonzero(np.core.defchararray.find(namaes, YEstr) != -1)
        if y_end_row.size <= 0:
            warnings.warn(message="WARNING:No y_end colum found in list_manual_Z_crop csv file")
        XSstr = 'x_start'
        x_start_row = np.flatnonzero(np.core.defchararray.find(namaes, XSstr) != -1)
        if x_start_row.size <= 0:
            warnings.warn(message="WARNING:No x_start colum found in list_manual_Z_crop csv file")
        XEstr = 'x_end'
        x_end_row = np.flatnonzero(np.core.defchararray.find(namaes, XEstr) != -1)
        if x_end_row.size <= 0:
            warnings.warn(message="WARNING:No x_end colum found in list_manual_Z_crop csv file")
        Z_values = values[:, Zrow[0]]
        XS_values = values[:, x_start_row[0]]
        XE_values = values[:, x_end_row[0]]
        YS_values = values[:, y_start_row[0]]
        YE_values = values[:, y_end_row[0]]
        # check if Z  are in list
        if np.any(Z == Z_values):
            src_Z_index = np.where(Z_values == Z)
            Y_startS = YS_values[src_Z_index]
            Y_endS = YE_values[src_Z_index]
            X_startS = XS_values[src_Z_index]
            X_endS = XE_values[src_Z_index]
        else:  # no crop in  image
            Y_startS = 0
            Y_endS = image.shape[0] - 1
            X_startS = 0
            X_endS = image.shape[1] - 1
        if image.ndim == 3:
            for i in range(image.shape[2]):
                image_crop_one = image[Y_startS:Y_endS, X_startS:X_endS, i]
                image_crop_one = np.expand_dims(image_crop_one, axis=-1)
                if i != 0:
                    image_crop = np.concatenate((image_crop, image_crop_one), axis=2)
                else:
                    image_crop = image_crop_one
        if image.ndim == 2:
            image_crop = image[Y_startS:Y_endS, X_startS:X_endS]
        # pad cropped image with zeros until size of image
        dim = 1
        [image, image_crop] = zero_pad(image, image_crop, dim)
        dim = 0
        [image, image_crop] = zero_pad(image, image_crop, dim)
        if extra_figs:
            plt.figure()
            plt.imshow(image_crop)  # , norm=matplotlib.colors.LogNorm(vmin=-1, vmax=1))
            plt.savefig(localsubjectpath + 'Z=' + str(Z) + '_cropped_image.png', format='png')
            plt.close()
    else:
        image_crop = image
    return image_crop


# todo add if phase correlatioin fails (based on??) then do feature based registration instead of phase correlation --> in case not mounted properly

def registration_Z(src3d, src3d_denoise, des3d_denoise, src3d_feature, des3d_feature, count_shiftZ, shiftZ, angleZ,
                   apply_transform, rigid_2d3d, error_overlap, find_rot, degree_thres, denoise_all, max_Z_proj_affine,
                   seq_dir, maxZshift_percent, Z, list_180_Z_rotation, auto_180_Z_rotation, localsubjectpath,
                   Z_reg_denoise_or_feature):
    """
    This registers 2D to 3D along Z axis
        Args:
        -   src3d: raw source data to transform
        -	src3d_feature, des3d_feature: the source and destination feature map, used to register rigid
        -   src3d_denoise, des3d_denoise: the source and destination deionised data, used to affine
        -   count_shiftZ: the number of countZ to load from saved array if needed
        -   shiftZ, angleZ: saved shifitZ data translation and angle
        -   apply_transform: True if run saved transformations, false otherwise
        -   rigid_2d3d: True if run rigid registration, false otherwise
        -   error_overlap,
        -   find_rot: set to true if find rotation, other wise false
        -   degree_thres,
        -   denoise_all: set to ture to apply to all images, false if max projection used
        -   max_Z_proj_affine: set to true for max projection
        -   seq_dir = this is 'top' if POS1 is above POS 0 and 'bottom' if POS 1 is below POS 0
        -   maxZshift_percent: "This is used place limits on the Z plane to Z plane translation. This is the percent of the image allowed for translation, if above this threhold set to 0 translation. The default is 25% of the image. (ex:--maxZshift_percent=25)(default:25)")
        -   Z: current Z value for file names
        -   list_180_Z_rotation: if user defined list of images to rotate 180 degrees
        -   auto_180_Z_rotation: Set to true to automatically find the 180 degree rotation
        -   localsubjectpath: file path to save files
        -   Z_reg_denoise_or_feature: this is set to 'feaure' or 'denoise'. This registeres eaither denosie or freautre input based on this input

        Returns:
        -	src3d_T_feature: source feature file shifted with registration values
        -   des3d_feature : destination feature file
        -   count_shiftZ, shiftZ:  registration shift
        -   src3d_T: source raw file shifted with registration values

    """

    # todo remove these extra names
    src3d_denoise_crop = src3d_denoise
    des3d_denoise_crop = des3d_denoise
    src3d_feature_crop = src3d_feature
    des3d_feature_crop = des3d_feature
    # todo remove these extra names

    # todo nice ppt on optial flow" https://github.com/aplyer/gefolki/blob/master/COREGISTRATION.pdf
    Z_max_shift0 = src3d.shape[0] * (maxZshift_percent / 100)
    Z_max_shift1 = src3d.shape[1] * (maxZshift_percent / 100)
    if denoise_all:
        if max_Z_proj_affine:
            src3d_denoise_one = np.max(src3d_denoise_crop, axis=2)  # todo change back to max?
            des3d_denoise_one = np.max(des3d_denoise_crop, axis=2)
        else:
            if seq_dir == 'top':
                src3d_denoise_one = src3d_denoise_crop[:, :, 0]
                des3d_denoise_one = des3d_denoise_crop[:, :, -1]
            elif seq_dir == 'bottom':
                src3d_denoise_one = src3d_denoise_crop[:, :, -1]
                des3d_denoise_one = des3d_denoise_crop[:, :, 0]
            else:
                warnings.warn(message="WARNING:opticalZ_dir variable not defined correctly")
    else:
        src3d_denoise_one = src3d_denoise_crop
        des3d_denoise_one = des3d_denoise_crop
    error_allZ = []
    if Z_reg_denoise_or_feature == 'denoise':
        src3d_reg = src3d_denoise_one
        des3d_reg = des3d_denoise_one
    elif Z_reg_denoise_or_feature == 'feature':
        src3d_reg = src3d_feature_crop
        des3d_reg = des3d_feature_crop
    if list_180_Z_rotation is not None:
        if Z in list_180_Z_rotation:
            # deffine theta as 180 degrees for transformation
            theta = 180
            # set auto_180_Z_rotation for this Z to false if true
            auto_180_Z_rotation = False
        else:
            theta = 0
    else:
        theta = 0
    theta_rad = math.radians(theta)
    if auto_180_Z_rotation:
        new_angle_rad = auto_detect_180(des3d_reg, src3d_reg, localsubjectpath)
        theta_rad = new_angle_rad
    if rigid_2d3d:
        if apply_transform:
            error = 0  # not calculated
            shift = shiftZ[count_shiftZ]
            angle_rad = angleZ[count_shiftZ]
            count_shiftZ = count_shiftZ + 1
        else:
            # find rotation
            if find_rot:
                angle_rad = phase_corr_rotation(des3d_reg, src3d_reg, degree_thres,
                                                theta_rad)
            else:
                angle_rad = 0.0
            # basic phase corr only
            shift, error, diffphase = skimage.registration.phase_cross_correlation(des3d_reg,
                                                                                   src3d_reg)
            if np.abs(shift[0]) > Z_max_shift1 or np.abs(shift[1]) > Z_max_shift0:
                warnings.warn(
                    message="WARNING: Z shift exceeded limits, shift X = {} shift y = {}. Seeting Z translation to zero.".format(
                        shift[1], shift[0]))
                shift[0] = 0
                shift[1] = 0
            print('Z Phase translation is {}'.format(shift))
            shiftZ.append(shift)
            angleZ.append(angle_rad)
        error_allZ.append(error)
        # calculate OVERALL SHIFT from inital guess + phase correlations
        SKI_Trans_all_Z = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=(angle_rad + theta_rad),
                                                                translation=(shift[1], shift[0]))
        # apply to image in src3d feature map
        src3d_T_feature = skimage.transform.warp(src3d_feature, SKI_Trans_all_Z._inv_matrix, order=0, mode='edge',
                                                 preserve_range=True)
        # this chnage back to unit 8 dtype from float 64, this and preserve range are needed to keep shift as binary image
        src3d_T_feature = src3d_T_feature.astype(src3d_feature.dtype)
        # loop comprehension to apply to all images in src3d
        src3d_T = [
            skimage.transform.warp(src3d[:, :, i], SKI_Trans_all_Z._inv_matrix, mode='edge')
            for i in range(src3d.shape[2])]
        if np.max(src3d_T) < 1:  # only if max less then one convert to unit
            src3d_T = skimage.img_as_uint(src3d_T, force_copy=False)
        else:
            src3d_T = np.array(src3d_T, dtype=src3d.dtype)
        src3d_T = np.transpose(src3d_T, axes=[1, 2, 0])
        # loop comprehension to apply to all images in src3d_denoise
        if denoise_all:
            src3d_T_denoise = [skimage.transform.warp(src3d_denoise[:, :, i], SKI_Trans_all_Z._inv_matrix, mode='edge')
                               for i in range(src3d_denoise.shape[2])]
            if np.max(src3d_T_denoise) < 1:  # only if max less then one convert to unit
                src3d_T_denoise = skimage.img_as_uint(src3d_denoise, force_copy=False)
            else:
                src3d_T_denoise = np.array(src3d_T_denoise, dtype=src3d_denoise.dtype)
            src3d_T_denoise = np.transpose(src3d_T_denoise, axes=[1, 2, 0])
        else:
            # APPLY SHIFT to denoise value
            src3d_T_denoise = skimage.transform.warp(src3d_denoise, SKI_Trans_all_Z._inv_matrix, order=0,
                                                     mode='edge', preserve_range=True)
            src3d_T_denoise = src3d_T_denoise.astype(src3d_denoise.dtype)
    else:  # --- Compute the optical flow
        if apply_transform:
            error = 0  # not calculated
            [v_shift, u_shift] = shiftZ[count_shiftZ]
            angle_rad = angleZ[count_shiftZ]
            rotation_trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=angle_rad,
                                                                   translation=(0, 0))
            count_shiftZ = count_shiftZ + 1
        else:
            # find rotation
            if find_rot:
                angle_rad = phase_corr_rotation(des3d_reg, src3d_reg, degree_thres, theta_rad)
            else:
                angle_rad = 0.0
            rotation_trans = skimage.transform.SimilarityTransform(matrix=None, scale=1,
                                                                   rotation=(angle_rad + theta_rad),
                                                                   translation=(0, 0))
            # phase correlation here --> then motion based
            shift, error, diffphase = skimage.registration.phase_cross_correlation(des3d_reg,
                                                                                   src3d_reg)
            if np.abs(shift[0]) > Z_max_shift1 or np.abs(shift[1]) > Z_max_shift0:
                warnings.warn(
                    message="WARNING: Z shift exceeded limits, shift X = {} shift y = {}. Seeting Z translation to zero.".format(
                        shift[1], shift[0]))
                shift[0] = 0
                shift[1] = 0
            print('Z Phase translation is {}'.format(shift))
            # calculate OVERALL SHIFT from inital guess + phase correlations
            Tran_Z_phase = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=(angle_rad + theta_rad),
                                                                 translation=(shift[1], shift[0]))
            # apply to image in src3d_denoise to use to calculate V and U
            src3d_denoise_phase_shift = skimage.transform.warp(src3d_reg, Tran_Z_phase._inv_matrix, order=0,
                                                               mode='edge', preserve_range=True)
            src3d_denoise_phase_shift = src3d_denoise_phase_shift.astype(src3d_reg.dtype)
            v, u = skimage.registration.optical_flow_tvl1(des3d_reg, src3d_denoise_phase_shift)
            # add phase and motion correction together
            v_shift = v + -(shift[0])
            u_shift = u + -(shift[1])
            # check V and U are correctly added for all images
            src3d_denoise_phase_shift_raw = skimage.transform.warp(src3d_denoise_phase_shift,
                                                                   np.array([row_coords + v, col_coords + u]), order=0,
                                                                   mode='nearest', preserve_range=True)
            src3d_reg_check = skimage.transform.warp(src3d_reg, np.array([row_coords + v_shift, col_coords + u_shift]),
                                                     order=0, mode='nearest', preserve_range=True)
            testzeros = src3d_denoise_phase_shift_raw - src3d_reg_check
            if testzeros.max() != testzeros.min():
                warnings.warn(
                    message="WARNING: optical registration not added correctly to phase correlation for Z = {}".format(
                        Z))
            shiftZ.append([v_shift, u_shift])
            angleZ.append((angle_rad + theta_rad))
        error_allZ.append([error])
        # --- Use the estimated optical flow for registration
        nr, nc = src3d_feature.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        # apply to image in src3d feature map
        src3d_feature_rot = skimage.transform.warp(src3d_feature, rotation_trans._inv_matrix, order=0, mode='edge',
                                                   preserve_range=True)
        src3d_feature_rot = src3d_feature_rot.astype(src3d_feature.dtype)
        src3d_T_feature = skimage.transform.warp(src3d_feature_rot,
                                                 np.array([row_coords + v_shift, col_coords + u_shift]), order=0,
                                                 mode='nearest', preserve_range=True)
        # this chnage back to unit 8 dtype from float 64, this and preserve range are needed to keep shift as binary image
        src3d_T_feature = src3d_T_feature.astype(src3d_feature.dtype)
        # loop comprehension to apply to all images in src3d
        src3d_T_rot = [
            skimage.transform.warp(src3d[:, :, i], rotation_trans._inv_matrix, order=0, mode='edge',
                                   preserve_range=True) for i
            in range(src3d.shape[2])]
        if np.max(src3d_T_rot) < 1:  # only if max less then one convert to unit
            src3d_T_rot = skimage.img_as_uint(src3d_T_rot, force_copy=False)
        else:
            src3d_T_rot = np.array(src3d_T_rot, dtype=src3d.dtype)
        src3d_T_rot = np.transpose(src3d_T_rot, axes=[1, 2, 0])
        src3d_T = [skimage.transform.warp(src3d_T_rot[:, :, i], np.array([row_coords + v_shift, col_coords + u_shift]),
                                          mode='nearest') for i in range(src3d.shape[2])]
        if np.max(src3d_T) < 1:  # only if max less then one convert to unit
            src3d_T = skimage.img_as_uint(src3d_T, force_copy=False)
        else:
            src3d_T = np.array(src3d_T, dtype=src3d.dtype)
        src3d_T = np.transpose(src3d_T, axes=[1, 2, 0])
        # apply to denoise
        if denoise_all:
            src3d_T_denoise_rot = [
                skimage.transform.warp(src3d_denoise[:, :, i], rotation_trans._inv_matrix, order=0, mode='edge',
                                       preserve_range=True) for i in range(src3d_denoise.shape[2])]
            if np.max(src3d_T_denoise_rot) < 1:  # only if max less then one convert to unit
                src3d_T_denoise_rot = skimage.img_as_uint(src3d_denoise, force_copy=False)
            else:
                src3d_T_denoise_rot = np.array(src3d_T_denoise_rot, dtype=src3d_denoise.dtype)
            src3d_T_denoise_rot = np.transpose(src3d_T_denoise_rot, axes=[1, 2, 0])
            src3d_T_denoise = [
                skimage.transform.warp(src3d_T_denoise_rot, np.array([row_coords + v_shift, col_coords + u_shift]),
                                       order=0, mode='nearest', preserve_range=True) for i in
                range(src3d_denoise.shape[2])]
            if np.max(src3d_T_denoise) < 1:  # only if max less then one convert to unit
                src3d_T_denoise = skimage.img_as_uint(src3d_denoise, force_copy=False)
            else:
                src3d_T_denoise = np.array(src3d_T_denoise, dtype=src3d_denoise.dtype)
            src3d_T_denoise = np.transpose(src3d_T_denoise, axes=[1, 2, 0])
        else:
            src3d_T_denoise_rot = skimage.transform.warp(src3d_denoise, rotation_trans._inv_matrix, order=0,
                                                         mode='edge',
                                                         preserve_range=True)
            src3d_T_denoise_rot = src3d_T_denoise_rot.astype(src3d_denoise.dtype)
            src3d_T_denoise = skimage.transform.warp(src3d_T_denoise_rot,
                                                     np.array([row_coords + v_shift, col_coords + u_shift]), order=0,
                                                     mode='nearest', preserve_range=True)
            # this chnage back to unit 8 dtype from float 64, this and preserve range are needed to keep shift as binary image
            src3d_T_denoise = src3d_T_denoise.astype(src3d_denoise.dtype)

    return src3d_T, src3d_T_feature, src3d_T_denoise, count_shiftZ, shiftZ, error_allZ


"""

def affine_registation(destination, source, degree_thres, theta_rad):
    # AFFINE REGISTRATION
    import torch
    from torch import optim

    # Create field grid (height x width x 2)
    grid = registration.create_grid(field.shape)

    # Create torch tensors
    stack_ = torch.as_tensor(stack, dtype=torch.float32)
    field_ = torch.as_tensor(field, dtype=torch.float32)
    grid_ = torch.as_tensor(grid, dtype=torch.float32)

    # Define parameters and optimizer
    linear = torch.nn.Parameter(torch.eye(3)[:, :2])  # first two columns of rotation matrix
    translation = torch.nn.Parameter(torch.tensor([rig_x, rig_y, rig_z]))  # translation vector
    affine_optimizer = optim.Adam([{'params': linear, 'lr': lr_linear},
                                   {'params': translation, 'lr': lr_translation}])

    # Optimize
    for i in range(affine_iters):
        # Zero gradients
        affine_optimizer.zero_grad()

        # Compute gradients
        pred_grid = registration.affine_product(grid_, linear, translation)  # w x h x 3
        pred_field = registration.sample_grid(stack_, pred_grid)
        corr_loss = -(pred_field * field_).sum() / (torch.norm(pred_field) *
                                                    torch.norm(field_))
        print('Corr at iteration {}: {:5.4f}'.format(i, -corr_loss))
        corr_loss.backward()

        # Update
        affine_optimizer.step()

    # Save em (originals will be modified during non-rigid registration)
    affine_linear = linear.detach().clone()
    affine_translation = translation.detach().clone()

    return

def crop_Z(image, list_manual_Z_crop, Z):
    This crops a given image with defined y_start, y_end, x_start, x_end values. These are read in from list_manual_Z_crop
        Args:
        -	image: the image to apply cropping to
        -   list_manual_Z_crop: the path to the excel file with cropping parameters for a given Z
        -   Z: Z value to search for given cropping parameters

        Returns:
        -	image_cropped: this is the cropped image. After cropping image is zero padded.

    # if list_manual_Z_crop is given then crop image before registration below
    if np.any(list_manual_Z_crop) is not None:
        # load data
        # list_manual_Z_crop='/Users/kaleb/Documents/CSHL/2d_3D_linear_reg/test_crop.csv'
        npcsv = np.genfromtxt(list_manual_Z_crop, delimiter=',', encoding='utf8', dtype=np.str)
        # get Z values --> find Z in first row somewhere  --> add .lower to can look at captial or not
        namaes = np.char.lower(npcsv[0])  # names is first row
        values = np.zeros([npcsv.shape[0] - 1, npcsv.shape[1]])
        for i in range(1, npcsv.shape[0]):
            vi = i - 1
            values[vi][:] = list(map(int, npcsv[i]))
        # find row with Z, 'Y_start', 'Y_end', 'X_start', 'X_end'
        zstr = 'z'
        Zrow = np.flatnonzero(np.core.defchararray.find(namaes, zstr) != -1)
        if Zrow.size <= 0:
            warnings.warn(message="WARNING:No Z colum found in list_manual_Z_crop csv file")
        YSstr = 'y_start'
        y_start_row = np.flatnonzero(np.core.defchararray.find(namaes, YSstr) != -1)
        if y_start_row.size <= 0:
            warnings.warn(message="WARNING:No y_start colum found in list_manual_Z_crop csv file")
        YEstr = 'y_end'
        y_end_row = np.flatnonzero(np.core.defchararray.find(namaes, YEstr) != -1)
        if y_end_row.size <= 0:
            warnings.warn(message="WARNING:No y_end colum found in list_manual_Z_crop csv file")
        XSstr = 'x_start'
        x_start_row = np.flatnonzero(np.core.defchararray.find(namaes, XSstr) != -1)
        if x_start_row.size <= 0:
            warnings.warn(message="WARNING:No x_start colum found in list_manual_Z_crop csv file")
        XEstr = 'x_end'
        x_end_row = np.flatnonzero(np.core.defchararray.find(namaes, XEstr) != -1)
        if x_end_row.size <= 0:
            warnings.warn(message="WARNING:No x_end colum found in list_manual_Z_crop csv file")
        Z_values = values[:, Zrow[0]]
        XS_values = values[:, x_start_row[0]]
        XE_values = values[:, x_end_row[0]]
        YS_values = values[:, y_start_row[0]]
        YE_values = values[:, y_end_row[0]]
        # check if Z  are in list
        if  np.any(Z == Z_values):
            src_Z_index = np.where(Z_values == Z)
            Y_startS = YS_values[src_Z_index]
            Y_endS = YE_values[src_Z_index]
            X_startS = XS_values[src_Z_index]
            X_endS = XE_values[src_Z_index]
        else: #no crop in  image
            Y_startS = 0
            Y_endS = image.shape[0] -1
            X_startS = 0
            X_endS = image.shape[1] -1

        if image.ndim == 3:
            #todo fix this line of code
            image_crop = image[Y_startS:Y_endS][X_startS:X_endS][:]
        if image.ndim == 2:
            image_crop = image[Y_startS:Y_endS][X_startS:X_endS]
        # pad the Zplane in the X then Y
        dim = 1
        [src3d_denoise_crop, des3d_denoise_crop] = func2d3d.zero_pad(src3d_denoise_crop, des3d_denoise_crop, dim)
        dim = 0
        [src3d_denoise_crop, des3d_denoise_crop] = func2d3d.zero_pad(src3d_denoise_crop, des3d_denoise_crop, dim)
        src3d_feature_crop = src3d_feature[Y_startS:Y_endS][X_startS:X_endS]
        des3d_feature_crop = des3d_feature[Y_startD:Y_endD][X_startD:X_endD]
        # pad the Zplane in the X then Y
        dim = 1
        [src3d_feature_crop, des3d_feature_crop] = func2d3d.zero_pad(src3d_feature_crop, des3d_feature_crop, dim)
        dim = 0
        [src3d_feature_crop, des3d_feature_crop] = func2d3d.zero_pad(src3d_feature_crop, des3d_feature_crop, dim)
        Y_shift_crop = X_startS - X_startD  # X_endS  X_startD  X_endD
        X_shift_crop = Y_startS - Y_startD  # Y_endS Y_startD Y_endD
    else:
        image_cropped = image
    return image_cropped
def remodeling_Z(src3d, src3d_denoise, des3d_denoise, src3d_feature, des3d_feature, count_shiftZ, shiftZ, angleZ,
                   apply_transform, rigid_2d3d, error_overlap, find_rot, degree_thres, denoise_all, max_Z_proj_affine, seq_dir):
                   
    #https://github.com/marsmarcin/Generate-3D-models-from-2D-images
      
    #https://github.com/search?q=itk+2d+3d+registration&type=repositories               
                   
                   
    #https://github.com/InsightSoftwareConsortium/ITKTwoProjectionRegistration            
                   
                   
    #ML based? https://github.com/hzxie/Pix2Vox

    #https://github.com/gift-surg/NiftyMIC

    #https://github.com/elhuhdron/emdrp

    #https://pypi.org/project/neutompy/ this uses : https://www.astra-toolbox.com/docs/algs/BP.html

    #https://itk.org/ITKExamples/src/IO/ImageBase/Creade3DFromSeriesOf2D/Documentation.html

    This registers 2D to 3D along Z axis
        Args:
        -	rc3d_feature, des3d_feature: the source and destination feature map, used to register rigid
        -   src3d_denoise, des3d_denoise: the source and destination deionised data, used to affine
        -   count_shiftZ: the number of countZ to load from saved array if needed
        -   shiftZ: saved shifitZ data
        -   apply_transform: True if run saved transformations, false otherwise
        -   rigid_2d3d: True if run rigid registration, false otherwise

        Returns:
        -	src3d_T_feature: source feature file shifted with registration values
        -   des3d_feature : destination feature file
        -   ount_shiftZ, shiftZ:  registration shift
        -   src3d_T: source raw file shifted with registration values


    https://github.com/InsightSoftwareConsortium/ITKElastix/blob/master/examples/ITK_Example05_PointRegistration.ipynb
    
    #maybe take segmentation --> find centroids --> then apply this  to it?  
    
    

    #use:  Contour propagation

    #maybe https://github.com/opencv/opencv_contrib/tree/master/modules/sfm
    
    #note for this segment is refeering to a 2D slice
    #https://www.slicer.org/
    #https://github.com/SlicerRt/SegmentRegistration#segment-registration
    
    
    #https://github.com/SlicerRt/SegmentRegistration#segment-registration



    return src3d_T, src3d_T_feature, src3d_T_denoise, count_shiftZ, shiftZ, error_allZ




    #run rolling ball to remove uneven background intensity #https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rolling_ball.html#sphx-glr-auto-examples-segmentation-plot-rolling-ball-py
    #convert image to float for calculations below?
    denoised1 = skimage.util.img_as_float(denoised1)
    # calibrate rolling_ball
    if denoised1.ndim == 3:
        parameter_ranges_rolling_ball = {
            'kernel': [restoration.ellipsoid_kernel((3, 3, 1), 0.1), restoration.ellipsoid_kernel((5, 5, 1), 0.1),
                       restoration.ellipsoid_kernel((11, 11, 1), 0.1), restoration.ellipsoid_kernel((21, 21, 1), 0.1),
                       restoration.ellipsoid_kernel((301, 301, 1), 0.1)]}
    elif denoised1.ndim == 2:
        parameter_ranges_rolling_ball = {
            'kernel': [restoration.ellipsoid_kernel((3, 3), 0.1), restoration.ellipsoid_kernel((5, 5), 0.1),
                       restoration.ellipsoid_kernel((11, 11), 0.1), restoration.ellipsoid_kernel((21, 21), 0.1),
                       restoration.ellipsoid_kernel((301, 301), 0.1)]}
    _, (parameters_tested_rolling_ball, losses_rolling_ball) = calibrate_denoiser(denoised1, restoration.rolling_ball, parameter_ranges_rolling_ball, extra_output=True)
    best_parameters_rolling_ball = parameters_tested_rolling_ball[np.argmin(losses_rolling_ball)]
    background = _invariant_denoise(denoised1, restoration.rolling_ball, denoiser_kwargs=best_parameters_rolling_ball)
    denoised = denoised1 - background




        # if blank then run gaussian #todo check if we still have blank images not being low pass filtered???
    #    if A.max() < blank_thres * A.mean():
    #    denoised = skimage.filters.difference_of_gaussians(denoised1, 1, FFT_max_gaussian)
    # else:
    #    denoised = denoised1
    # run gussian after above that way of can be taken care above it will be
    # run difference_of_gaussians and find optimal value to get ride of low freq signal
    # parameter_ranges_gaussians = {'low_sigma': np.arange(1, 2, 1), 'high_sigma': np.arange(1, 10, 0.5)} #this gives option  of no difference of gausian b/c high and low sigma could both be 1
    # _, (parameters_tested_gaussians, losses_gaussians) = calibrate_denoiser(denoised1, skimage.filters.difference_of_gaussians, denoise_parameters=parameter_ranges_gaussians, extra_output=True)
    # best_parameters_gaussians = parameters_tested_gaussians[np.argmin(losses_gaussians)]
    # denoised = _invariant_denoise(denoised1, skimage.filters.difference_of_gaussians, denoiser_kwargs=best_parameters_gaussians)
    # todo ONLY RUN ABOVE IF losses_gaussians is less then denoised1 file
    
def denoise_old(A, FFT_max_gaussian, high_thres):
    This denoise data with difference of gausian , wavlet, non-local means and TV norm
        Args:
        -	A Numpy array to be denoised
        - high_thres: this is the threshold to very high values
        -FFT_max_gaussian: this if high sigma for difference of gausian
        Returns:
        -	denoised: denoised image

    # todo try and autoencoder here instead? https://www.pyimagesearch.com/2020/02/24/denoising-autoencoders-with-keras-tensorflow-and-deep-learning/
    # first make autoendoer with MNIST and add noise --> then apply this encoder to this dataset: https://towardsdatascience.com/image-noise-reduction-in-10-minutes-with-convolutional-autoencoders-d16219d2956a

    sigma_est_ori = np.mean(skimage.restoration.estimate_sigma(A, multichannel=False))
    print(f"estimated noise standard deviation = {sigma_est_ori}")

    # todo optimize denoise? --> basically try range of options?
    #todo add optimizer option in code??
    # if what value sigma_est_ori do we skip denoise? https://scikit-image.org/docs/dev/auto_examples/filters/plot_j_invariant_tutorial.html#sphx-glr-auto-examples-filters-plot-j-invariant-tutorial-py

    #todo this to find BEST denoising algorithum with BEST parameters

    #see: https://scikit-image.org/docs/dev/auto_examples/filters/plot_j_invariant_tutorial.html#sphx-glr-auto-examples-filters-plot-j-invariant-tutorial-py
    #Comparing Different Classes of Denoiser

    # remove extremely high values --> remove hair
    if (A > (high_thres * A.mean())).any():
        warnings.warn(message='High intensity Image detected')
        high_thres_dec = high_thres / 100
        # remove whole connected image to this
        int_thres = (high_thres * A.mean())
        area_thres = A.shape[1] * high_thres_dec
        A = remove_large_obj(A, area_thres, int_thres, FFT_max_gaussian, high_thres_dec)

    # use band pass filtering to remove shadow
    filt_A = skimage.filters.difference_of_gaussians(A, 1, FFT_max_gaussian)
    # denoise with wavlet
    denoise_wave = skimage.restoration.denoise_wavelet(filt_A, sigma=None, wavelet='haar', mode='soft',
                                                       wavelet_levels=None, multichannel=False, convert2ycbcr=False,
                                                       method='BayesShrink', rescale_sigma=True)
    # Non - localmeans
    sigma_est = np.mean(skimage.restoration.estimate_sigma(denoise_wave, multichannel=False))
    patch_kw = dict(patch_size=5,  # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=False)
    # this is used for salt and peper
    denoise_nl_mean = skimage.restoration.denoise_nl_means(denoise_wave, h=1.15 * sigma_est, fast_mode=False,
                                                           preserve_range=True, **patch_kw)

    #todo run this wihtout TV norm? will this work? are we denoisng too much on some images?

    # TV norm --> is this too MUCH???
    # this ALSO used for salt and peper
    denoise = skimage.restoration.denoise_tv_chambolle(denoise_nl_mean, weight=0.1, eps=0.0002, n_iter_max=200,
                                                       multichannel=False)
    # differance of gaussian again.. why?
    filt_B = skimage.filters.difference_of_gaussians(denoise, 1, FFT_max_gaussian)
    denoised = filt_B
    # clip negative values? --> this removes to much of images durring denoise
    # convert to image if need be
    # if denoised.max() < 1:
    #    denoised = skimage.img_as_uint(denoised, force_copy=False)
    return denoised


    # this is a way to try a bunch of wavlet denoise options and use best:
    # https://scikit-image.org/docs/dev/auto_examples/filters/plot_j_invariant_tutorial.html#sphx-glr-auto-examples-filters-plot-j-invariant-tutorial-py
    # low_values_flags = denoise < denoise.mean()/2
    # denoise[low_values_flags] = 0
    # this removes small bright and dark spots
    # med = skimage.filters.median(denoise, skimage.morphology.disk(7))
    # filt_B = skimage.filters.difference_of_gaussians(med, 1, FFT_max_gaussian)
    # skimage.morphology.area_closing(denoise, area_threshold=3, connectivity=1, parent=None, tree_traverser=None)
    # skimage.morphology.area_opening(image, area_threshold=3, connectivity=1, parent=None, tree_traverser=None)
    # filt_B = skimage.filters.difference_of_gaussians(denoise, 1, FFT_max_gaussian)
    # test = denoise - filt_B
    # segmentation: https://scikit-image.org/docs/0.18.x/auto_examples/segmentation/plot_morphsnakes.html#sphx-glr-auto-examples-segmentation-plot-morphsnakes-py
    # or try antss denoise: ants.denoise_image
    # imagedenoise = ants.denoise_image(imagenoise, ants.get_mask(image))
    # filt_A = skimage.filters.difference_of_gaussians(A, 1, FFT_max_gaussian)
    # A_ant = ants.from_numpy(data=filt_A)
    # denoiseA=ants.denoise_image(A_ant)
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    # thresholds = threshold_multiotsu(wave_denoise)
    # Using the threshold values, we generate the three regions.
    # regions = np.digitize(wave_denoise, bins=thresholds)
    # set stuff below mean to zero? --> for some reason this works best
    # low_values_flags = A < A.mean()
    # A[low_values_flags] = 0

def findoverlay(A, B, diroverlap):
    import skimage
    This finds the overlay given A and B
        Args:
        -	A: 3D set of images to overlap source (moving image)
        -   B: 3D set of images to overlap target
        -   diroverlap: the direction of overlap, the direction the source image moves --> up, down, left or right

        Returns:
        -	target_overlap: pixel number of overlay
        -   overlay_var: the variance of the overlay across 2D images ---> should be pretty small s
    # get mean of A and B

    A_mean = np.max(A, axis=2)
    B_mean = np.max(B, axis=2)
    # A_denosie=denoise(A_mean, FFT_max_gaussian, high_thres)
    # B_denosie=denoise(B_mean, FFT_max_gaussian, high_thres)
    # shift, error = feature_CV2(B_denosie, A_denosie)
    # for all sets of 2D images
    maxover_ind = []
    target_overlap = []
    # for i in range(A.shape[2]):
    image_similarity = []
    if diroverlap == 'up' or diroverlap == 'down':
        scan_amount = round(int((int(A.shape[0]))) / 2)
    elif diroverlap == 'left' or diroverlap == 'right':
        scan_amount = round(int((int(A.shape[1]))) / 2)
    # for all pixels shift along diroverlap
    for ii in range(7, scan_amount):  # for only 1/2 of image
        # shift image by one pixel along overlap direction
        # find image overlap ORDER H,W,D (Y,X,Z)
        if diroverlap == 'up':  # LEFT --> for B start at end of Y direction, for A start at beginning of Y direction (FIX X)
            A_over = A_mean[:ii, :]
            B_over = B_mean[-ii:, :]
        elif diroverlap == 'down':  # RIGHT --> for B start at beginning of Y direction , for A start at end of Y direction (FIX X)
            A_over = A_mean[-ii:, :]
            B_over = B_mean[:ii, :]
        elif diroverlap == 'left':  # UP --> for B start and end of X direction, for A start at beginning ofr X direction (FIX Y)
            A_over = A_mean[:, :ii]
            B_over = B_mean[:, -ii:]
        elif diroverlap == 'right':  # downs --> for B start at beginning of X direction , for A start at end of X direction (FIX Y)
            A_over = A_mean[:, -ii:]
            B_over = B_mean[:, :ii]
        # calculate overlap image similarity
        if ii < 7:
            image_similarity_one = skimage.metrics.structural_similarity(A_over, B_over, win_size=3)
        else:
            image_similarity_one = skimage.metrics.structural_similarity(A_over, B_over)
        image_similarity.append(image_similarity_one)
    # find maximum
    maxover_ind.append(image_similarity.index(max(image_similarity)))
    # target_overlap = maxover_ind  # np.median(maxover_ind)
    # define overlap based on derivative
    grad_image = np.gradient(image_similarity)
    target_overlap = [int(round(np.mean([grad_image.argmax(), grad_image.argmin()])))]

    return target_overlap




       # [target_overlapY] = findoverlay(srcZ, desZ, diroverlap)

        #[target_overlapX] = findoverlay(source_good, destination_good, diroverlap)

def initial_transform(target_overlap, diroverlap):
    This gets the initial transform to apply as first guess for ICP
        Args:
        -	overlap: overlap number of pixels
        -   diroverlap: the direction of the overlap

        Returns:
        -	Int_Trans: initial transform
    # NOTE ARRAY in H, W, D or (Y,X,Z) order

    # find image overlap
    if diroverlap == 'left' or diroverlap == 'right':
        X_shift = target_overlap
        Y_shift = 0
        SKI_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                          translation=(X_shift, Y_shift))
        Int_Trans = SKI_Trans._inv_matrix
    elif diroverlap == 'up' or diroverlap == 'down':
        X_shift = 0
        Y_shift = target_overlap
        SKI_Trans = skimage.transform.SimilarityTransform(matrix=None, scale=1, rotation=0,
                                                          translation=(X_shift, Y_shift))
        Int_Trans = SKI_Trans._inv_matrix
    # todo change for 3d (is this needed?) --> not needed for skimage.transform.warp --> is needed for o3d.pipelines.registration.registration_icp ?
    # MODIFY FOR 3d
    # Int_Trans= ??? trans_init_array
    # Int_Trans=abs(Int_Trans) #get rid of any negatives

    return Int_Trans


    #last resort... try feature detection again?????




    #TRY ANTS here???? #https://readthedocs.org/projects/antspy/downloads/pdf/latest/
    des3d_denoise_rotated = skimage.transform.rotate(des3d_denoise, 180)
    ref = des3d_denoise_rotated#io.imread('some_multiframe_image.tif')
    mov = src3d_denoise#io.imread('another_multiframe_image.tif')
    im1=ref.astype('float32')
    im2=mov.astype('float32')
    im1ant = ants.from_numpy(data=im1)
    im2ant = ants.from_numpy(data=im2)#Translation
    mytx = ants.registration(fixed=im1ant, moving=im2ant, type_of_transform="DenseRigid", initial_transform=None, outprefix="test", dimension=2)  # “Similarity”
    # apply transfrom to unsegmented data
    p2_reg_ants = ants.apply_transforms(fixed=im1ant, moving=im2ant, transformlist=mytx['fwdtransforms'])
    # convert back to numpy
    p2_reg=p2_reg_ants.numpy()
    plt.figure()
    plt.imshow(p2_reg)







    #try? https://pypi.org/project/pystackreg/
    from pystackreg import StackReg
    from skimage import io
    #ImportError: numpy.core.multiarray failed to import
    #here need to upgrade numpy
    des3d_feature_rotated = skimage.transform.rotate(des3d_feature, 180)
    ref = des3d_feature_rotated#io.imread('some_multiframe_image.tif')
    mov = src3d_feature#io.imread('another_multiframe_image.tif')
    # Rigid Body transformation
    sr = StackReg(StackReg.RIGID_BODY)
    out_rot = sr.register_transform(ref, mov)


    #THIS DOESNT REALLY WORK... this method consitantly works for rotation of SAME image (without translation)
    import imregpoc
    #try result = imregpoc.imregpoc(ref,cmp)
    import imreg_dft as ird

    #make constraints dictionary for scale
    myconstraints = {
        "scale": [1, 0.05], #this is a scale of 1 without and stardev
        "angle": [180, 5]
    }

    #TRY 180 degrees:
    des3d_denoise_rotated = skimage.transform.rotate(des3d_denoise, 180)
    result = ird.imreg.similarity(des3d_denoise_rotated, src3d_denoise, numiter=10, constraints=myconstraints)


    #try zero degrees:


    # "angle": [180, 10] #this is a scale of 1 without and stardev
    #use best option here??? 0 +/- error OR 180 +/- error????



    #try: result = ird.imreg.similarity(des3d_feature_rotated, src3d_feature, numiter=3) #this gives angle of 'angle': -177.72196137651014,.... pretty good
    #can we specificy scale?/ --scale MEAN[,STD] of 1?
    #result = ird.similarity(des3d_denoise_rotated, src3d_denoise, numiter=10)



    documentation on page 16: https://imreg-dft.readthedocs.io/_/downloads/en/latest/pdf/

    This finds rotation, scale and translation with fft based phase correlation based on: https://scikit-image.org/docs/0.18.x/auto_examples/registration/plot_register_rotation.html#sphx-glr-auto-examples-registration-plot-register-rotation-py
        Args:
        -	source,destination: the source and destination  array to register
        Returns:
        -	error: registration error
        -   shift : registration shift
        -   recovered_angle : registration rotation
        -   scale : registration scale



    # window images
    wimage = destination * skimage.filters.window('hann', destination.shape)
    rts_wimage = source * skimage.filters.window('hann', source.shape)

    #Register rotation and scaling on a translated image - Part 2
    # work with shifted FFT magnitudes
    image_fs = np.abs(scipy.fftpack.fftshift(scipy.fftpack.fft2(wimage)))
    rts_fs = np.abs(scipy.fftpack.fftshift(scipy.fftpack.fft2(rts_wimage)))
    # Create log-polar transformed FFT mag images and register
    shape = image_fs.shape
    radius = shape[0] // 8  # only take lower frequencies
    warped_image_fs = skimage.transform.warp_polar(image_fs, radius=radius, output_shape=shape, scaling='log', order=0)
    warped_rts_fs = skimage.transform.warp_polar(rts_fs, radius=radius, output_shape=shape, scaling='log', order=0)
    warped_image_fs = warped_image_fs[:shape[0] // 2, :]  # only use half of FFT
    warped_rts_fs = warped_rts_fs[:shape[0] // 2, :]
    shifts, error, phasediff = skimage.registration.phase_cross_correlation(warped_image_fs,warped_rts_fs, upsample_factor=10)
    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    scale = np.exp(shiftc / klog)

    return shifts, recovered_angle, scale, error


    # convert feature map into X, Y list of points for destination
    des3d_index = des3d_feature.nonzero()
    des3d_index_reX = des3d_index[1].reshape(des3d_index[1].__len__(), 1)
    des3d_index_reY = des3d_index[0].reshape(des3d_index[0].__len__(), 1)
    des3d_points = np.hstack((des3d_index_reX, des3d_index_reY))
    #des3d_points = des3d_points.T
    # convert feature map into X, Y list of points for  source
    src3d_index = src3d_feature.nonzero()
    src3d_index_reX = src3d_index[1].reshape(src3d_index[1].__len__(), 1)
    src3d_index_reY = src3d_index[0].reshape(src3d_index[0].__len__(), 1)
    src3d_points = np.hstack((src3d_index_reX, src3d_index_reY))
    #src3d_points = src3d_points.T
    
imprort ICP #version 2.1.1

#TODO TWEEK THIS!!!

#See: https://engineering.purdue.edu/kak/distICP/ICP-2.1.1.html#ICP
import ICP

icp = ICP.ICP(
           binary_or_color = "binary",
           pixel_correspondence_dist_threshold = 40,
           auto_select_model_and_data = 1,
           calculation_image_size = 200,
           iterations = 16,
           model_image = "triangle1.jpg",
           data_image = "triangle2.jpg",
       )

icp.extract_pixels_from_binary_image("model")
icp.extract_pixels_from_binary_image("data")
icp.icp()
icp.display_images_used_for_binary_image_icp()
icp.display_results_as_movie()
icp.cleanup_directory()



#Iterative Closest Point (ICP) SLAM example
#author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı


import math

import matplotlib.pyplot as plt
import numpy as np

#  ICP parameters
EPS = 0.0001
MAX_ITER = 100

show_animation = True


def icp_matching(previous_points, current_points):
    "
    Iterative Closest Point matching
    - input
    previous_points: 2D points in the previous frame
    current_points: 2D points in the current frame
    - output
    R: Rotation matrix
    T: Translation vector
    "
    H = None  # homogeneous transformation matrix

    dError = np.inf
    preError = np.inf
    count = 0

    while dError >= EPS:
        count += 1

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(previous_points[0, :], previous_points[1, :], ".r")
            plt.plot(current_points[0, :], current_points[1, :], ".b")
            plt.plot(0.0, 0.0, "xr")
            plt.axis("equal")
            plt.pause(0.1)

        indexes, error = nearest_neighbor_association(previous_points, current_points)
        Rt, Tt = svd_motion_estimation(previous_points[:, indexes], current_points)
        # update current points
        current_points = (Rt @ current_points) + Tt[:, np.newaxis]

        dError = preError - error
        print("Residual:", error)

        if dError < 0:  # prevent matrix H changing, exit loop
            print("Not Converge...", preError, dError, count)
            break

        preError = error
        H = update_homogeneous_matrix(H, Rt, Tt)

        if dError <= EPS:
            print("Converge", error, dError, count)
            break
        elif MAX_ITER <= count:
            print("Not Converge...", error, dError, count)
            break

    R = np.array(H[0:2, 0:2])
    T = np.array(H[0:2, 2])

    return R, T


def update_homogeneous_matrix(Hin, R, T):

    H = np.zeros((3, 3))

    H[0, 0] = R[0, 0]
    H[1, 0] = R[1, 0]
    H[0, 1] = R[0, 1]
    H[1, 1] = R[1, 1]
    H[2, 2] = 1.0

    H[0, 2] = T[0]
    H[1, 2] = T[1]

    if Hin is None:
        return H
    else:
        return Hin @ H


def nearest_neighbor_association(previous_points, current_points):

    # calc the sum of residual errors
    delta_points = previous_points - current_points
    d = np.linalg.norm(delta_points, axis=0)
    error = sum(d)

    # calc index with nearest neighbor assosiation
    d = np.linalg.norm(np.repeat(current_points, previous_points.shape[1], axis=1)
                       - np.tile(previous_points, (1, current_points.shape[1])), axis=0)
    indexes = np.argmin(d.reshape(current_points.shape[1], previous_points.shape[1]), axis=1)

    return indexes, error


def svd_motion_estimation(previous_points, current_points):
    pm = np.mean(previous_points, axis=1)
    cm = np.mean(current_points, axis=1)

    p_shift = previous_points - pm[:, np.newaxis]
    c_shift = current_points - cm[:, np.newaxis]

    W = c_shift @ p_shift.T
    u, s, vh = np.linalg.svd(W)

    R = (u @ vh).T
    t = pm - (R @ cm)

    return R, t


def main():
    print(__file__ + " start!!")

    # simulation parameters
    nPoint = 1000
    fieldLength = 50.0
    motion = [0.5, 2.0, np.deg2rad(-10.0)]  # movement [x[m],y[m],yaw[deg]]

    nsim = 3  # number of simulation

    for _ in range(nsim):

        # previous points
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py))

        # current points
        cx = [math.cos(motion[2]) * x - math.sin(motion[2]) * y + motion[0]
              for (x, y) in zip(px, py)]
        cy = [math.sin(motion[2]) * x + math.cos(motion[2]) * y + motion[1]
              for (x, y) in zip(px, py)]
        current_points = np.vstack((cx, cy))

        R, T = icp_matching(previous_points, current_points)
        print("R:", R)
        print("T:", T)


if __name__ == '__main__':
    main()




def icp_feature(src3d_feature, des3d_feature):
    "Calculate the ICP between two feature images"
    # convert feature map into X, Y list of points for destination
    des3d_index = des3d_feature.nonzero()
    des3d_index_reX = des3d_index[1].reshape(des3d_index[1].__len__(), 1)
    des3d_index_reY = des3d_index[0].reshape(des3d_index[0].__len__(), 1)
    des3d_points = np.hstack((des3d_index_reX, des3d_index_reY))
    des3d_points = des3d_points.T
    # convert feature map into X, Y list of points for  source
    src3d_index = src3d_feature.nonzero()
    src3d_index_reX = src3d_index[1].reshape(src3d_index[1].__len__(), 1)
    src3d_index_reY = src3d_index[0].reshape(src3d_index[0].__len__(), 1)
    src3d_points = np.hstack((src3d_index_reX, src3d_index_reY))
    src3d_points=src3d_points.T


    R, T = icp_matching(src3d_points, des3d_points)





    #save feature map as an image
    from PIL import Image
    im_src3d = Image.fromarray(src3d_feature)
    im_src3d.save("src3d_feature.png")
    im_des3d = Image.fromarray(des3d_feature)
    im_des3d.save("des3d_feature.png")


    #NOTE: this uses model_im = Image.open(model_image) to open images

    # applly ICP
    icp = ICP.ICP(
        binary_or_color="binary",
        pixel_correspondence_dist_threshold=10,
        auto_select_model_and_data=0, #this way model and data are defined by user (=0)
        calculation_image_size=1000, #set this to size of image... which direction??? this is max dim so set to larger of the two
        iterations=20,
        model_image="src3d_feature.png",
        data_image="des3d_feature.png",
    )
    icp.extract_pixels_from_binary_image("model")
    icp.extract_pixels_from_binary_image("data")
    icp.icp()

    icp.display_images_used_for_binary_image_icp()
    icp.display_results_as_movie()
    icp.cleanup_directory()
    

    # REMOVE feature map as an image


    # get transltion and rotation values
    translation = icp.T
    rotation = icp.R
    # output as shift values? (does rotation matter?)
    shift = np.zeros(3)
    shift[0] = closest_translation_y
    shift[1] = closest_translation_x

    #error = rmsdiff(source, source_transformed, destination)
    shift[2] = closest_rot_angle

    # show results
    plt.plot(des3d_points[:, 0], des3d_points[:, 1], 'rx', label='reference points')
    plt.plot(src3d_points[:, 0], src3d_points[:, 1], 'b1', label='points to be aligned')
    plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
    plt.legend()
    plt.show()



    return shift, error

# EXTRA
    #todo for testing
    #get feature map for each optical slice with filter and segment
    src3d_feature_all = [func2d3d.preproc(np.squeeze(src3d[:, :, i]), args.FFT_max_gaussian, args.checkerboard_size, args.seg_interations, args.seg_smooth) for i in range(src3d.shape[2])]
    des3d_feature_all = [func2d3d.preproc(np.squeeze(des3d[:, :, i]), args.FFT_max_gaussian, args.checkerboard_size, args.seg_interations, args.seg_smooth) for i in range(des3d.shape[2])]
    #convert nested list from comhension for loop into np array
    src3d_feature_all = np.array(src3d_feature_all) #todo do we need a dtype input here??
    des3d_feature_all = np.array(des3d_feature_all)

    src3d_feature_all = []
    for i in range(src3d.shape[2]):
        print(i)
        src3d_feature_one = func2d3d.preproc(np.squeeze(src3d[:, :, i]), args.FFT_max_gaussian, args.checkerboard_size,args.seg_interations, args.seg_smooth)
        src3d_feature_all = np.concatenate((src3d_feature_all, src3d_feature_one), axis=2)
    #src3d_feature_all = [func2d3d.preproc(np.squeeze(src3d[:, :, i]), args.FFT_max_gaussian, args.checkerboard_size, args.seg_interations,args.seg_smooth) for i in range(src3d.shape[2])]
    src3d_feature_all = np.array(src3d_feature_all)
    print(src3d_feature_all.shape)
    print(src3d_feature_all.dtype)
    src3d_feature = src3d_feature_all[:, :, 0]
    print(src3d_feature.shape)
    src3d_feature2 = src3d_feature_all[:, :, 0]
    #todo for testing

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



    # TODO add mutal inifromation registration? regional mutual information with elastic? mutal infromation is basically histogram matching...
    # or ICP instead of phase correlatin on feautre?

    I guess we have to use ANTS here
    import ants

    im1=im1.astype('float32')
    im2=im2.astype('float32')
    im1ant = ants.from_numpy(data=im1)
    im2ant = ants.from_numpy(data=im2)#Translation
    mytx = ants.registration(fixed=im1ant, moving=im2ant, type_of_transform="Rigid", initial_transform=None, outprefix="test", dimension=2)  # “Similarity”
    # apply transfrom to unsegmented data
    p2_reg_ants = ants.apply_transforms(fixed=im1ant, moving=im2ant, transformlist=mytx['fwdtransforms'])
    # convert back to numpy
    p2_reg=p2_reg_ants.numpy()
    plt.figure()
    plt.imshow(p2_reg)


    import pyelastix
    ELASTIX_PATH= '/Users/kaleb/Downloads/elastix-5.0.1-mac 2'#need path to ELASTIX_PATH from https://github.com/SuperElastix/elastix/releases/tag/5.0.1

    im1=np.ascontiguousarray(im1)
    im2=np.ascontiguousarray(im2)
    # Get params and change a few values
    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = 200
    params.FinalGridSpacingInVoxels = 10
    # Apply the registration (im1 and im2 can be 2D or 3D)
    im1_deformed, field = pyelastix.register(im1, im2, params)
# ICP: A Python implementation of the [Iterative closest point][1] algorithm for 2D point clouds, based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans" by F. Lu and E. Milios.

def euclidean_distance(point1, point2):
    Euclidean distance between two points.
    :param point1: the first point as a tuple (a_1, a_2, ..., a_n)
    :param point2: the second point as a tuple (b_1, b_2, ..., b_n)
    :return: the Euclidean distance

    a = np.array(point1)
    b = np.array(point2)

    return np.linalg.norm(a - b, ord=2)


def point_based_matching(point_pairs):

    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.

    :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
    :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points


    x_mean = 0
    y_mean = 0
    xp_mean = 0
    yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None

    for pair in point_pairs:
        (x, y), (xp, yp) = pair

        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = 0
    s_y_yp = 0
    s_x_yp = 0
    s_y_xp = 0
    for pair in point_pairs:
        (x, y), (xp, yp) = pair

        s_x_xp += (x - x_mean) * (xp - xp_mean)
        s_y_yp += (y - y_mean) * (yp - yp_mean)
        s_x_yp += (x - x_mean) * (yp - yp_mean)
        s_y_xp += (y - y_mean) * (xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean * math.cos(rot_angle) - y_mean * math.sin(rot_angle))
    translation_y = yp_mean - (x_mean * math.sin(rot_angle) + y_mean * math.cos(rot_angle))

    return rot_angle, translation_x, translation_y


def icp(reference_points, points, max_iterations=100, distance_threshold=0.3, convergence_translation_threshold=1e-3,
        convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):

    An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
    of N 2D (reference) points.

    :param reference_points: the reference point set as a numpy array (N x 2)
    :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
    :param max_iterations: the maximum number of iteration to be executed
    :param distance_threshold: the distance threshold between two points in order to be considered as a pair
    :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                              transformation to be considered converged
    :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                               to be considered converged
    :param point_pairs_threshold: the minimum number of point pairs the should exist
    :param verbose: whether to print informative messages about the process (default: False)
    :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
             transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2


    transformation_history = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)

    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = nbrs.kneighbors(points)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

        # if only few point pairs, stop process
        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if verbose:
                print('No better solution can be found!')
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s, c]])
        aligned_points = np.dot(points, rot.T)
        aligned_points[:, 0] += closest_translation_x
        aligned_points[:, 1] += closest_translation_y

        # update 'points' for the next iteration
        points = aligned_points

        # update transformation history
        transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

        # check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                and (abs(closest_translation_x) < convergence_translation_threshold) \
                and (abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    return transformation_history, points, closest_rot_angle, closest_translation_x, closest_translation_y

def remove_large_obj(A, extra_figs, area_thres, int_thres):
This removes large and high intensity objects from image

    #area_thres = 10000 or X% of image
    #int_thres = 10000 or X% greater then median
    #21743 is with object adn 1015 is without object

    #TODO ISSUE HERE IS needs lots of inputs and BOX is not completly around area, just use threshold instead

    #remove large objects (if needed)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from skimage import data
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.measure import label, regionprops
    from skimage.morphology import closing, square, binary_dilation, disk, square, area_opening, opening
    from skimage.color import label2rgb

    A = imarray3D[:, :, 1]
    FFT_max_gaussian=10
    filt_A = skimage.filters.difference_of_gaussians(A, 1, FFT_max_gaussian)
    # denoise with wavlet this might not have much of an effect
    denoise = skimage.restoration.denoise_wavelet(filt_A, multichannel=False, rescale_sigma=True)
    # set stuff below mean to zero? --> for some reason this works best
    low_values_flags = denoise < np.median(denoise)
    denoise[low_values_flags] = 0
    
    image = A
    # apply threshold
    thresh = 2*np.median(A)#1000 #threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)

    dialted = binary_dilation(cleared, selem=None)

    #dialted = binary_dilation(dialted1, selem=square(300))



    # label image regions
    label_image = label(dialted)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    if extra_figs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(A)

    for region in regionprops(label_image, intensity_image=A):
        # take regions with large enough areas
        if region.area >= area_thres and region.max_intensity >= int_thres:
            #define this as a mask to REMOVE from image
            #for this box set value to zero? mean? median?
            (min_row, min_col, max_row, max_col) = region.bbox
            high_values_flags = np.array(np.zeros(A.shape), dtype=bool)
            high_values_flags[min_row:max_row, min_col:max_col] = True
            A[high_values_flags] = np.median(A)
            plt.figure()
            plt.imshow(A)
            if extra_figs:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
    if extra_figs:
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    try:
        x
    except NameError:
        mask_A = A
    return mask_A

from sklearn.neighbors import NearestNeighbors
# import cv2
import pandas as pd
import copy
# import open3d as o3d

def transform_data(A_xy, T, dim_num):
    '''
    This Applies the rotation and translation transform to image
        Args:
        -	A: Numpy array image (in 2D) OR 3D
        -   T: transformation for 2D or 3D image

        Returns:
        -	A_T: transformed image

    '''

    rotated = skimage.transform.warp(text, tform)

    # seporate XY data from Color data
    # if 2D or 3D
    A_corr = A_xy[:, :-1]
    A_color = A_xy[:, -1]
    # limit T
    Ts = T[:dim_num, :dim_num]
    # transform XY or XYZ data
    A_T = np.dot(Ts, A_corr.T).T

    # ARE WE MISSING Z deppth in registarting 2D data together here???? .... does this matter?
    # what about 4th dem in XYZ?

    # ROUND X Y or X Y Z to whole pixel resolution, makes image conversion possible, and sub-pixel resolution is not useful
    A_Tr = A_T.round()
    # get rid of negative zero
    A_Tr = A_Tr + 0
    # add color value back to transformed data
    A_TrC = np.zeros(A_xy.shape)
    A_TrC[:, :-1] = A_Tr
    A_TrC[:, -1] = A_color
    return A_TrC

def img_to_XYC(Aimg):
    '''
    This converts Aimg into XYC format
        Args:
        -	A: Numpy array image (in 2D) OR 3D

        Returns:
        -	A_xyz: data in XYC or XYZ C

    '''
    dim_num = Aimg.ndim
    # if 2d image --> gives XY C format
    if dim_num == 2:
        # pre allocate
        A_xyz = np.ones(((Aimg.shape[0] * Aimg.shape[1]), (dim_num + 1)))
        i = 0  # start counting at 0
        for X in range(Aimg.shape[0] - 1):
            for Y in range(Aimg.shape[1] - 1):
                A_xyz[i, 0] = X
                A_xyz[i, 1] = Y
                A_xyz[i, 2] = Aimg[X, Y]
                i = i + 1
    # if 3D image --> gives XYZ C format
    if dim_num == 3:
        # pre alloccate
        A_xyz = np.ones(((Aimg.shape[0] * Aimg.shape[1] * Aimg.shape[2]), (dim_num + 1)))
        i = 0  # start counting at 0
        for X in range(Aimg.shape[0] - 1):
            for Y in range(Aimg.shape[1] - 1):
                for Z in range(Aimg.shape[2] - 1):
                    A_xyz[i, 0] = X
                    A_xyz[i, 1] = Y
                    A_xyz[i, 2] = Z
                    A_xyz[i, 3] = Aimg[X, Y, Z]
                    i = i + 1
    return A_xyz


def XYC_to_img(AB_xyz_avg):
    '''
    This converts AB_xyz into image matrix
        Args:
        -	AB_xyz_avg: Numpy array image 3D, ONLY 3D inputs, if multiple 2D concat they need to be assigned Z value prior to this function

        Returns:
        -	AB_img: transformed image
    '''
    # convert XY or XYZ into matrix, this is REALLY SLOW CODE to do make this code faster somehow maybe reshape color into 3d matrix, then no for loop??? #
    dim_num_1 = AB_xyz_avg.shape[1]
    dim_num = dim_num_1 - 1
    if dim_num == 2:
        i = 0
        df = pd.DataFrame(AB_xyz_avg[:, :dim_num], columns=list('XY'))
        AB_img = np.zeros([int(AB_xyz_avg[:, 0].max() + 1), int(AB_xyz_avg[:, 1].max() + 1)])
        for idx, coord in df.iterrows():
            x, y = tuple(coord)
            AB_img[int(x), int(y)] = int(AB_xyz_avg[i, dim_num])
            i = i + 1
    # if 3d
    if dim_num == 3:
        i = 0
        df = pd.DataFrame(AB_xyz_avg[:, :dim_num], columns=list('XYZ'))
        AB_img = np.zeros(
            [int(AB_xyz_avg[:, 0].max() + 1), int(AB_xyz_avg[:, 1].max() + 1), int(AB_xyz_avg[:, 2].max() + 1)])
        for idx, coord in df.iterrows():
            x, y, z = tuple(coord)
            AB_img[int(x), int(y), int(z)] = int(AB_xyz_avg[i, dim_num])
            i = i + 1
    return AB_img


        #TODO steps to try: 1st: segmenation,
    # PREPROCESS edge detection #https://web.cs.ucdavis.edu/~hamann/WangZhaoCappsHamannPaperDraftFrom07212015.pdf
    # apply Canny edge detector to these image for iterative closest point


    # what if we zero out anything below mean???
    # A[ A < A.mean()] = 0
    #or maybe just X percent?


    # use median to get ride of salt and peper
    filt_A_im = np.uint8(filt_A)
    filt_SM_A = skimage.filters.rank.median(filt_A_im, skimage.morphology.disk(median_size))
    Aimg = np.uint8(filt_SM_A)
    edgesA3 = auto_canny(Aimg)
    # for B
    filt_B = skimage.filters.difference_of_gaussians(B, 1, FFT_max_gaussian)

    # B[ B < B.mean()] = 0

    # use median to get ride of salt and peper
    filt_B_im = np.uint8(filt_B)
    filt_SM_B = skimage.filters.rank.median(filt_B_im, skimage.morphology.disk(median_size))
    Bimg = np.uint8(filt_SM_B)
    edgesB3 = auto_canny(Bimg)


# function for linear registration between 2 images ---> correlation?
# use intersection over union? http://ronny.rest/tutorials/module/localization_001/iou/#:~:text=Intersect%20over%20Union%20(IoU)%20is,calculating%20IoU%20is%20as%20follows.
# for IOU would need to find MAX IOU for all x and y?
# or try Lucas-Kanade method ?
# this is linear registration for partially overlaping images
# https://www.sciencedirect.com/science/article/pii/S1877705811019254
# or use SVD for rigid trnasform??? #http://nghiaho.com/?page_id=671
# search whole image or block of image to find maximum overlap --> this should be around 15% (in all directions top/bottom left/right)
# expectation is close to 100% reproduability in 15% overlap



def XYZ_to_image(XYZ_data):
    # x, y, z are integers... does not work for fraacctional values, aka values with transform applied
    n = np.max(XYZ_data)  # set array as largest value in one dimension
    binary_array = np.zeros([n] * XYZ_data.shape[1])
    # Builds coordinates
    # if 3d
    if XYZ_data.shape[1] == 3:
        df = pd.DataFrame(XYZ_data, columns=list('XYZ'))
        for idx, coord in df.iterrows():
            x, y, z = tuple(coord)
            binary_array[x, y, z] = 1
    # if 2d
    if XYZ_data.shape[1] == 2:
        df = pd.DataFrame(XYZ_data, columns=list('XY'))
        for idx, coord in df.iterrows():
            x, y = tuple(coord)
            binary_array[x, y] = 1
    return binary_array


def image_to_XYZ(binary_array):
    # if 3d
    if binary_array.ndim == 3:
        [x, y, z] = np.nonzero(binary_array)
        XYZ_data = np.stack([x, y, z], axis=1)

    # if 2D
    if binary_array.ndim == 2:
        [x, y] = np.nonzero(binary_array)
        XYZ_data = np.stack([x, y], axis=1)

    return XYZ_data


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # Dassert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def icp(A, B, init_pose=None, max_iterations=200, tolerance=0.001):

    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation, this is the inital position of image A (source)
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge

    # assert A.shape == B.shape #why is this needed???? ... point cloud can have different length and still registar?

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)  # this is the transpose of a given array
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)  # this is the dot product of the init_pose and the source image
        # ad an initial position to the destination image?

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i

def HPF(image, window_size):
    '''
    Calculates HPF and removes low frequencies of window size
    Input:
      image: Image to remove lower frequencies
      window_size: size of frequencies to remove
    Returns:
      image_HPF: image without lower frequency changes
    '''
    # fft to convert the image to freq domain
    f = np.fft.fft2(image)
    # shift the center
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows / 2, cols / 2
    # remove the low frequencies by masking with a rectangular window of size 60x60
    # High Pass Filter (HPF)
    # remove low freq
    window_size_half = window_size / 2
    fshift[int(crow - window_size_half):int(crow + window_size_half),
    int(ccol - window_size_half):int(ccol + window_size_half)] = 0

    fshift[int(crow - window_size_half):int(crow + window_size_half),
    int(ccol - window_size_half):int(ccol + window_size_half)] = 0

    # shift back (we shifted the center before)
    f_ishift = np.fft.ifftshift(fshift)
    # inverse fft to get the image back
    img_back = np.fft.ifft2(f_ishift)
    # get rid of complex part
    img_back = np.abs(img_back)
    # noramlize
    norm = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
    image_HPF = norm * 255

    '''
    # OR WHAT IF WE BAND PASS INSTead????
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.data import gravel
    from skimage.filters import difference_of_gaussians, window
    from scipy.fftpack import fftn, fftshift

    image = gravel()
    wimage = image * window('hann', image.shape)  # window image to improve FFT
    filtered_image = difference_of_gaussians(image, 1, 12)
    filtered_wimage = filtered_image * window('hann', image.shape)
    im_f_mag = fftshift(np.abs(fftn(wimage)))
    fim_f_mag = fftshift(np.abs(fftn(filtered_wimage)))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Original Image')
    ax[0, 1].imshow(np.log(im_f_mag), cmap='magma')
    ax[0, 1].set_title('Original FFT Magnitude (log)')
    ax[1, 0].imshow(filtered_image, cmap='gray')
    ax[1, 0].set_title('Filtered Image')
    ax[1, 1].imshow(np.log(fim_f_mag), cmap='magma')
    ax[1, 1].set_title('Filtered FFT Magnitude (log)')
    plt.show()
    '''

    return image_HPF


# see: https://johnwlambert.github.io/icp/
# Iterative closest point WITH KNOWN --> SWITCH TO THIS https://github.com/ClayFlannigan/icp/blob/master/icp.py


# or should i just use  cross correlation --> use this for slice to volume? and ICP for image overlap??
# step 1: overlap all image with ICP
# step 2: use cross coreltation for each slice ridgid shift ?  --> or is this taken care by 3d ICP?

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def image_overlap(AB_xyz):
 
    Calculates the percent overlap in two images and similarity of overlap  --> function called from within find_rigid_alignment?

        Args:
        -	AB_xy: Numpy array of X Y Z coordinates of two images to be combined

        Returns:
        -	overlap_perc: overlap of two images
        -	image_sim: similarity of overlap area
        -   AB_xy_avg: Numpy array of X Y Z coordinates of two images with average values used for overlapped area

    # find the  X Y Z values that are repeated here (therefore in both source and destination)
    AB_xyz_noC = AB_xyz[:, :-1]
    AB_xyz_COLOR = AB_xyz[:, -1]
    unq, unq_idx, unq_cnt = np.unique(AB_xyz_noC, return_inverse=True, return_counts=True, axis=0)
    AB_xy_avg[:, :(unq.shape[1])] = unq  # here fill in XYZ values into new variable
    ndim = AB_xy_avg.shape[1] - 1  # get number of dimensions
    # preallocate new_matrix
    AB_xy_avg = np.zeros([unq.shape[0], AB_xyz.shape[1]])
    overlapXYZ = np.where(unq_cnt > 1)
    index_repeat_1 = np.zeros([np.shape(overlapXYZ)[1], ndim + 1])
    index_repeat_2 = np.zeros([np.shape(overlapXYZ)[1], ndim + 1])
    ir = 0
    #todo seed this for loop up -- make for loop in C code? with map?
    #use list comprehension of below...

    AB_xy_avg = [(np.mean(AB_xyz_COLOR[np.where(unq_idx == i)])) for i in range(len(unq))] #newlist = map(np.mean(AB_xyz_COLOR[np.where(unq_idx == i)]), oldlist)
    repeatedXY=np.where(unq_cnt > 1)
    index_repeat_1C= [AB_xyz_COLOR[np.where(unq_idx == i)[0][0]] for i in range(len(repeatedXY))]
    index_repeat_2C= [AB_xyz_COLOR[np.where(unq_idx == i)[0][1]] for i in range(len(repeatedXY))]
    index_repeat_XY = [AB_xy_avg[i] for i in range(len(repeatedXY))]


    for i in range(len(unq)): #if cant do list comprehension here do map??
        all_index = np.where(unq_idx == i)
        AB_xy_avg[i, ndim] = np.mean(AB_xyz_COLOR[all_index])
        # find repeated area --> make this a for loop instead with list comperhension or map(), filter() and reduce()
        if unq_cnt[i] > 1:  # so if more the 1 xyz color value --> use filter() instead of if????
            index_repeat_1[ir, :-1] = AB_xy_avg[i, :-1]
            index_repeat_1[ir, -1] = AB_xyz_COLOR[all_index[0][0]]
            index_repeat_2[ir, :-1] = AB_xy_avg[i, :-1]
            index_repeat_2[ir, -1] = AB_xyz_COLOR[all_index[0][1]]
            ir += 1
    overlap_perc =  #percentage of pixels that had 2 values to





    # calculate percent overlap --> aka how much image shifted --> so is this what % of image_overlap is image?
    [x1ovr, y1ovr] = im1_overlap.shape
    [x2ovr, y2ovr] = im2_overlap.shape
    [x1, y1] = im1.shape
    [x2, y2] = im2.shape
    percent_overlap1 = ((x1ovr * y1ovr) / (x1 * y1)) * 100
    percent_overlap2 = ((x2ovr * y2ovr) / (x2 * y2)) * 100
    percent_overlap = np.mean([percent_overlap1, percent_overlap2])
    # calcualte image similarity

    image_similarity = skimage.metrics.structural_similarity(im1_overlap,
                                                             im2_overlap)  # this gives structral similarity of the two images

    return AB_xy_avg, overlap_perc, image_sim

def auto_canny(image, sigma=0.33):
    # if 2d
    if image.ndim == 2:
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

    # if 3D
    if image.ndim == 3:
        edged = np.zeros(image.shape)
        for i in range(image.shape[2]):
            image_one = image[:, :, i]
            # compute the median of the single channel pixel intensities
            v = np.median(image_one)
            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edge_one = cv2.Canny(image_one, lower, upper)
            edged[:, :, i] = edge_one
            del edge_one, image_one

    # return the edged image
    return edged

#use this for global registraion? ICP for local refinement?
def phase_corr_registation(A,B) #maybe try: https://github.com/YoshiRi/ImRegPOC

    #try: https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
    
    #https://medium.com/analytics-vidhya/image-stitching-with-opencv-and-python-1ebd9e0a6d78 
    
    
    
    import cv2
    import numpy as np
    img_ = cv2.imread('/Users/kaleb/Documents/CSHL/ML_basecalling/code/data/113/img_000000000_Il-A_022.tif')
    #img_ = cv2.imread('/Users/kaleb/Documents/CSHL/ML_basecalling/code/data/213/img_000000000_Il-A_022.tif')
    #img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img = cv2.imread('/Users/kaleb/Documents/CSHL/ML_basecalling/code/data/213/img_000000000_Il-A_022.tif')
    #img = cv2.imread('original_image_right.jpg')
    #img = cv2.resize(img, (0,0), fx=1, fy=1)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # find key points
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))
    #FLANN_INDEX_KDTREE = 0
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks = 50)
    #match = cv2.FlannBasedMatcher(index_params, search_params)
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.03*n.distance:
            good.append(m)
    draw_params = dict(matchColor=(0,255,0),
                           singlePointColor=None,
                           flags=2)
    img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
    #cv2.imshow("original_image_drawMatches.jpg", img3)
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        #cv2.imshow("original_image_overlapping.jpg", img2)
    else:
        print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
    dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0],0:img.shape[1]] = img
    cv2.imshow("original_image_stitched.jpg", dst)
    def trim(frame):
        #crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        #crop top
        if not np.sum(frame[-1]):
            return trim(frame[:-2])
        #crop top
        if not np.sum(frame[:,0]):
            return trim(frame[:,1:])
        #crop top
        if not np.sum(frame[:,-1]):
            return trim(frame[:,:-2])
        return frame
    cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    #cv2.imsave("original_image_stitched_crop.jpg", trim(dst))









    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage.feature import register_translation
    from skimage.feature.register_translation import _upsampled_dft
    from scipy.ndimage import fourier_shift


    #get phase shift from imaages
    shift, error, diffphase = register_translation(image, offset_image)




    image = data.camera()
    shift = (-22.4, 13.32)
    # The shift corresponds to the pixel offset relative to the reference image
    offset_image = fourier_shift(np.fft.fftn(image), shift)
    offset_image = np.fft.ifftn(offset_image)
    print(f"Known offset (y, x): {shift}")



    # pixel precision first
    shift, error, diffphase = register_translation(image, offset_image)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(offset_image.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    # Show the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Cross-correlation")

    plt.show()

    print(f"Detected pixel offset (y, x): {shift}")

    # subpixel precision
    shift, error, diffphase = register_translation(image, offset_image, 100)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    ax2.imshow(offset_image.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')

    # Calculate the upsampled DFT, again to show what the algorithm is doing
    # behind the scenes.  Constants correspond to calculated values in routine.
    # See source code for details.
    cc_image = _upsampled_dft(image_product, 150, 100, (shift * 100) + 75).conj()
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Supersampled XC sub-area")

    plt.show()

    print(f"Detected subpixel offset (y, x): {shift}")

    # EXTRA
    # rows, cols = img.shape #https://www.geeksforgeeks.org/python-opencv-affine-transformation/
    # dst = cv2.warpAffine(img, T, (cols, rows))
    
    # Make C a homogeneous representation of B
    # i = A_corr.shape[0]
    # C = np.ones((i , (dim_num + 1)))
    # C[:, 0:dim_num] = A_corr
    # Transform C
    # A_T = np.dot(T, C.T).T #why for identiify is A_corr and A_Tr not the same? transform does werid stuff?


"""
