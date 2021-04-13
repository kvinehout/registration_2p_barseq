#!/bin/bash

#this converts 2D barseq data into 2D barseq data
#python Main_2d_3d.py --remotesubjectpath='/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/' --localsubjectpath='/grid/zador/home/vinehout/code/2d_3D_linear_reg/' --X_dir='right' --Y_dir='top' --image_type='Il-A' --seq_dir='top' --opticalZ_dir='top' --input_overlap=0.15 --server='zadorstorage2.cshl.edu' --user='imageguest' --password='zadorlab' --FFT_max_gaussian=10 --error_overlap=0.10 --blank_thres=1.5 --checkerboard_size=6 --seg_interations=35 --seg_smooth=3 --POS_folder_X -6 -3 --POS_folder_Y -2 -1 --POS_folder_Z 0 -6 --extra_figs=True --max_Z_proj_affine=False --rigid_2d3d=True

pwd

#multiple remote paths to add register folders together
python Main_2d_3d.py --remotesubjectpath='/home/imagestorage/xichen/20210308BaylorSample1/lectin_131415_1/' --remotesubjectpath='/home/imagestorage/xichen/20210308BaylorSample1/lectin_161718_1/' --localsubjectpath='/Users/kaleb/Documents/CSHL/2d_3D_linear_reg/' --image_type='Il-A' --X_dir='right' --Y_dir='top' --seq_dir='top' --opticalZ_dir='bottom' --input_overlap=0.15 --server='zadorstorage4.cshl.edu' --user='imageguest' --password='zadorlab' --output='registration_save' --extra_figs=False --max_Z_proj_affine=False --find_rot=True --rigid_2d3d=False --denoise_all=True




#--rigid_2d3d=True  --apply_transform=True --saved_transforms_path='/grid/zador/home/vinehout/code/2d_3D_linear_reg/registration/Il-A'


#this converts apply barseq registration transformation to new data
#python Main_2d_3d.py --remotesubjectpath='/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/' --localsubjectpath='/grid/zador/home/vinehout/code/2d_3D_linear_reg/' --X_dir='right' --Y_dir='top' --image_type='Il-A' --seq_dir='top' --opticalZ_dir='top' --input_overlap=0.15 --server='zadorstorage2.cshl.edu' --user='imageguest' --password='zadorlab' --FFT_max_gaussian=10 --error_overlap=0.10 --blank_thres=1.5 --checkerboard_size=6 --seg_interations=35 --seg_smooth=3 --POS_folder_X -6 -3 --POS_folder_Y -2 -1 --POS_folder_Z 0 -6 --apply_transform=True --saved_transforms_path='/Users/kaleb/Documents/CSHL/ML_basecalling/code/2d_3D_linear_reg/registration/Il-A'


#this registers 3D barseq from above to 3D 2 photon imaging
#python Main_3d_3d.py --remotesubjectpath='/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/' --localsubjectpath='/grid/zador/home/vinehout/code/2d_3D_linear_reg/' --image_type='Il-A' --seq_dir='top' --input_overlap=0.15 --server='zadorstorage2.cshl.edu' --user='imageguest' --password='zadorlab' --FFT_max_gaussian=10 --error_overlap=0.10 --blank_thres=1.5
