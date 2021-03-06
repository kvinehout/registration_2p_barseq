#!/bin/bash

#this converts 2D barseq data into 2D barseq data
#python Main_2d_3d.py --remotesubjectpath='/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/' --localsubjectpath='/grid/zador/home/vinehout/code/2d_3D_linear_reg/' --X_dir='right' --Y_dir='top' --image_type='Il-A' --seq_dir='top' --opticalZ_dir='top' --input_overlap=0.15 --server='zadorstorage2.cshl.edu' --user='imageguest' --password='zadorlab' --FFT_max_gaussian=10 --error_overlap=0.10 --blank_thres=1.5 --checkerboard_size=6 --seg_interations=35 --seg_smooth=3 --POS_folder_X -6 -3 --POS_folder_Y -2 -1 --POS_folder_Z 0 -6 --extra_figs=True --max_Z_proj_affine=False --rigid_2d3d=True

#multiple remote paths to add register folders together

export TF_FORCE_GPU_ALLOW_GROWTH=true #this allows n2V to work with GPU???


#todo run on this

#zadorstorage2.cshl.edu:/home/cristiansoitu/vessels/20210517/slices/lectin_1/ file: P-Cy5


#zador lab sample 2, not as good as other sample
#python Main_2d_3d.py --remotesubjectpath='/home/imagestorage/xichen/20210428-2psample/' --localsubjectpath='/Users/kaleb/Documents/CSHL/2d_3D_linear_reg/' --image_type='P-Cy5' --X_dir='right' --Y_dir='top' --seq_dir='top' --opticalZ_dir='bottom' --input_overlap=0.15 --server='zadorstorage4.cshl.edu' --user='imageguest' --password='zadorlab' --output='registration_baylor2_2p' --extra_figs=True --max_Z_proj_affine=True --find_rot=True --rigid_2d3d=True --denoise_all=False

#baylor sample 1
#python Main_2d_3d.py --remotesubjectpath='/home/imagestorage/xichen/20210308BaylorSample1/lectin_131415_1/' --remotesubjectpath='/home/imagestorage/xichen/20210308BaylorSample1/lectin_161718_1/' --localsubjectpath='/Users/kaleb/Documents/CSHL/2d_3D_linear_reg/' --image_type='Il-A' --X_dir='right' --Y_dir='top' --seq_dir='top' --opticalZ_dir='bottom' --input_overlap=0.15 --server='zadorstorage4.cshl.edu' --user='imageguest' --password='zadorlab' --output='registration_save_for_2p' --extra_figs=True --max_Z_proj_affine=True --find_rot=True --rigid_2d3d=True --denoise_all=False

#baylor sample 2
python Main_2d_3d.py --remotesubjectpath='/home/imagestorage/xichen/20210504BCM25618/lectin_2/' --localsubjectpath='/Users/kaleb/Documents/CSHL/2d_3D_linear_reg/' --image_type='P-Cy5' --X_dir='right' --Y_dir='top' --seq_dir='top' --opticalZ_dir='bottom' --input_overlap=0.15 --server='zadorstorage4.cshl.edu' --user='imageguest' --password='zadorlab' --output='registration_baylor2_2p_Zreg_denoise_manual_180_log_norm' --extra_figs=True --max_Z_proj_affine=True --find_rot=True --rigid_2d3d=True --denoise_all=False --segment_otsu=True --auto_180_Z_rotation=False --list_180_Z_rotation 16 17 --Z_log_transform=True

#--rigid_2d3d=True  --apply_transform=True --saved_transforms_path='/grid/zador/home/vinehout/code/2d_3D_linear_reg/registration/Il-A'

#this converts apply barseq registration transformation to new data
#python Main_2d_3d.py --remotesubjectpath='/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/' --localsubjectpath='/grid/zador/home/vinehout/code/2d_3D_linear_reg/' --X_dir='right' --Y_dir='top' --image_type='Il-A' --seq_dir='top' --opticalZ_dir='top' --input_overlap=0.15 --server='zadorstorage2.cshl.edu' --user='imageguest' --password='zadorlab' --FFT_max_gaussian=10 --error_overlap=0.10 --blank_thres=1.5 --checkerboard_size=6 --seg_interations=35 --seg_smooth=3 --POS_folder_X -6 -3 --POS_folder_Y -2 -1 --POS_folder_Z 0 -6 --apply_transform=True --saved_transforms_path='/Users/kaleb/Documents/CSHL/ML_basecalling/code/2d_3D_linear_reg/registration/Il-A'

#this registers 3D barseq from above to 3D 2 photon imaging
#python Main_3d_3d.py --remotesubjectpath='/home/imagestorage/ZadorConfocal1/xiaoyin/20201205JB050tomatolectinlabeling647/lectin_1/' --localsubjectpath='/grid/zador/home/vinehout/code/2d_3D_linear_reg/' --image_type='Il-A' --seq_dir='top' --input_overlap=0.15 --server='zadorstorage2.cshl.edu' --user='imageguest' --password='zadorlab' --FFT_max_gaussian=10 --error_overlap=0.10 --blank_thres=1.5
