#!/bin/bash

docker build -t kvinehout/docker:latest .

#so run sigulairty on the docker image instead of docker
docker run -v /var/run/docker.sock:/var/run/docker.sock -v /Users/kaleb/Documents/CSHL/ML_basecalling/code/2d_3D_linear_reg:/output --privileged -t --rm singularityware/docker2singularity  kvinehout/docker:latest

#use this line on the cluster
#singularity run -B /grid/zador/home/vinehout/code:/Users/kaleb/Documents/CSHL/ML_basecalling/code/2d_3D_linear_reg kvinehout_docker_latest-2020-11-19-e10a31669022.simg

#docker run kvinehout/docker:latest

