# This is used to run 2d-to-3d code

# pakages to get: pandas, paramiko, scikit-image, opencv-python, open3d, numpy, scikit-learn

FROM python:3.7-slim-buster

WORKDIR /Users/kaleb/Documents/CSHL/2d_3D_linear_reg

COPY ./ ./

RUN pip install pandas==1.2.1

RUN pip install paramiko==2.7.2

RUN pip install scikit-image==0.18.1

#get dependancies for CV2
RUN apt-get update && apt-get install -y python3-opencv

RUN pip install opencv-python==4.5.1.48

RUN pip install numpy==1.19.5

RUN pip install scikit-learn==0.24.1

RUN pip install Pillow==8.1.0

RUN pip install brainrender==2.0.2.9

RUN pip install imio==0.0.5

RUN pip install matplotlib==3.3.3

RUN pip install antspyx==0.2.4

RUN pip install pynumparser==1.4.1

RUN pip install blockmatching==1.2.1

RUN pip install image-similarity-measures==0.3.5

RUN pip install imreg_dft==2.0.0

ENTRYPOINT ["/bin/bash", "./Main_file.sh"]

#EXTRA
#RUN pip install ICP==2.1.1
#RUN pip install cvxpy==1.1.11
#install simple elastic pip install pyelastix==1.2
#RUN pip install open3d==0.12.0