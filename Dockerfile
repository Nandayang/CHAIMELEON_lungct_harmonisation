FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime 

WORKDIR /home/chaimeleon/ICL_HAR_CT

COPY model ./model
COPY testcase ./testsample
COPY utils ./utils
COPY predict.sh ./predict.sh
COPY sr_model.py ./sr_model.py
COPY weight.pth ./weight.pth
COPY requirements.txt ./requirements.txt
COPY unet_r231-d5d2fc3d.pth ./unet_r231-d5d2fc3d.pth

RUN apt-get update && apt-get install -y git 
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/JoHof/lungmask


RUN groupadd -g 1000 chaimeleon && \
    useradd --create-home --shell /bin/bash --uid 1000 --gid 1000 chaimeleon 

RUN echo "chaimeleon:chaimeleon" | chpasswd

USER root
RUN mkdir -p /home/chaimeleon/datasets && \
    mkdir -p /home/chaimeleon/persistent-home && \
    mkdir -p /home/chaimeleon/persistent-shared-folder



WORKDIR /home/chaimeleon/ICL_HAR_CT

