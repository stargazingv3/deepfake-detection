# BASE CONTAINER
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#imgcorruptions
ENV DEBIAN_FRONTEND=noninteractive

# APT INSTALLS
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev git \
    #imgcorruptions
    libgl1 libglib2.0-0 
#RUN apt-get update && apt-get install libgl1  -y
RUN apt -y install python3-pip
#RUN apt-get install -y libglib2.0-0 

# PIP
RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt /tmp
RUN python3 -m pip install -r /tmp/requirements.txt

RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /.local && chmod 777 /.local

# APP DIRECTORY PATH
WORKDIR /app
