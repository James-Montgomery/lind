FROM ubuntu:bionic

################################################################################
USER root
ENV HOME /root
WORKDIR $HOME

################################################################################

SHELL [ "/bin/bash", "-c" ]

################################################################################

# Instructions to install python from source
# https://github.com/matthewfeickert/Docker-Python3-Ubuntu

RUN apt-get update && apt-get install -y \
        software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip
RUN python3.7 -m pip install pip
RUN apt-get update && apt-get install -y \
    python3-distutils \
    python3-setuptools

################################################################################

RUN echo $HOME
COPY . $HOME
RUN pip3 install -e ".[tests, r_backends, static_designs]"
