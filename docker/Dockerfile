FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    build-essential autoconf libtool pkg-config software-properties-common \
    python3.8 python3.8-dev python3-pip bison flex git libbz2-dev wget

# Install cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get --allow-unauthenticated install -y \
    cmake kitware-archive-keyring

WORKDIR /omega
COPY ./requirements.txt /omega

RUN python3.8 -m pip install --upgrade pip setuptools
RUN python3.8 -m pip install -r /omega/requirements.txt --find-links https://storage.googleapis.com/jax-releases/jax_releases.html
