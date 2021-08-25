FROM nvidia/cuda:11.1-base

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    python3.8 python3-pip python3-dev bison flex git make g++ cmake libbz2-dev

WORKDIR /omega
COPY ./requirements.txt /omega

RUN pip install -r /omega/requirements.txt --find-links https://storage.googleapis.com/jax-releases/jax_releases.html

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-11.1/compat/