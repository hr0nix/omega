#!/usr/bin/env bash

docker run --gpus all -it --env PYTHONPATH=/home/bob/omega --mount type=bind,source=$(realpath $(dirname $0))/../,target=/home/bob/omega hr0nix/omega_env:1.6 /bin/bash
