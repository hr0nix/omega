#!/usr/bin/env bash

docker run --network host --gpus all -it --user $(id -u):$(id -g) --env WANDB_CONFIG_DIR=/omega/.wandb/config --env PYTHONPATH=/omega --env HOME=/omega --mount type=bind,source=$(realpath $(dirname $0))/../,target=/omega hr0nix/omega_env:1.13 /bin/bash
