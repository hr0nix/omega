#!/usr/bin/env bash

docker run --gpus all -it --env PYTHONPATH=/omega --env OMEGA_EXPERIMENTS_DIR=/omega/experiments --mount type=bind,source=$(realpath $(dirname $0))/../,target=/omega hr0nix/omega_env:1.5 /bin/bash
