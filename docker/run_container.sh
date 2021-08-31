#!/usr/bin/env bash

docker run --gpus all -it --env PYTHONPATH=/omega --mount type=bind,source=$(pwd)/../,target=/omega hr0nix/omega_env:1.2 /bin/bash
