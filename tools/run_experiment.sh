#!/usr/bin/env bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

if [ -z $OMEGA_EXPERIMENTS_DIR ]; then
  echo "OMEGA_EXPERIMENTS_DIR environment variable is not set"
  exit 1
fi

EXPERIMENTS_DIR=$OMEGA_EXPERIMENTS_DIR
EXP_DIR=$EXPERIMENTS_DIR/$1

if [[ ! -d $EXPERIMENTS_DIR ]]; then
  echo "Experiments dir $EXPERIMENTS_DIR does not exist"
  exit 1
fi

if [[ ! -d $EXP_DIR ]]; then
  echo "Experiment dir $EXP_DIR does not exist"
  exit 1
fi

python3.8 ./tools/train_agent.py --config $EXP_DIR/config.yaml --checkpoints $EXP_DIR/checkpoints --game-logs $EXP_DIR/games --tb-logs $EXP_DIR/logs