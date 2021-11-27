#!/usr/bin/env bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_name> <parent_experiment_name>"
    exit 1
fi

if [ -z $OMEGA_EXPERIMENTS_DIR ]; then
  echo "OMEGA_EXPERIMENTS_DIR environment variable is not set"
  exit 1
fi

EXPERIMENTS_DIR=$OMEGA_EXPERIMENTS_DIR
NEW_EXP_DIR=$EXPERIMENTS_DIR/$1
OLD_EXP_DIR=$EXPERIMENTS_DIR/$2

if [[ ! -d $EXPERIMENTS_DIR ]]; then
  echo "Experiments dir $EXPERIMENTS_DIR does not exist"
  exit 1
fi

if [[ -d $NEW_EXP_DIR ]]; then
  echo "Experiment dir $NEW_EXP_DIR already exists"
  exit 1
fi

if [[ ! -d $OLD_EXP_DIR ]]; then
  echo "Experiment dir $OLD_EXP_DIR does not exist"
  exit 1
fi

mkdir -p $NEW_EXP_DIR
cp $OLD_EXP_DIR/config.yaml $NEW_EXP_DIR/

vim $NEW_EXP_DIR/config.yaml
