#!/usr/bin/env bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <episode_file_name>"
    exit 1
fi

python3.8 /usr/local/lib/python3.8/dist-packages/nle/scripts/ttyplay2.py -s 1 -a -f $1