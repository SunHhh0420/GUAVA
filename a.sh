#!/usr/bin/env bash
set -e
cd /home/sh/GUAVA
# activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh && conda activate GUAVA
export PYTHONPATH=/home/sh/GUAVA
python main/test.py \
  -d '0' \
  -m assets/GUAVA \
  -s outputs/example \
  --data_path assets/example/tracked_video/6gvP8f5WQyo__056