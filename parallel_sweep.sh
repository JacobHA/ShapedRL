#!/bin/bash
EXPERIMENT=${1:-"atari-mini"}
N_PARALLEL_JOBS=${2:-5}
DEVICE=${3:-"cuda"}
for (( i=1;i<=$1;i++ )); do {
  python sweep.py --n_runs=1 --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
} done