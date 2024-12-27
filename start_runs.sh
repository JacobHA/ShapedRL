#!/bin/bash

ENVNAME=${1:-"Humanoid-v4"}
# 30 humanoids will require 10GB GPU and 200GB RAM for 1M buffer
# 30 pendulums will require 10GB GPU and ~20GB RAM for 200k buffer
ALGO=${2:-"td3"}
ETA=${3:-"0.5"}
N_PARALLEL=${4:-"15"}
# baseline
if [ $ETA == "0.0" ];
then
  for i in $(seq 1 $N_PARALLEL);
  do
    python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
    # wait to make sure the runs don't try to write to the same log directory
    sleep 0.5
  done
# shaped
else
  for i in $(seq 1 $N_PARALLEL);
  do
    python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
    sleep 0.5
  done
fi
wait