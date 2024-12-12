#!/bin/bash

ENVNAME=${1:-"Humanoid-v4"}
# 30 humanoids will require 10GB GPU and 200GB RAM for 1M buffer
# 30 pendulums will require 10GB GPU and ~20GB RAM for 200k buffer
ALGO=${2:-"td3"}
ETA=${3:-"0.5"}
# baseline
if [ $ETA == "0.0" ];
then
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0 &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --eta=0.0
# shaped
else
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA &
  python run.py --count=1 --algo=$ALGO --env=$ENVNAME --do_shape --eta=$ETA
fi
wait