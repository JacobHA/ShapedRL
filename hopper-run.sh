#!/bin/bash

# Define the eta values
ETAS=(0.0 0.5 1.0 2.0 3.0 5.0 10.0)

# Loop over eta values in parallel
for ETA in "${ETAS[@]}"; do
  # loop over 10 parallel runs
  for i in $(seq 1 20);
  do
    python run.py --count=1 --algo=td3 --env=Pendulum-v1 --do_shape --eta=$ETA &
  done
  wait
done