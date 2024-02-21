#!/bin/bash
#SHAPE=${2:-"True"}
#NOTERM=${3:-"False"}
echo "starting $EXPN tasks"
for (( i=1;i<=$1;i++ )); do {
  echo $i;
  python maze_run.py &
} done