#!/bin/bash
#SBATCH --job-name=shape-%a
#SBATCH --output=shape-%a.out
#SBATCH --error=shape-%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Load the required modules
module load anaconda/3.9
# activate conda
source /home/$USER/.bashrc
conda activate u-chi-learning

# Set the Weights and Biases environment variables
export WANDB_MODE=offline
wandb offline

# Start the evaluations
EXPERIMENT=${1:-"atari-mini"}
N_PARALLEL_JOBS=${2:-5}
DEVICE=${3:-"cuda"}
# echo "starting $N_PARALLEL_JOBS tasks"
# parallel --jobs N_PARALLEL_JOBS python sweep.py --n_runs=1 --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE
#for i in {1..5}; do {
#  python sweep.py --n_runs=1 --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
#} done

python sweep.py --n_runs=1 --local-wandb=True --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
python sweep.py --n_runs=1 --local-wandb=True --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
python sweep.py --n_runs=1 --local-wandb=True --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
python sweep.py --n_runs=1 --local-wandb=True --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
python sweep.py --n_runs=1 --local-wandb=True --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
python sweep.py --n_runs=1 --local-wandb=True --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
python sweep.py --n_runs=1 --local-wandb=True --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE &
python sweep.py --n_runs=1 --local-wandb=True --proj=shaping --exp-name=$EXPERIMENT -d $DEVICE