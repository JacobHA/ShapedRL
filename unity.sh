#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=48GB  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o outfiles/%j.out  # %j = job ID

#SBATCH --array=1-30

# module load cuda/10.1.243
# /modules/apps/cuda/10.1.243/samples/bin/x86_64/linux/release/deviceQuery
# Load the conda environment:
module load miniconda/22.11.1-1
# eval "$(conda shell.bash hook)"
conda activate /home/jacob_adamczyk001_umb_edu/.conda/envs/rlenv
export CPATH=$CPATH:$CONDA_PREFIX/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Set the Weights and Biases environment variables
export WANDB_MODE=offline
wandb offline

# Start the evaluations
EXPERIMENT=${1:-"atari-full"}
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
wait