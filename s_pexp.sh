#!/bin/bash
#SBATCH --job-name=mujoco
#SBATCH --time=03-23:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --cpus-per-task=4

#set an account to use
#if not used then default will be used
# for scavenger users, use this format:
##SBATCH --account=pi_first.last
# for contributing users, use this format:
##SBATCH --account=math

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=outfiles/%x-%A_%a.err
#SBATCH --output=outfiles/%x-%A_%a.out

# set the partition where the job will run.  Multiple partitions can
# be specified as a comma separated list
# Use command "sinfo" to get the list of partitions
##SBATCH --partition=AMD6276
#SBATCH --partition=DGXA100
# https://www.umb.edu/rc/hpc/chimera/chimera_scheduler

#When submitting to the GPU node, these following three lines are needed:
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# Can comment this out (for single node jobs)
##SBATCH --array=1-10

# Put your job commands here, including loading any needed
# modules or diagnostic echos. Needed for GPU partitions:
export USER=jacob.adamczyk001
export HOME=/home/jacob.adamczyk001
source /etc/profile

echo "starting task $SLURM_ARRAY_TASK_ID"
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

N = ${1:-1}
for task in {1..$N}; do {

} done
#!/bin/bash
echo "starting $EXPN tasks"
for (( i=1;i<=$1;i++ )); do {
  echo $i;
  python maze_run.py &
} done

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"
