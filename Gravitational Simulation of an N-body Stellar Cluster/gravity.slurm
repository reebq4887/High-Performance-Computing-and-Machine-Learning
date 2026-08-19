#!/bin/bash
# Avon submission script for PX457 2025 Assignment 5
# You WILL need to modify this as the Assignment progresses!

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3700mb
#SBATCH --time=00:02:00
#SBATCH --partition PX457

# Change into the directory where I submitted the job
cd $SLURM_SUBMIT_DIR

# Modules as needed
module purge
module load GCC OpenMPI

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun ./gravity -N 750 -S 20000

exit 0