#!/bin/bash

# Job name
#SBATCH --job-name=more_pstprc
# Output file name
#SBATCH --output=outfiles/%x_%j.out
#SBATCH --error=outfiles/%x_%j.err
# SBATCH --time=1-23:59:59

# Set the required partition [change]
#SBATCH --partition=parallel-long --qos=parallel-long
# Number of processes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

# Memory per process
# SBATCH --mem-per-cpu=5500M
#SBATCH --mem=185G
# To turn hyperthreading off
#SBATCH --hint=nomultithread
#SBATCH --array=13-15

source /mnt/pfs/rajarshi.chattopadhyay/miniconda3/bin/activate && conda activate p311

# export OPENBLAS_NUM_THREADS=32
# export NUMBA_NUM_THREADS=32

date
# which mpirun
# echo "node ip: " `hostname -i`
# # for i in {10..50..4}; do
# # jupyter lab --no-browser --ip=0.0.0.0 --port=8888
# done
# time python -u npy_to_h5py.py
# python -u traj-latest.py $SLURM_ARRAY_TASK_ID
python -u postproc_new_new_new.py $SLURM_ARRAY_TASK_ID

date