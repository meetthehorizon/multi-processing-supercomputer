#!/bin/bash
#SBATCH --job-name=DataParallel
#SBATCH --output=DataParallel.out
#SBATCH --error=DataParallel.err
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

module load python3.8
source /home/kshitij.cse22.itbhu/venv/bin/activate

echo "Starting the job"
python3 script.py
echo "Job finished"
