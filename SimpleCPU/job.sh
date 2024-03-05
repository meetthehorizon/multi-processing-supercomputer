#!/bin/bash
#SBATCH --job-name=cpu
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --partition="cpu"

module load python3.8
source /home/kshitij.cse22.itbhu/venv/bin/activate

echo "Starting the job"
python3 -u script.py
echo "Job finished"
