#!/bin/bash
#SBATCH --job-name=DataParallel
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --nodes=2
#SBATCH --ntasks=40
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

source /home/kshitij.cse22.itbhu/miniconda3/bin/activate
conda activate pytorch

/home/apps/spack/share/spack/setup-env.sh
spack load cuda
spack load cuda@12.3.0

echo "Starting at $(date)"
python test.py
echo "Finished at $(date)"
