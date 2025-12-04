#!/bin/bash

#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=100g
#SBATCH -p l40-gpu
#SBATCH -t 07-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/longleaf/%j.out

source ~/.bashrc
conda activate llava

nvidia-smi


