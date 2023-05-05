#!/bin/bash
#SBATCH --job-name=mvr
#SBATCH --nodes=1
#SBATCH --output=mvr.out
#SBATCH --error=mvr.err
#SBATCH --cpus-per-task=5
#SBATCH --mem=64GB
#SBATCH --time=4-23:00:00
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate py37
bash mvr_scripts/train_mvr_xnli.sh

