#!/bin/bash
#SBATCH -p 3090-gcondo --gres=gpu:1
# SBATCH -p gpu-he --gres=gpu:1 --constraint=a6000
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 11:00:00

export CUR_DIR=/users/mzuo6/tdmpc2-jax

module load cuda/12.1.1

cd $CUR_DIR
source venv/bin/activate

python train.py $@
