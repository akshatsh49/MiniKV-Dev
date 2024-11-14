#!/bin/bash
#SBATCH --job-name=MiniKV
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcjw-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=18:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --mem=128G # running into OOM error
#SBATCH --cpus-per-task=16

# module load python/3.11.6
source ~/.bashrc
module load cuda/12.4.0

# the line below moves model ckpts to the /scratch/ partition, with lot of space but large loading time
# change this to some ~/ location of the model you are actively working with
export HF_DATASETS_CACHE="/u/ding3/.cache"
export HF_HOME=$HF_DATASETS_CACHE
export HF_HUB_CACHE=$HF_DATASETS_CACHE
export HF_ASSETS_CACHE=$HF_DATASETS_CACHE
export TRANSFORMERS_CACHE=$HF_DATASETS_CACHE

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; print(torch.__path__[0])")/lib