#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -o /home/obeesley/pioneerML/model_weights/train_fast3_%j.log
#SBATCH -e /home/obeesley/pioneerML/model_weights/train_fast3_%j.err
#SBATCH -J purity_fast3

apptainer exec --nv --cleanenv --contain --fakeroot \
  --bind /tmp:/tmp \
  --bind $HOME/.cache:$HOME/.cache \
  --bind /home/obeesley/pioneerML:/pioneerML \
  --bind /data/nvme0/test_ml_data:/data \
  --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  pytorch.sif \
  python3 /pioneerML/unified_reco/train_fast3.py
