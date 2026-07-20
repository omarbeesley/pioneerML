#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -o /home/obeesley/pioneerML/model_weights/train_tail_%j.log
#SBATCH -e /home/obeesley/pioneerML/model_weights/train_tail_%j.err
#SBATCH -J purity_tail

# OPTIONAL batch launcher. For interactive use just run `apshell` on the gpu
# node, then:  python3 /pioneerML/unified_reco/train_tail.py --epochs 15
#
# Binds mirror the `apshell` alias exactly:
#   /home/obeesley/pioneerML  -> /pioneerML
#   /data/nvme0/prod_ml_data  -> /data    (NB: not an identity bind)
# pytorch.sif has torch/torch_geometric/torch_cluster/pyarrow. Data generation
# needs PyROOT, which lives in the host conda `research` env, not the sif.

apptainer exec --nv --cleanenv --contain --fakeroot \
  --bind /tmp:/tmp \
  --bind $HOME/.cache:$HOME/.cache \
  --bind /home/obeesley/pioneerML:/pioneerML \
  --bind /data/nvme0/prod_ml_data:/data \
  --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  /data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif \
  python3 /pioneerML/unified_reco/train_tail.py \
    --train_path /data/tail_reveal_training/data.parquet \
    --val_path   /data/tail_reveal_validation/data.parquet \
    --weights_dir /pioneerML/model_weights \
    --run_name PURITY_TAIL \
    --epochs 15
