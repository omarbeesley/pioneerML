#!/bin/bash
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o /home/obeesley/pioneerML/unified_reco/attn_slice_probe_%j.log
#SBATCH -e /home/obeesley/pioneerML/unified_reco/attn_slice_probe_%j.err
#SBATCH -J attn_slice

# Slice-level attention probe + context ablation on a real pileup event.
# Never run this on the login node: it opens a parquet and a checkpoint.
#
# Binds follow train_tail.sbatch, plus /data/raid3 because the eval parquets and
# the container itself live there. Inside the container the repo is /pioneerML.
#
# Defaults: the ra5gr main model pinned at epoch19 (pin epochN.pth, never
# _best.pth -- the trainer can overwrite _best mid-eval), on the origin-resliced
# michel eval sample. Reslicing matters here twice over: the model trains on
# resliced input, and the probe picks slices by per-slice PDG, which is only
# clean once origin_reslice has separated the two chains' hits.
#
# ra5gr was trained with the best_xy chi2 gate, so inference must set
# PURITY_GATE_MODE=best_xy or models_v2 falls back to the 3d gate.

CKPT=${CKPT:-/raid3/PIONEER/data/ML_TEST/scratch_pipeline/model_weights/main_hybridv2_ra5gr_epoch19.pth}
DATA=${DATA:-/raid3/PIONEER/data/ML_TEST/scratch_pipeline/tail10/eval_michel_osl.parquet}
OUT=${OUT:-/pioneerML/figures/talk/attn_slice_probe.npz}

apptainer exec --cleanenv --contain --fakeroot \
  --bind /tmp:/tmp \
  --bind $HOME/.cache:$HOME/.cache \
  --bind /home/obeesley/pioneerML:/pioneerML \
  --bind /data/raid3/eliza7:/raid3 \
  --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  --env PURITY_GATE_MODE=${PURITY_GATE_MODE:-best_xy} \
  /data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif \
  python3 /pioneerML/unified_reco/attn_slice_probe.py \
    --checkpoint "$CKPT" \
    --dataset "$DATA" \
    --out "$OUT" \
    --scan_events 8000 \
    --max_candidates 8 \
    $EXTRA \
    --mc_passes 32 \
    --layer 1
