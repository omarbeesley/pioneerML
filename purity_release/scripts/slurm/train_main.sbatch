#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 47:00:00
#SBATCH -J purity_train_main
#SBATCH -o purity_train_main_%j.log
# Train the reconstruction model with the main_hybridv2_filmg2 recipe.
# Submit from the release root. Phase 1 = filmg (epochs 1-4, angle-decorrelation
# weight 0.5); phase 2 = filmg2 (resume phase-1 epoch 4, epochs 5-20, weight 0.05,
# pinned LRs). Run phase 2 after phase 1:
#
#   J=$(sbatch --parsable --export=ALL,PHASE=1,DATA=/data/train10M_fixed,OUT=$PWD/outputs/weights scripts/slurm/train_main.sbatch)
#   sbatch --dependency=afterok:$J --export=ALL,PHASE=2,DATA=/data/train10M_fixed,OUT=$PWD/outputs/weights scripts/slurm/train_main.sbatch
#
# DATA holds train.parquet and val.parquet; OUT receives checkpoints and loss history.
# SMOKE=1 runs one 2000-event epoch with five validation batches instead.
set -eo pipefail
RELEASE=${RELEASE:-$SLURM_SUBMIT_DIR}
SIF=${SIF:-/data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif}
: "${PHASE:?set PHASE=1 or 2}" "${DATA:?set DATA}" "${OUT:?set OUT}"
[ -d "$RELEASE/src/purity" ] || { echo "RELEASE=$RELEASE is not the release root"; exit 2; }
mkdir -p "$OUT"
NAME1=${NAME1:-main_hybridv2_filmg}
NAME2=${NAME2:-main_hybridv2_filmg2}

run() {
  apptainer exec --nv --cleanenv --contain --pwd /tmp \
    --bind "$RELEASE":/release --bind "$DATA":/data --bind "$OUT":/out \
    --env PYTHONPATH=/release/src \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$SIF" python3 -m purity.training.main \
      --train_path /data/train.parquet --val_path /data/val.parquet --weights_dir /out \
      --dropout 0.0 --weight_decay 1e-3 --epoch_mode chunked --chunk_seed 12345 "$@"
}

echo "[train_main] host=$(hostname) phase=$PHASE release=$RELEASE"
if [ "${SMOKE:-0}" = 1 ]; then
  run --run_name "smoke_phase$PHASE" --epochs 1 --epoch_size 2000 --stage2_epoch 1 \
      --val_max_batches 5 --epoch_mode random
elif [ "$PHASE" = 1 ]; then
  run --run_name "$NAME1" --epochs 4 --w_angle_decorr 0.5 --val_max_batches 1000
elif [ "$PHASE" = 2 ]; then
  run --run_name "$NAME2" --resume "/out/${NAME1}_epoch4.pth" \
      --lr_base 2.5e-4 --lr_event 2.5e-4 --lr_bias 5e-4 \
      --epochs 20 --val_max_batches 1000
else
  echo "PHASE must be 1 or 2"; exit 2
fi
echo "[train_main] phase $PHASE done"
