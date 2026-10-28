#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J purity_train_tail
#SBATCH -o purity_train_tail_%j.log
# Train the tail-reveal model with the PURITY_TAIL10 recipe. Submit from the release root:
#
#   sbatch --export=ALL,DATA=/data/tail10,OUT=$PWD/outputs/weights scripts/slurm/train_tail.sbatch
#
# DATA holds train_000_osl.parquet ... train_044_osl.parquet and val_0_osl.parquet.
# HISTORICAL=1 reproduces the TAIL10 DataLoader behaviour (first five shards every
# epoch); the default rotates through all shards. SMOKE=1 runs a 2000-event epoch.
set -eo pipefail
RELEASE=${RELEASE:-$SLURM_SUBMIT_DIR}
SIF=${SIF:-/data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif}
: "${DATA:?set DATA}" "${OUT:?set OUT}"
[ -d "$RELEASE/src/purity" ] || { echo "RELEASE=$RELEASE is not the release root"; exit 2; }
mkdir -p "$OUT"
NAME=${NAME:-PURITY_TAIL}
EXTRA=()
[ "${HISTORICAL:-0}" = 1 ] && EXTRA+=(--historical_tail10_loader)

run() {
  apptainer exec --nv --cleanenv --contain --pwd /tmp \
    --bind "$RELEASE":/release --bind "$DATA":/data --bind "$OUT":/out \
    --env PYTHONPATH=/release/src \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$SIF" python3 -m purity.training.tail \
      --val_path /data/val_0_osl.parquet --weights_dir /out "$@"
}

echo "[train_tail] host=$(hostname) release=$RELEASE historical=${HISTORICAL:-0}"
if [ "${SMOKE:-0}" = 1 ]; then
  run --train_path /data/train_000_osl.parquet --run_name smoke_tail \
      --epochs 1 --epoch_size 2000 --val_max_batches 5 --overwrite
else
  run --train_path '/data/train_{:03d}_osl.parquet' --run_name "$NAME" --epochs 10 "${EXTRA[@]}"
fi
echo "[train_tail] done"
