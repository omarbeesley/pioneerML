#!/bin/bash
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem 28G
#SBATCH -t 03:00:00
#SBATCH -J purity_mix_main
#SBATCH --array=0-43
#SBATCH -o purity_mix_main_%A_%a.log
# Mix the reconstruction-model training data with the train10M_fixed recipe:
# 40 training shards (tasks 0-39) and 4 validation shards (40-43) of 250,000 events,
# mixed with TRAIN_VAL_OPTS from the training/validation unmixed splits and
# origin-resliced. Submit from the release root:
#
#   J=$(sbatch --parsable --export=ALL,UNMIXED=/data/unmixed_split,OUT=$PWD/outputs/train10M scripts/slurm/mix_train_main.sbatch)
#   sbatch --dependency=afterok:$J --export=ALL,OUT=$PWD/outputs/train10M scripts/slurm/concat_train_main.sbatch
#
# Seeds follow the historical launcher (9000+task, 9500+task). The mixer is the
# release revision, so the result follows the recipe but is not the archived pool.
set -eo pipefail
RELEASE=${RELEASE:-$SLURM_SUBMIT_DIR}
SIF=${SIF:-/data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif}
: "${UNMIXED:?set UNMIXED}" "${OUT:?set OUT}"
[ -d "$RELEASE/src/purity" ] || { echo "RELEASE=$RELEASE is not the release root"; exit 2; }
[ -d /scratch ] || { echo "no /scratch"; exit 4; }
G=${SLURM_ARRAY_TASK_ID}
SCR=/scratch/$USER/mixmain_${SLURM_ARRAY_JOB_ID}_$G; trap 'rm -rf "$SCR"' EXIT
rm -rf "$SCR"; mkdir -p "$SCR/out" "$OUT"
if [ "$G" -lt 40 ]; then ONLY=training; NARG="--num_train 250000"; SEED=$((9000+G)); SUB=training_5_11; DEST=mix_$(printf '%02d' $G)
else ONLY=validation; NARG="--num_val 250000"; SEED=$((9500+G)); SUB=validation_5_11; DEST=valmix_$(printf '%02d' $G); fi

py() {
  apptainer exec --cleanenv --contain --pwd /tmp \
    --bind "$RELEASE":/release --bind "$UNMIXED":/unmixed --bind "$SCR":/work --bind "$OUT":/out \
    --env PYTHONPATH=/release/src "$SIF" python3 -m "$@"
}
py purity.preprocessing.generate_benchmarks --only $ONLY $NARG --seed $SEED --chunk_size 50000 \
   --data_dir /unmixed --lut_dir /release/assets/luts --out_dir /work/out
mkdir -p "$OUT/$DEST"
py purity.preprocessing.origin_reslice --input "/work/out/$SUB/data.parquet" --output "/out/$DEST/data.parquet"
echo "[mix_main $G] done -> $OUT/$DEST/data.parquet"
