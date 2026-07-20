#!/bin/bash
#SBATCH -p compute
#SBATCH -J r2p_convert
#SBATCH -c 1                       # cores per task (see "fatter tasks" note below)
#SBATCH --mem-per-cpu=6G
#SBATCH -t 04:00:00
#SBATCH -o /home/obeesley/pioneerML/model_weights/convert_%A_%a.log
#SBATCH -e /home/obeesley/pioneerML/model_weights/convert_%A_%a.err
#
# Throttled, file-sharded ROOT->parquet conversion.
# ONE array task converts ONE ROOT file -> one parquet shard.
#
#   *** The %K in --array is your cluster-footprint knob: at most K tasks run    ***
#   *** at once no matter how many nodes are idle. Concurrent cores ~= K * (-c). ***
#
# SUBMIT (from login1). %10 = never use more than 10 tasks (=10 cores) at once:
#   ROOT_DIR=/data/raid3/eliza7/PIONEER/data/ML_TEST/pie/train
#   OUT_DIR=/data/raid3/eliza7/PIONEER/data/ML_TEST/unmixed_shards/pie/train
#   N=$(ls "$ROOT_DIR"/*.root | wc -l)
#   sbatch --array=0-$((N-1))%10 convert_array.sbatch "$ROOT_DIR" "$OUT_DIR"
#   # 3rd arg "--keep_pidif" for the pidif channel only.
#
# FATTER TASKS (fewer tasks, each multi-core): give each task a GROUP of files and
# more cores so root_to_parquet's --nprocs shards them in parallel. Set -c C and
# FILES_PER_TASK=C; footprint is then K*C cores. Example (8 tasks max, 4 cores each):
#   FPT=4; N=$(ls "$ROOT_DIR"/*.root | wc -l); T=$(( (N + FPT - 1) / FPT ))
#   sbatch --array=0-$((T-1))%8 -c $FPT --export=ALL,FILES_PER_TASK=$FPT \
#          convert_array.sbatch "$ROOT_DIR" "$OUT_DIR"
#
set -euo pipefail

ROOT_DIR="$1"; OUT_DIR="$2"; EXTRA="${3:-}"
FILES_PER_TASK="${FILES_PER_TASK:-1}"
CORES="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$OUT_DIR"

# This task's slice of the sorted ROOT-file list.
mapfile -t ALL < <(ls "$ROOT_DIR"/*.root | sort)
START=$(( SLURM_ARRAY_TASK_ID * FILES_PER_TASK ))
GROUP=( "${ALL[@]:START:FILES_PER_TASK}" )
if [ "${#GROUP[@]}" -eq 0 ]; then
    echo "[task $SLURM_ARRAY_TASK_ID] no files in slice; nothing to do."; exit 0
fi
OUT="$OUT_DIR/part_$(printf '%04d' "$SLURM_ARRAY_TASK_ID").parquet"
echo "[task $SLURM_ARRAY_TASK_ID] host=$(hostname) cores=$CORES files=${#GROUP[@]} -> $OUT"

source /home/obeesley/main/setenv.sh
source /home/obeesley/miniconda3/etc/profile.d/conda.sh
conda activate research

# Stage this task's ROOT files to node-local /tmp so many concurrent tasks don't all
# stream 2.4 GB files off the 1 GbE NFS at once (the conversion is otherwise IO-bound).
# If node /tmp is too small for a big group, set STAGE_LOCAL=0 to read from NFS instead.
STAGE_LOCAL="${STAGE_LOCAL:-1}"
if [ "$STAGE_LOCAL" = "1" ]; then
    STAGE="/tmp/r2p_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
    mkdir -p "$STAGE"
    trap 'rm -rf "$STAGE"' EXIT
    for f in "${GROUP[@]}"; do cp "$f" "$STAGE/"; done
    INPUT="$STAGE"
else
    # symlink the group into a dir so --input (a directory) sees exactly this slice
    STAGE="/tmp/r2p_links_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
    mkdir -p "$STAGE"
    trap 'rm -rf "$STAGE"' EXIT
    for f in "${GROUP[@]}"; do ln -sf "$f" "$STAGE/"; done
    INPUT="$STAGE"
fi

python /home/obeesley/pioneerML/unified_reco/root_to_parquet.py \
    --input "$INPUT" \
    --output "$OUT" \
    --nprocs "$CORES" \
    --seed $(( 1000 + SLURM_ARRAY_TASK_ID )) \
    --event_id_offset $(( SLURM_ARRAY_TASK_ID * 1000000000 )) \
    ${EXTRA}

echo "[task $SLURM_ARRAY_TASK_ID] done -> $OUT"
