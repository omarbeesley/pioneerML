#!/bin/bash
#SBATCH -p compute
#SBATCH -c 16
#SBATCH --mem=28G
#SBATCH -t 03:00:00
#SBATCH -J purity_mix_tail
#SBATCH --array=0-50
#SBATCH -o purity_mix_tail_%A_%a.log
# Mix the tail-model data with the tail10 recipe. Tasks 0-44 are training shards,
# 45-46 validation, 47-50 the pure-class pie/michel/mudif/pidif evaluation pools.
# Generation-task groups and seeds (7000+task) follow the historical launcher.
# Submit from the release root:
#
#   sbatch --export=ALL,POOLS=/data/parquet,OUT=$PWD/outputs/tail10 scripts/slurm/mix_train_tail.sbatch
#
# POOLS holds <channel>/<channel>_<job>_t<group>_c*.parquet for michel, pie, mudif
# and pidif. The mixer is the release revision, so the result follows the recipe but
# is not the archived tail10 realization.
set -eo pipefail
RELEASE=${RELEASE:-$SLURM_SUBMIT_DIR}
SIF=${SIF:-/data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif}
: "${POOLS:?set POOLS}" "${OUT:?set OUT}"
[ -d "$RELEASE/src/purity" ] || { echo "RELEASE=$RELEASE is not the release root"; exit 2; }
[ -d /scratch ] || { echo "no /scratch"; exit 4; }
i=${SLURM_ARRAY_TASK_ID}
SCR=/scratch/$USER/mixtail_${SLURM_ARRAY_JOB_ID}_$i; trap 'rm -rf "$SCR"' EXIT
rm -rf "$SCR"; mkdir -p "$SCR" "$OUT"
if   [ $i -le 44 ]; then MODE=mixed; NAME=train_$(printf '%03d' $i); MG=$i; PG=$((i%23)); UG=$((i%36)); DG=$((i%16)); NEV=200000
elif [ $i -le 46 ]; then MODE=mixed; NAME=val_$((i-45)); MG=$i; PG=23; UG=36; DG=16; NEV=200000
else
  case $i in
    47) MODE=pie;    DG=17;; 48) MODE=michel; DG=17;; 49) MODE=mudif;  DG=18;; 50) MODE=pidif; DG=19;;
  esac
  NAME=eval_$MODE; MG=48; PG=24; UG=38; NEV=150000
  [ $i -eq 49 ] && UG=39
fi

py() {
  apptainer exec --cleanenv --contain --pwd /tmp \
    --bind "$RELEASE":/release --bind "$POOLS":/pools --bind "$SCR":/work --bind "$OUT":/out \
    --env PYTHONPATH=/release/src "$SIF" python3 -m "$@"
}
py purity.preprocessing.mix_tail_shard --task $i --mode $MODE --mgrp $MG --pgrp $PG --ugrp $UG --dgrp $DG \
   --nev $NEV --seed $((7000+i)) --pool_dir /pools --lut_dir /release/assets/luts \
   --scratch /work --out /work/$NAME.parquet
py purity.preprocessing.origin_reslice --input /work/$NAME.parquet --output /out/${NAME}_osl.parquet
echo "[mix_tail $i] done -> $OUT/${NAME}_osl.parquet"
