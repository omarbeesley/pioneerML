#!/bin/bash
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 04:00:00
#SBATCH -J purity_concat_main
#SBATCH -o purity_concat_main_%j.log
# Join the mix_train_main.sbatch shards into train.parquet and val.parquet, dropping
# events with non-finite hit values. Submit from the release root with OUT set to
# the same directory as the mixing job.
set -eo pipefail
RELEASE=${RELEASE:-$SLURM_SUBMIT_DIR}
SIF=${SIF:-/data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif}
: "${OUT:?set OUT}"
[ -d "$RELEASE/src/purity" ] || { echo "RELEASE=$RELEASE is not the release root"; exit 2; }
py() {
  apptainer exec --cleanenv --contain --pwd /tmp \
    --bind "$RELEASE":/release --bind "$OUT":/out \
    --env PYTHONPATH=/release/src "$SIF" python3 -m purity.preprocessing.concat_clean "$@"
}
py --input '/out/mix_*/data.parquet' --output /out/train.parquet
py --input '/out/valmix_*/data.parquet' --output /out/val.parquet
