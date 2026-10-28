#!/bin/bash
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -o /home/obeesley/pioneerML/unified_reco/bhabha_scan_%j.log
#SBATCH -e /home/obeesley/pioneerML/unified_reco/bhabha_scan_%j.err
#SBATCH -J bhabha_scan

# Scan a mixed-events parquet for Bhabha-scatter candidates (see bhabha_scan.py).
# Compute node only -- loads the full parquet.

DATA=${DATA:-/raid3/PIONEER/data/ML_TEST/scratch_pipeline/tail10/eval_michel_osl.parquet}
OUT=${OUT:-/pioneerML/figures/talk/survey/bhabha}

apptainer exec --cleanenv --contain --fakeroot \
  --bind /tmp:/tmp \
  --bind $HOME/.cache:$HOME/.cache \
  --bind /home/obeesley/pioneerML:/pioneerML \
  --bind /data/raid3/eliza7:/raid3 \
  /data/raid3/eliza7/PIONEER/data/ML_TEST/pytorch.sif \
  python3 /pioneerML/unified_reco/bhabha_scan.py \
    --dataset "$DATA" \
    --out "$OUT" \
    --scan_events 150000 \
    --min_e_hits 3 \
    --min_bias 8 \
    --max_candidates 20 \
    $EXTRA
