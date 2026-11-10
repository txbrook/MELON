#!/usr/bin/env bash
# Reproduces the Mouse E15.5 spatial multi-omics analysis reported in the MELON
# manuscript. Self-contained: run from inside the supplementary bundle.
#
# Usage:
#     bash run_mouse_e15.sh /path/to/Mouse_E15.5
#
# The data directory must contain:
#     mRNA_data.h5, ATAC_data.h5, coords.csv
#
# Outputs are written to ./results/example_mouse_e15/.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:-}"
if [[ -z "$DATA_DIR" || ! -d "$DATA_DIR" ]]; then
    echo "Usage: bash run_mouse_e15.sh /path/to/Mouse_E15.5" >&2
    exit 1
fi

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
cd "$HERE"

OUT_DIR="results/example_mouse_e15"
SEED=42
EPOCHS=100
TARGET_K=12

mkdir -p "$OUT_DIR"

# Step 1 — train MELON on RNA + ATAC with TF-IDF/LSI ATAC preprocessing.
python run_melon.py \
    --mod1_h5      "$DATA_DIR/mRNA_data.h5" \
    --mod2_h5      "$DATA_DIR/ATAC_data.h5" \
    --coords_csv   "$DATA_DIR/coords.csv" \
    --atac_preprocess tfidf_lsi \
    --output_dir   "$OUT_DIR" \
    --epochs       "$EPOCHS" \
    --seed         "$SEED"

# Step 2 — cluster the learned embedding.
python cluster_melon.py \
    --melon_dir      "$OUT_DIR" \
    --target_k       "$TARGET_K" \
    --methods        spectral,gmm,kmeans,walktrap,louvain,leiden \
    --primary_method leiden
