#!/bin/bash
# scripts/fit_visual_cca.sh
# -------------------------
# Phase 1A: fit visual GCCA across uni_v1 + gigapath + resnet50
# Run once per task; output saved to cca_artifacts/visual_cca_<task>.npz

TASK='RCC'
MAG='20x_256px_0px_overlap'
UNIFIED='path/to/TCGA-RCC-Unified'
FEATURE_ROOT="${UNIFIED}/${MAG}"
SPLIT_CSV="splits/TCGA_RCC_16shots_5folds/splits_0.csv"
ENCODERS="uni_v1 gigapath resnet50"   # MUST MATCH --encoder_order in training script

mkdir -p cca_artifacts logs

python -u -m cca.fit_visual_cca \
    --feature_root "$FEATURE_ROOT" \
    --encoders $ENCODERS \
    --train_csv "$SPLIT_CSV" \
    --n_calibration 150000 \
    --n_holdout 20000 \
    --latent_dim 512 \
    --reg 1e-3 \
    --output "cca_artifacts/visual_cca_${TASK}_${MAG}.npz" \
    --seed 0 2>&1 | tee logs/fit_visual_cca_${TASK}_${MAG}.log
