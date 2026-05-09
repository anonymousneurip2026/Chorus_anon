#!/bin/bash
# run_rcc_sweep.sh
# ------------------
# Unified script to run TCGA-RCC benchmarking sweep.
# Usage: bash run_rcc_sweep.sh <GPU_ID> <SHOTS>

GPU_ID=${1:-0}
SHOTS=${2:-16}
VARIANTS=("none" "v1_embed" "v2_attn" "v3_local" "v4_hybrid" "v5_precision" "v6_threshold" "v9_inverse" "v10_temp")

TASK='RCC'
MAG='20x_256px_0px_overlap'
FOLDS=5
LOG_DIR='logs/'
FEATURE_ROOT="path/to/TCGA-RCC-Unified/${MAG}"

mkdir -p $LOG_DIR

echo "Starting ${SHOTS}-shot sweep for TCGA-RCC on GPU $GPU_ID..."

for var in "${VARIANTS[@]}"; do
    EXP="CHORUS/${TASK}_${var}_${SHOTS}shots"
    CCA_PATH="cca_artifacts/visual_cca_${TASK}_${SHOTS}shots_${MAG}.npz"
    
    if [ ! -f "$CCA_PATH" ]; then
        echo "ERROR: CCA artifact missing: $CCA_PATH"
        continue
    fi

    echo "=========================================================="
    echo "STARTING VARIANT: $var | SHOTS: $SHOTS"
    echo "=========================================================="
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    python -u main.py \
        --seed 1 \
        --drop_out \
        --early_stopping \
        --lr 1e-4 \
        --k $FOLDS \
        --task "task_tcga_rcc_subtyping" \
        --results_dir "results/${EXP}/" \
        --exp_code "${TASK}_${SHOTS}shots_${var}" \
        --model_type CHORUS \
        --mode transformer \
        --log_data \
        --split_dir "TCGA_RCC_${SHOTS}shots_${FOLDS}folds" \
        --text_prompt_path "text_prompt/TCGA_RCC_two_scale_text_prompt.csv" \
        --prototype_number 16 \
        --window_size 8 \
        --sim_threshold 0.8 \
        --cca_visual_path "$CCA_PATH" \
        --chorus_variant ${var} \
        --encoder_dir uni_v1=${FEATURE_ROOT}/uni_v1 \
        --encoder_dir gigapath=${FEATURE_ROOT}/gigapath \
        --encoder_dir resnet50=${FEATURE_ROOT}/resnet50 \
        --encoder_order uni_v1 gigapath resnet50 \
        2>&1 | tee "${LOG_DIR}${TASK}_${SHOTS}shots_${var}.log"
done

echo "RCC ${SHOTS}-shot sweep completed."
