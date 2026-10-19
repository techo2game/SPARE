#!/bin/bash
# Train SPARE on Real3D-AD, then evaluate. For each category: register the
# training samples, build the memory bank, generate synthetic anomalies, train.
# By default, all categories; to choose:
#   CATEGORIES="airplane chicken" scripts/train.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CFG=configs/real3d-ad/base.yaml

[ -d data/Real3D/airplane/train ] || { echo "Real3D-AD not found in data/Real3D (README, step 2)" >&2; exit 1; }
[ -f pretrain/parenet.tar ] || { echo "PARENet weights not found in pretrain/ (README, step 2)" >&2; exit 1; }

if [ -z "$CATEGORIES" ]; then
    CATEGORIES=$(python -c "from utils.dataset import all_real3dad_cates; print(' '.join(all_real3dad_cates))")
fi

for c in $CATEGORIES; do
    if [ ! -f spatial_memory_banks/${c}_spatial_chamfer_memory_bank.pt ]; then
        echo "=== $c: building the memory bank"
        python compute_train_variability.py $CFG --category $c --use_adam_9dof --save_transforms --first_sample_only
        python build_spatial_fpfh_memory_bank.py $CFG --category $c
    fi
    echo "=== $c: generating synthetic anomalies"
    python generate_synthetic_anomalies.py $CFG --category $c
    echo "=== $c: training"
    python train_fpfh_chamfer.py \
        --training_config configs/training/sigmoid_fusion.yaml \
        --dataset_config $CFG \
        --category $c
done

CATEGORIES="$CATEGORIES" scripts/evaluate.sh
