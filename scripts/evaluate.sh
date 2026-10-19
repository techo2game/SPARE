#!/bin/bash
# Evaluate SPARE on Real3D-AD with the models in logs/ (pretrained or your own).
# By default, every category that has a model; to choose:
#   CATEGORIES="airplane chicken" scripts/evaluate.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CFG=configs/real3d-ad/base.yaml

if [ -z "$CATEGORIES" ]; then
    CATEGORIES=$(ls -d logs/*/*/checkpoints 2>/dev/null | cut -d/ -f2 | sort -u | tr '\n' ' ')
fi
[ -n "$CATEGORIES" ] || { echo "No models in logs/: extract the pretrained models (README, step 2) or run scripts/train.sh" >&2; exit 1; }
[ -d data/Real3D/airplane/train ] || { echo "Real3D-AD not found in data/Real3D (README, step 2)" >&2; exit 1; }

# Memory banks: register the training samples, then build the bank (once per category)
for c in $CATEGORIES; do
    if [ ! -f spatial_memory_banks/${c}_spatial_chamfer_memory_bank.pt ]; then
        echo "=== $c: building the memory bank"
        python compute_train_variability.py $CFG --category $c --use_adam_9dof --save_transforms --first_sample_only
        python build_spatial_fpfh_memory_bank.py $CFG --category $c
    fi
done

python test_fusion_model.py \
    --dataset_config $CFG \
    --testing_config configs/testing/sigmoid_fusion.yaml \
    --category $CATEGORIES
