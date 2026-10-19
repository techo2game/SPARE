#!/bin/bash
# Compile the C++/CUDA extensions SPARE needs. The Docker container runs this
# automatically on first start (docker/entrypoint.sh).
# Also fetches the submodules and applies SPARE's small patch to PARENet.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git submodule update --init --recursive

# Apply SPARE's PARENet patch once; fails if it does not apply
PATCH=../patches/parenet_torch_cluster_knn.patch
if git -C PARENet apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "PARENet patch already applied"
else
    git -C PARENet apply "$PATCH"
    echo "Applied patches/parenet_torch_cluster_knn.patch"
fi

# PARENet C++ extension (pareconv.ext)
(cd PARENet && python setup.py build_ext --inplace)

# SPARE CUDA op (farthest point sampling)
./cuda_ops/build.sh
