#!/bin/bash
# Build SPARE's CUDA extension: farthest point sampling on a GPU KD-tree.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building FPS KDTree..."
cd "$SCRIPT_DIR/fps_kdtree"
./build.sh
