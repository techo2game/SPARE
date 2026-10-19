#!/bin/bash
# FPS KDTree CUDA Extension - Build script
# Usage: ./build.sh

set -e

cd "$(dirname "$0")"

echo "================================"
echo "Building FPS KDTree Extension"
echo "================================"
echo ""

# Clean old builds
echo "[1/4] Cleaning old builds..."
rm -rf build dist *.egg-info
rm -f fps_kdtree_quickfps_cuda*.so
echo "Done"
echo ""

# Build extension
echo "[2/4] Building extension..."
python setup.py build_ext
echo "Done"
echo ""

# Find the built .so file
echo "[3/4] Locating built extension..."
SO_FILE=$(find build/lib.* -name "fps_kdtree_quickfps_cuda*.so" | head -1)

if [ -z "$SO_FILE" ]; then
    echo "Error: Could not find built .so file"
    exit 1
fi

echo "Found: $SO_FILE"
echo ""

# Copy to current directory
echo "[4/4] Installing..."
cp "$SO_FILE" .
echo "Done"
echo ""

# Verify
echo "Verifying..."
python -c "import torch, fps_kdtree_quickfps_cuda; print('OK')"

echo ""
echo "BUILD SUCCESSFUL"
echo "Usage: from cuda_ops.fps_kdtree.fps_kdtree_quickfps_gpu import fps_kdtree_quickfps"
