#!/bin/bash
# Start SPARE's Docker environment (the repository is mounted inside it).
#
#   ./run.sh                        open a shell
#   ./run.sh scripts/evaluate.sh    run a single command
#
# The first run builds the image and compiles the C++/CUDA extensions.
# To use an existing copy of Real3D-AD:  REAL3D_AD=/path/to/Real3D-AD ./run.sh
set -e
cd "$(dirname "$(readlink -f "$0")")"
IMAGE=spare

command -v docker >/dev/null || { echo "Docker is required: https://docs.docker.com/engine/install/" >&2; exit 1; }

# Submodules (in case the repository was cloned without --recursive)
[ -f PARENet/setup.py ] || git submodule update --init --recursive

# Build the image if missing or if its definition changed
HASH=$(cat Dockerfile requirements.lock.txt docker/entrypoint.sh | sha256sum | cut -c1-16)
if [ "$(docker image inspect -f '{{index .Config.Labels "spare.hash"}}' $IMAGE 2>/dev/null)" != "$HASH" ]; then
    echo "Building the SPARE Docker image (first run only, several minutes)..."
    docker build --label spare.hash=$HASH -t $IMAGE .
fi

# Dataset: data/Real3D (see README), or an existing copy mounted there
MOUNTS=(-v "$PWD":/workspace/SPARE)
if [ -n "$REAL3D_AD" ]; then
    [ -d "$REAL3D_AD/airplane/train" ] || { echo "REAL3D_AD=$REAL3D_AD does not contain Real3D-AD (no airplane/train)" >&2; exit 1; }
    mkdir -p data/Real3D   # mount point, created as the user rather than by Docker as root
    MOUNTS+=(-v "$(readlink -f "$REAL3D_AD")":/workspace/SPARE/data/Real3D:ro)
fi

# Options of the workflow scripts
ENVS=()
[ -n "$CATEGORIES" ] && ENVS+=(-e CATEGORIES)

TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)

status=0
docker run --rm "${TTY[@]}" --gpus all --user "$(id -u):$(id -g)" "${MOUNTS[@]}" "${ENVS[@]}" $IMAGE "$@" || status=$?
if [ $status -eq 125 ]; then
    echo "Docker could not start the container. GPU access requires the NVIDIA Container Toolkit:" >&2
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" >&2
fi
exit $status
