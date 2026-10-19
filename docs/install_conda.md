# Installation with conda

The [Docker quick start](../README.md#quick-start) is the easiest way to get started. This page installs the same environment with conda instead.

## Requirements

- Linux with an NVIDIA GPU
- CUDA toolkit 12.4 (`nvcc`) and a compatible host compiler (gcc 13 or older)

The reference environment (the Docker image) is Ubuntu 22.04 with gcc 11. CUDA 12.4 does not support gcc 14+, and its headers do not compile with glibc 2.41+ (e.g. Ubuntu 25.04). On such systems, use Docker.

## Steps

```bash
git clone --recursive <this-repo-url> SPARE
cd SPARE

conda create -n spare python=3.10
conda activate spare

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install -r requirements.lock.txt

./scripts/build_extensions.sh
```

If your default gcc is newer than 13 but an older one is installed, select it for the build:

```bash
CC=gcc-13 CXX=g++-13 ./scripts/build_extensions.sh
```

## Data and pretrained models

Download Real3D-AD and the pretrained models and place them as described in the [README, step 2](../README.md#2-download-the-data-and-the-pretrained-models). If Real3D-AD is stored elsewhere, link it: `mkdir -p data && ln -s /path/to/Real3D-AD data/Real3D`.

## Run

```bash
scripts/evaluate.sh   # evaluate the models in logs/
scripts/train.sh      # train, then evaluate
```

Both accept `CATEGORIES="airplane car"` to run some categories only.
