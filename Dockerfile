# Reference environment for SPARE: CUDA 12.4, gcc 11, Python 3.10, PyTorch 2.4.0.
# Use ./run.sh, which builds this image and starts it with the right options.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python-is-python3 \
        git build-essential ninja-build \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Exact versions resolved in this image
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install -r /tmp/requirements.lock.txt

COPY docker/entrypoint.sh /usr/local/bin/spare-entrypoint
WORKDIR /workspace/SPARE
ENTRYPOINT ["/usr/local/bin/spare-entrypoint"]
CMD ["bash"]
