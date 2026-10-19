#!/bin/bash
# Container entrypoint: compile SPARE's C++/CUDA extensions on first start.
# The repository is mounted at run time, so they cannot be built into the image;
# they are written into the mounted repository and later starts skip this step.
set -e

if [ ! -f scripts/build_extensions.sh ]; then
    exec "$@"   # repository not mounted: plain environment
fi

if ! ls PARENet/pareconv/ext*.so \
        cuda_ops/fps_kdtree/fps_kdtree_quickfps_cuda*.so >/dev/null 2>&1; then
    echo "First start: compiling the C++/CUDA extensions (about a minute)..."
    if ! ./scripts/build_extensions.sh > build_extensions.log 2>&1; then
        tail -n 30 build_extensions.log
        echo "Compiling the extensions failed; full log in build_extensions.log" >&2
        exit 1
    fi
    echo "Done."
fi

exec "$@"
