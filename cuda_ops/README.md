# CUDA extension

`fps_kdtree/` implements farthest point sampling on a GPU KD-tree, built on [cudaKDTree](https://github.com/ingowald/cudaKDTree) (git submodule in `cudaKDTree/`).

It is compiled automatically by the Docker container (or by `scripts/build_extensions.sh`). To rebuild it by hand:

```bash
./cuda_ops/build.sh
```

Usage:

```python
from cuda_ops.fps_kdtree.fps_kdtree_quickfps_gpu import fps_kdtree_quickfps
indices = fps_kdtree_quickfps(points, num_samples)   # points: (N, 3) CUDA tensor
```
