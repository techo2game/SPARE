# Third-party code

SPARE's GPL-3.0 license covers only the code in this repository. The
dependencies below are **git submodules**. They are fetched from their
upstream repositories and are not redistributed here. Each one stays under its
own license.

| Path | Upstream | Pinned commit | License |
|------|----------|---------------|---------|
| `PARENet/` | https://github.com/yaorz97/PARENet | `71baada` | No license file upstream at the pinned commit. Check with the authors before redistributing it. |
| `cuda_ops/cudaKDTree/` | https://github.com/ingowald/cudaKDTree | `0e174fe` | Apache-2.0 |

## Local modifications

- `patches/parenet_torch_cluster_knn.patch` replaces PARENet's `pointops` KNN
  query with `torch_cluster.knn`, so the `pointops` CUDA extension is not
  needed. `scripts/build_extensions.sh` applies it.
- `parenet_real3dad/` holds SPARE's Real3D-AD training setup for PARENet
  (config, dataset, train/test scripts). It imports the unmodified upstream
  `backbone`, `model` and `loss` modules from `PARENet/experiments/3DMatch`.

If you use PARENet, please cite:

> Yao et al., "PARE-Net: Position-Aware Rotation-Equivariant Networks for
> Robust Point Cloud Registration", ECCV 2024.
