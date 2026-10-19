"""
Setup script for QuickFPS CUDA extension.

Install with:
    cd cuda_ops/fps_kdtree
    python setup.py install
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cudakdtree_path = os.path.join(current_dir, '../cudaKDTree')

setup(
    name='fps_kdtree_quickfps_cuda',
    version='1.0.0',
    author='',
    description='QuickFPS: Bucket-based FPS with KDTree pruning (CUDA)',
    ext_modules=[
        CUDAExtension(
            name='fps_kdtree_quickfps_cuda',
            sources=[
                os.path.join(current_dir, 'fps_kdtree_quickfps_cuda.cpp'),
                os.path.join(current_dir, 'fps_kdtree_quickfps_kernel.cu'),
            ],
            include_dirs=[cudakdtree_path],  # Add cudaKDTree for GPU tree building
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math', '-std=c++17', '-DTORCH_USE_CUDA_DSA']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.7.0',
        'numpy>=1.18.0'
    ]
)
