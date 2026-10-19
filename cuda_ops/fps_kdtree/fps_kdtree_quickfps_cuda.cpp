/*
 * QuickFPS PyTorch C++ Extension
 * PyBind11 wrapper for QuickFPS CUDA kernels
 * Uses cudaKDTree for GPU-parallel tree construction
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

// Bucket metadata (must match kernel definition)
struct KDTreeBucket {
    float bbox_min[3];
    float bbox_max[3];
    int point_left;
    int point_right;
    int max_point_id;
    float max_distance;
    int pad[2];
};

// CUDA kernels (implemented in .cu file)
extern "C" {
    void build_kdtree_gpu(float* d_points, int N, int bucket_size);

    void extract_buckets_from_implicit_tree(
        const float* d_points,
        int N,
        int bucket_size,
        KDTreeBucket** d_buckets_out,
        int* num_buckets_out
    );

    void fps_kdtree_quickfps_kernel(
        const float* d_input_points,
        int N,
        int n_samples,
        int bucket_size,
        int start_idx,
        const KDTreeBucket* d_buckets,  // GPU pointer!
        int num_buckets,
        int* h_sampled_indices
    );
}

/**
 * QuickFPS: Bucket-based FPS with KDTree pruning
 *
 * @param points: (N, 3) torch.Tensor on CUDA, dtype=float32
 * @param n_samples: Number of points to sample
 * @param bucket_size: Max points per KDTree leaf bucket
 * @param start_idx: Starting point index
 * @return: (n_samples,) torch.LongTensor of selected point indices
 */
torch::Tensor fps_kdtree_quickfps(
    torch::Tensor points,
    int64_t n_samples,
    int64_t bucket_size,
    int64_t start_idx
) {
    // Input validation
    TORCH_CHECK(points.is_cuda(), "points must be on CUDA device");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                "points must have shape (N, 3)");

    int N = points.size(0);
    TORCH_CHECK(n_samples > 0 && n_samples <= N,
                "n_samples must be in range [1, N]");
    TORCH_CHECK(bucket_size > 0,
                "bucket_size must be positive");
    TORCH_CHECK(start_idx >= 0 && start_idx < N,
                "start_idx must be in range [0, N)");

    // Set CUDA device to match input tensor
    int device = points.get_device();
    cudaError_t err_device = cudaSetDevice(device);
    TORCH_CHECK(err_device == cudaSuccess,
                "cudaSetDevice failed: ", cudaGetErrorString(err_device));

    // Ensure points are contiguous and make a working copy
    auto points_contiguous = points.contiguous();

    // Create working copy of points on GPU (cudaKDTree will reorder in-place!)
    auto points_copy = points_contiguous.clone();
    float* d_points = points_copy.data_ptr<float>();

    // Build KDTree on GPU using SpatialKDTree (FAST! ~20ms for 1M points)
    // Creates explicit node structure with configured bucket size
    build_kdtree_gpu(d_points, N, bucket_size);

    // Extract leaf buckets on GPU (FULLY GPU-PARALLEL!)
    KDTreeBucket* d_buckets;
    int num_buckets;
    extract_buckets_from_implicit_tree(
        d_points,       // GPU pointer to reordered points
        N,
        bucket_size,
        &d_buckets,     // Returns GPU pointer
        &num_buckets    // Returns host value
    );

    // Allocate output on host
    std::vector<int> h_sampled_indices(n_samples);

    // Call CUDA kernel with REORDERED points from cudaKDTree
    // IMPORTANT: Use d_points (reordered by tree), not original points!
    fps_kdtree_quickfps_kernel(
        d_points,       // GPU pointer to REORDERED points!
        N,
        n_samples,
        bucket_size,
        start_idx,
        d_buckets,      // GPU pointer to buckets!
        num_buckets,
        h_sampled_indices.data()
    );

    // Cleanup GPU buckets
    cudaFree(d_buckets);

    // Copy results to PyTorch tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(points.device());

    auto output = torch::empty({n_samples}, options);
    auto output_cpu = output.cpu();

    for (int i = 0; i < n_samples; i++) {
        output_cpu[i] = h_sampled_indices[i];
    }

    return output_cpu.to(points.device());
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fps_kdtree_quickfps", &fps_kdtree_quickfps,
          "QuickFPS: Bucket-based FPS with KDTree pruning (CUDA)",
          py::arg("points"),
          py::arg("n_samples"),
          py::arg("bucket_size") = 200,
          py::arg("start_idx") = 0);
}
