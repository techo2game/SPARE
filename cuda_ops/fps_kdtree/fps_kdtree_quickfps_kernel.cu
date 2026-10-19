/*
 * QuickFPS CUDA Kernels
 * Implements bucket-based FPS with KDTree hierarchical pruning
 * FULLY GPU-PARALLEL: Tree extraction and FPS all on GPU
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <float.h>
#include <stdio.h>

// cudaKDTree for GPU tree building
#include "../cudaKDTree/cukd/spatial-kdtree.h"

#define THREADS_PER_BLOCK 256

// Forward declare Node structure (must match cukd::SpatialKDTree<float3>::Node)
struct CUKDNode {
    float pos;           // Split position
    uint32_t offset;     // First child or primID offset
    uint16_t count;      // Number of prims (>0 = leaf)
    int16_t  dim;        // Split dimension
};

// Bucket metadata (leaf nodes only)
struct KDTreeBucket {
    float bbox_min[3];   // Bounding box min
    float bbox_max[3];   // Bounding box max
    int point_left;      // Start index in points array
    int point_right;     // End index (exclusive)
    int max_point_id;    // Current max distance point ID
    float max_distance;  // Current max distance value
    int pad[2];          // Padding to 48 bytes
};

// Point representation (32 bytes aligned)
struct Point3D {
    float pos[3];        // Position (12 bytes)
    float dis;           // Min distance to sampled set (4 bytes)
    int id;              // Original index (4 bytes)
    int bucket_id;       // Bucket assignment (4 bytes)
    int pad[2];          // Padding to 32 bytes
};

//=============================================================================
// Device Helper Functions
//=============================================================================

/**
 * Compute squared Euclidean distance between two 3D points
 */
__device__ __forceinline__ float euclidean_distance_sq(
    const float* a,
    const float* b
) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return dx*dx + dy*dy + dz*dz;
}

/**
 * Compute distance from point to bounding box
 * Returns 0 if point is inside bbox
 */
__device__ __forceinline__ float bound_distance(
    const float* bbox_min,
    const float* bbox_max,
    const float* point
) {
    float dx = fmaxf(bbox_min[0] - point[0], fmaxf(0.0f, point[0] - bbox_max[0]));
    float dy = fmaxf(bbox_min[1] - point[1], fmaxf(0.0f, point[1] - bbox_max[1]));
    float dz = fmaxf(bbox_min[2] - point[2], fmaxf(0.0f, point[2] - bbox_max[2]));
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

/**
 * Atomic min for floats (CUDA doesn't provide this)
 */
__device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;

    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        if (value >= old_val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

//=============================================================================
// Kernel: Extract Leaf Buckets from Implicit Tree (GPU)
//=============================================================================

/**
 * Compute node range directly from node index in left-balanced binary tree
 *
 * For a left-balanced tree with N points:
 * - Level L has nodes at indices [2^L - 1, 2^(L+1) - 2]
 * - Each node at level L covers approximately N / 2^L points
 * - Use bit operations for fast computation
 */
__device__ void compute_node_range(int node_idx, int N, int* start, int* end) {
    if (node_idx == 0) {
        *start = 0;
        *end = N;
        return;
    }

    // Find level: level = floor(log2(node_idx + 1))
    // Using __clz (count leading zeros) for fast log2
    int level = 31 - __clz(node_idx + 1);

    // First node at this level
    int level_start = (1 << level) - 1;

    // Position within this level (0-indexed)
    int pos_in_level = node_idx - level_start;

    // Each node at this level covers N / 2^level points (approximately)
    int points_per_node = N >> level;  // N / 2^level
    int remainder = N - (points_per_node << level);  // N % 2^level

    // First 'remainder' nodes get one extra point to distribute N points evenly
    if (pos_in_level < remainder) {
        *start = pos_in_level * (points_per_node + 1);
        *end = *start + points_per_node + 1;
    } else {
        *start = remainder * (points_per_node + 1) +
                 (pos_in_level - remainder) * points_per_node;
        *end = *start + points_per_node;
    }
}

/**
 * Extract leaf buckets from SpatialKDTree's explicit nodes
 * Each thread processes one node and checks if it's a leaf
 */
__global__ void extract_buckets_from_spatial_tree_kernel(
    const CUKDNode* d_nodes,             // Explicit nodes from SpatialKDTree
    const uint32_t* d_primIDs,           // Point indices per leaf
    const float3* d_data,                // Original points (NOT reordered!)
    int num_nodes,
    KDTreeBucket* d_buckets,             // Output buckets
    int* d_bucket_count                  // Atomic counter
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;

    const CUKDNode& node = d_nodes[node_idx];

    // Check if this is a leaf (count > 0 means leaf)
    if (node.count == 0) return;  // Inner node, skip

    // This is a leaf - compute bounding box from points
    float bbox_min[3] = {1e10f, 1e10f, 1e10f};
    float bbox_max[3] = {-1e10f, -1e10f, -1e10f};

    int start = node.offset;
    int end = node.offset + node.count;

    for (int i = start; i < end; i++) {
        uint32_t prim_id = d_primIDs[i];
        const float3& pt = d_data[prim_id];

        bbox_min[0] = fminf(bbox_min[0], pt.x);
        bbox_min[1] = fminf(bbox_min[1], pt.y);
        bbox_min[2] = fminf(bbox_min[2], pt.z);

        bbox_max[0] = fmaxf(bbox_max[0], pt.x);
        bbox_max[1] = fmaxf(bbox_max[1], pt.y);
        bbox_max[2] = fmaxf(bbox_max[2], pt.z);
    }

    // Atomically allocate bucket slot
    int bucket_id = atomicAdd(d_bucket_count, 1);

    // Store bucket
    for (int d = 0; d < 3; d++) {
        d_buckets[bucket_id].bbox_min[d] = bbox_min[d];
        d_buckets[bucket_id].bbox_max[d] = bbox_max[d];
    }
    d_buckets[bucket_id].point_left = start;
    d_buckets[bucket_id].point_right = end;
    d_buckets[bucket_id].max_point_id = -1;
    d_buckets[bucket_id].max_distance = -1.0f;
}

//=============================================================================
// Kernel: Initialize Points
//=============================================================================

__global__ void init_points_kernel(
    Point3D* d_points,           // Output: Point3D array
    const float* input_points,   // Input: (N, 3) float array (REORDERED by cudaKDTree!)
    int N,
    int start_idx                // First sampled point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Point3D pt;
    pt.pos[0] = input_points[idx * 3 + 0];
    pt.pos[1] = input_points[idx * 3 + 1];
    pt.pos[2] = input_points[idx * 3 + 2];

    // Initialize distance
    if (idx == start_idx) {
        pt.dis = 0.0f;  // First sampled point has distance 0
    } else {
        // Compute distance to first sampled point
        float ref_pos[3] = {
            input_points[start_idx * 3 + 0],
            input_points[start_idx * 3 + 1],
            input_points[start_idx * 3 + 2]
        };
        pt.dis = sqrtf(euclidean_distance_sq(pt.pos, ref_pos));
    }

    pt.id = idx;
    pt.bucket_id = -1;  // Will be assigned later

    d_points[idx] = pt;
}

//=============================================================================
// Kernel: Bucket Reduction (Find Max Distance Point Per Bucket)
//=============================================================================

__global__ void bucket_reduction_kernel(
    const Point3D* d_points,
    KDTreeBucket* d_buckets,
    int num_buckets
) {
    int bucket_id = blockIdx.x;
    if (bucket_id >= num_buckets) return;

    KDTreeBucket& bucket = d_buckets[bucket_id];
    int start = bucket.point_left;
    int end = bucket.point_right;

    // Shared memory for reduction
    __shared__ float s_max_dis[THREADS_PER_BLOCK];
    __shared__ int s_max_id[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    float local_max_dis = -1.0f;
    int local_max_id = -1;

    // Each thread processes points
    for (int i = start + tid; i < end; i += blockDim.x) {
        float dis = d_points[i].dis;
        if (dis > local_max_dis) {
            local_max_dis = dis;
            local_max_id = i;
        }
    }

    s_max_dis[tid] = local_max_dis;
    s_max_id[tid] = local_max_id;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max_dis[tid + stride] > s_max_dis[tid]) {
                s_max_dis[tid] = s_max_dis[tid + stride];
                s_max_id[tid] = s_max_id[tid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (tid == 0) {
        bucket.max_distance = s_max_dis[0];
        bucket.max_point_id = s_max_id[0];
    }
}

//=============================================================================
// Kernel: Find Global Max Across Buckets
//=============================================================================

__global__ void find_global_max_kernel(
    const KDTreeBucket* d_buckets,
    int num_buckets,
    int* d_global_max_id,
    float* d_global_max_dis
) {
    __shared__ float s_max_dis[THREADS_PER_BLOCK];
    __shared__ int s_max_id[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    float local_max_dis = -1.0f;
    int local_max_id = -1;

    for (int i = tid; i < num_buckets; i += blockDim.x) {
        if (d_buckets[i].max_distance > local_max_dis) {
            local_max_dis = d_buckets[i].max_distance;
            local_max_id = d_buckets[i].max_point_id;
        }
    }

    s_max_dis[tid] = local_max_dis;
    s_max_id[tid] = local_max_id;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max_dis[tid + stride] > s_max_dis[tid]) {
                s_max_dis[tid] = s_max_dis[tid + stride];
                s_max_id[tid] = s_max_id[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_global_max_id = s_max_id[0];
        *d_global_max_dis = s_max_dis[0];
    }
}

//=============================================================================
// Kernel: Update Distances (Core FPS with Bucket Pruning)
//=============================================================================

__global__ void update_distances_kernel(
    Point3D* d_points,
    KDTreeBucket* d_buckets,
    int num_buckets,
    const int* d_ref_point_idx  // GPU pointer to current reference point index
) {
    int bucket_id = blockIdx.x;
    if (bucket_id >= num_buckets) return;

    int ref_idx = *d_ref_point_idx;
    const Point3D& ref_point = d_points[ref_idx];

    KDTreeBucket& bucket = d_buckets[bucket_id];
    float lastmax_distance = bucket.max_distance;

    // Pruning check (thread 0 only)
    if (threadIdx.x == 0) {
        // Check current max point distance to ref
        if (bucket.max_point_id >= 0) {
            const Point3D& max_pt = d_points[bucket.max_point_id];
            float cur_distance = sqrtf(euclidean_distance_sq(max_pt.pos, ref_point.pos));

            if (cur_distance > lastmax_distance) {
                // Check bounding box distance
                float boundary_dist = bound_distance(bucket.bbox_min, bucket.bbox_max, ref_point.pos);

                if (boundary_dist >= lastmax_distance) {
                    // PRUNE: entire bucket can be skipped
                    return;
                }
                // Otherwise: need to update (fall through)
            }
        }
    }
    __syncthreads();  // Ensure all threads agree on pruning decision

    // Update all points in bucket
    int start = bucket.point_left;
    int end = bucket.point_right;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float dist_sq = euclidean_distance_sq(d_points[i].pos, ref_point.pos);
        float dist = sqrtf(dist_sq);

        // Atomic min to update distance
        atomicMinFloat(&d_points[i].dis, dist);
    }
}

//=============================================================================
// Host Interface Functions
//=============================================================================

extern "C" {

// GPU tree structure
using data_t = float3;
using data_traits = cukd::default_data_traits<float3>;
using SpatialTree = cukd::SpatialKDTree<data_t, data_traits>;
static SpatialTree* g_tree = nullptr;

/**
 * Build KDTree on GPU using Spatial KDTree (explicit node structure)
 */
void build_kdtree_gpu(float* d_points, int N, int bucket_size) {
    // Allocate tree structure if needed
    if (!g_tree) {
        g_tree = new SpatialTree();
    }

    float3* d_points_f3 = reinterpret_cast<float3*>(d_points);

    // Configure bucket size
    cukd::BuildConfig config;
    config.makeLeafThreshold = bucket_size;

    // Build tree with explicit nodes
    cukd::buildTree(*g_tree, d_points_f3, N, config);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR in build_kdtree_gpu: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // printf("DEBUG: Tree built successfully, N=%d, num_nodes=%d\n",
    //        N, g_tree->numNodes);
}

/**
 * Extract leaf buckets from SpatialKDTree (GPU)
 */
void extract_buckets_from_implicit_tree(
    const float* d_points,           // GPU pointer to points (ignored - use tree's data)
    int N,
    int bucket_size,
    KDTreeBucket** d_buckets_out,    // Allocated and returned (GPU pointer)
    int* num_buckets_out             // Number of buckets found (host value)
) {
    if (!g_tree) {
        printf("ERROR: Tree not built yet!\n");
        return;
    }

    // Allocate maximum possible buckets (one per node worst case)
    KDTreeBucket* d_buckets;
    cudaMalloc(&d_buckets, g_tree->numNodes * sizeof(KDTreeBucket));

    // Atomic counter for bucket count
    int* d_bucket_count;
    cudaMalloc(&d_bucket_count, sizeof(int));
    cudaMemset(d_bucket_count, 0, sizeof(int));

    // Launch kernel to extract buckets from SpatialKDTree nodes
    int threads = 256;
    int blocks = (g_tree->numNodes + threads - 1) / threads;

    // printf("DEBUG: Extracting buckets from %d nodes\n", g_tree->numNodes);

    extract_buckets_from_spatial_tree_kernel<<<blocks, threads>>>(
        reinterpret_cast<const CUKDNode*>(g_tree->nodes),  // GPU pointer to nodes
        g_tree->primIDs,                                   // GPU pointer to point indices
        g_tree->data,                                      // GPU pointer to points (float3)
        g_tree->numNodes,
        d_buckets,
        d_bucket_count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR launching extract_buckets_kernel: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR after extract_buckets_kernel sync: %s\n", cudaGetErrorString(err));
    }

    // Get actual bucket count
    int h_bucket_count;
    cudaMemcpy(&h_bucket_count, d_bucket_count, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("DEBUG: Extracted %d leaf buckets from %d nodes\n", h_bucket_count, g_tree->numNodes);

    cudaFree(d_bucket_count);

    // Return results
    *d_buckets_out = d_buckets;
    *num_buckets_out = h_bucket_count;
}

/**
 * Simple kernel to copy a single value to an array slot
 */
__global__ void copy_index_kernel(const int* src, int* dst, int offset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[offset] = *src;
    }
}

/**
 * QuickFPS main kernel (called from host)
 */
void fps_kdtree_quickfps_kernel(
    const float* d_input_points,  // GPU pointer to REORDERED points
    int N,
    int n_samples,
    int bucket_size,
    int start_idx,
    const KDTreeBucket* d_buckets,  // GPU pointer to buckets
    int num_buckets,
    int* h_sampled_indices          // Output: sampled indices (host array)
) {
    // Allocate Point3D array on GPU
    Point3D* d_points;
    cudaMalloc(&d_points, N * sizeof(Point3D));

    // Initialize points
    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;
    init_points_kernel<<<blocks, threads>>>(d_points, d_input_points, N, start_idx);

    // Allocate working memory
    int* d_global_max_id;
    float* d_global_max_dis;
    int* d_sampled_indices;  // GPU array for sampled indices
    cudaMalloc(&d_global_max_id, sizeof(int));
    cudaMalloc(&d_global_max_dis, sizeof(float));
    cudaMalloc(&d_sampled_indices, n_samples * sizeof(int));

    // Copy buckets to device (mutable copy for bucket state)
    KDTreeBucket* d_buckets_mut;
    cudaMalloc(&d_buckets_mut, num_buckets * sizeof(KDTreeBucket));
    cudaMemcpy(d_buckets_mut, d_buckets, num_buckets * sizeof(KDTreeBucket),
               cudaMemcpyDeviceToDevice);

    // First sampled point (directly to GPU)
    cudaMemcpy(d_sampled_indices, &start_idx, sizeof(int), cudaMemcpyHostToDevice);

    // Main FPS loop (ZERO HOST SYNC!)
    for (int sample_idx = 1; sample_idx < n_samples; sample_idx++) {
        // Find max distance point per bucket
        bucket_reduction_kernel<<<num_buckets, THREADS_PER_BLOCK>>>(
            d_points, d_buckets_mut, num_buckets
        );

        // Find global max across buckets
        find_global_max_kernel<<<1, THREADS_PER_BLOCK>>>(
            d_buckets_mut, num_buckets,
            d_global_max_id, d_global_max_dis
        );

        // Store selected index to GPU array (NO D2H COPY!)
        copy_index_kernel<<<1, 1>>>(d_global_max_id, d_sampled_indices, sample_idx);

        // Update distances with pruning
        update_distances_kernel<<<num_buckets, THREADS_PER_BLOCK>>>(
            d_points, d_buckets_mut, num_buckets,
            d_global_max_id  // GPU pointer!
        );
    }

    // Copy ALL sampled indices to host in ONE transfer
    cudaMemcpy(h_sampled_indices, d_sampled_indices, n_samples * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_global_max_id);
    cudaFree(d_global_max_dis);
    cudaFree(d_sampled_indices);
    cudaFree(d_buckets_mut);
}

} // extern "C"