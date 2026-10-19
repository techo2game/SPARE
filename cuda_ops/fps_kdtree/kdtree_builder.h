/*
 * KDTree Builder for QuickFPS
 * Builds bucket-based KDTree (KDLineTree variant) on CPU
 *
 * Algorithm:
 * 1. Recursively subdivide point cloud along longest axis (median split)
 * 2. Stop when bucket_size reached or max_depth exceeded
 * 3. Extract only leaf buckets (KDLineTree pattern)
 * 4. Reorder points for coalesced GPU access
 */

#ifndef KDTREE_BUILDER_H
#define KDTREE_BUILDER_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

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

// Build configuration
struct BuildConfig {
    int bucket_size;     // Max points per leaf (e.g., 200)
    int max_depth;       // Max tree depth (e.g., 20)

    BuildConfig() : bucket_size(200), max_depth(20) {}
};

class KDTreeBuilder {
public:
    KDTreeBuilder();
    ~KDTreeBuilder();

    /**
     * Build KDTree from point cloud
     *
     * @param points: (N, 3) point cloud (read-only, not modified)
     * @param N: Number of points
     * @param config: Build configuration
     * @return: Vector of leaf buckets
     */
    std::vector<KDTreeBucket> build(
        const float* points,  // (N, 3) - read-only, NOT modified
        int N,
        const BuildConfig& config
    );

    /**
     * Get mapping from reordered indices to original indices
     * Call this after build() to get the index mapping
     *
     * @return: Vector where result[i] = original_index of point i in reordered array
     */
    std::vector<int> get_original_indices() const;

private:
    // Internal structures
    struct PointIndex {
        int idx;            // Original point index
        float pos[3];       // Point position
    };

    struct TreeNode {
        std::vector<int> point_indices;  // Indices of points in this node
        float bbox_min[3];               // Bounding box
        float bbox_max[3];
        int depth;                       // Node depth

        // Leaf node indicator
        bool is_leaf() const {
            return left == nullptr && right == nullptr;
        }

        TreeNode* left;
        TreeNode* right;

        TreeNode() : left(nullptr), right(nullptr), depth(0) {}
        ~TreeNode() {
            delete left;
            delete right;
        }
    };

    // Build configuration
    BuildConfig config_;

    // Point storage (reordered during build)
    std::vector<PointIndex> points_;

    // Recursive tree construction
    TreeNode* subdivide(
        std::vector<int>& point_ids,
        int depth,
        const float* bbox_min,
        const float* bbox_max
    );

    // Extract leaf buckets from tree
    void extract_leaves(
        TreeNode* node,
        std::vector<KDTreeBucket>& buckets,
        std::vector<int>& reordered_indices
    );

    // Compute bounding box for point set
    void compute_bbox(
        const std::vector<int>& point_ids,
        float* bbox_min,
        float* bbox_max
    );

    // Find longest axis of bounding box
    int find_split_axis(
        const float* bbox_min,
        const float* bbox_max
    );

    // Partition points along axis (median split)
    int partition_points(
        std::vector<int>& point_ids,
        int axis,
        float* left_bbox_max,
        float* right_bbox_min
    );
};

#endif // KDTREE_BUILDER_H
