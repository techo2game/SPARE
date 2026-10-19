/*
 * KDTree Builder Implementation
 * CPU-based tree construction for QuickFPS
 */

#include "kdtree_builder.h"
#include <iostream>
#include <cstring>

KDTreeBuilder::KDTreeBuilder() {}

KDTreeBuilder::~KDTreeBuilder() {}

std::vector<KDTreeBucket> KDTreeBuilder::build(
    const float* points,
    int N,
    const BuildConfig& config
) {
    config_ = config;

    // Initialize point storage (internal copy for tree building)
    points_.resize(N);
    for (int i = 0; i < N; i++) {
        points_[i].idx = i;
        points_[i].pos[0] = points[i * 3 + 0];
        points_[i].pos[1] = points[i * 3 + 1];
        points_[i].pos[2] = points[i * 3 + 2];
    }

    // Initial point indices (all points)
    std::vector<int> all_indices(N);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    // Compute global bounding box
    float bbox_min[3], bbox_max[3];
    compute_bbox(all_indices, bbox_min, bbox_max);

    // Recursively build tree
    TreeNode* root = subdivide(all_indices, 0, bbox_min, bbox_max);

    // Extract leaf buckets with reordered indices
    std::vector<KDTreeBucket> buckets;
    reordered_indices_.clear();
    extract_leaves(root, buckets, reordered_indices_);

    // Buckets now store ranges into reordered_indices_ array
    // The caller must use this mapping to access points correctly

    // Cleanup
    delete root;

    return buckets;
}

std::vector<int> KDTreeBuilder::get_original_indices() const {
    // Return mapping: reordered_index → original_index
    std::vector<int> mapping(points_.size());
    for (size_t i = 0; i < points_.size(); i++) {
        mapping[i] = points_[i].idx;
    }
    return mapping;
}

KDTreeBuilder::TreeNode* KDTreeBuilder::subdivide(
    std::vector<int>& point_ids,
    int depth,
    const float* bbox_min,
    const float* bbox_max
) {
    TreeNode* node = new TreeNode();
    node->depth = depth;
    node->point_indices = point_ids;

    // Copy bounding box
    std::memcpy(node->bbox_min, bbox_min, 3 * sizeof(float));
    std::memcpy(node->bbox_max, bbox_max, 3 * sizeof(float));

    // Check termination conditions
    int count = point_ids.size();
    if (count <= config_.bucket_size || depth >= config_.max_depth) {
        // Leaf node
        return node;
    }

    // Find split axis (longest dimension)
    int split_axis = find_split_axis(bbox_min, bbox_max);

    // Partition points along axis (median split)
    float left_bbox_max[3], right_bbox_min[3];
    std::memcpy(left_bbox_max, bbox_max, 3 * sizeof(float));
    std::memcpy(right_bbox_min, bbox_min, 3 * sizeof(float));

    int split_idx = partition_points(point_ids, split_axis,
                                      left_bbox_max, right_bbox_min);

    // If partition failed (all points on one side), make leaf
    if (split_idx == 0 || split_idx == count) {
        return node;
    }

    // Split point indices
    std::vector<int> left_ids(point_ids.begin(), point_ids.begin() + split_idx);
    std::vector<int> right_ids(point_ids.begin() + split_idx, point_ids.end());

    // Recursively subdivide children
    float left_bbox_min[3], right_bbox_max[3];
    std::memcpy(left_bbox_min, bbox_min, 3 * sizeof(float));
    std::memcpy(right_bbox_max, bbox_max, 3 * sizeof(float));

    node->left = subdivide(left_ids, depth + 1, left_bbox_min, left_bbox_max);
    node->right = subdivide(right_ids, depth + 1, right_bbox_min, right_bbox_max);

    return node;
}

void KDTreeBuilder::extract_leaves(
    TreeNode* node,
    std::vector<KDTreeBucket>& buckets,
    std::vector<int>& reordered_indices
) {
    if (!node) return;

    if (node->is_leaf()) {
        // Create bucket for this leaf
        KDTreeBucket bucket;
        std::memcpy(bucket.bbox_min, node->bbox_min, 3 * sizeof(float));
        std::memcpy(bucket.bbox_max, node->bbox_max, 3 * sizeof(float));

        int point_offset = reordered_indices.size();
        bucket.point_left = point_offset;
        bucket.point_right = point_offset + node->point_indices.size();
        bucket.max_point_id = -1;      // Will be computed during FPS
        bucket.max_distance = -1.0f;   // Will be computed during FPS

        buckets.push_back(bucket);

        // Append point indices for this bucket
        for (int idx : node->point_indices) {
            reordered_indices.push_back(idx);
        }
    } else {
        // Recurse to children
        extract_leaves(node->left, buckets, reordered_indices);
        extract_leaves(node->right, buckets, reordered_indices);
    }
}

void KDTreeBuilder::compute_bbox(
    const std::vector<int>& point_ids,
    float* bbox_min,
    float* bbox_max
) {
    // Initialize to extremes
    for (int d = 0; d < 3; d++) {
        bbox_min[d] = std::numeric_limits<float>::max();
        bbox_max[d] = std::numeric_limits<float>::lowest();
    }

    // Find min/max in each dimension
    for (int idx : point_ids) {
        for (int d = 0; d < 3; d++) {
            float val = points_[idx].pos[d];
            if (val < bbox_min[d]) bbox_min[d] = val;
            if (val > bbox_max[d]) bbox_max[d] = val;
        }
    }
}

int KDTreeBuilder::find_split_axis(
    const float* bbox_min,
    const float* bbox_max
) {
    // Find axis with longest span
    int longest_axis = 0;
    float longest_span = bbox_max[0] - bbox_min[0];

    for (int d = 1; d < 3; d++) {
        float span = bbox_max[d] - bbox_min[d];
        if (span > longest_span) {
            longest_span = span;
            longest_axis = d;
        }
    }

    return longest_axis;
}

int KDTreeBuilder::partition_points(
    std::vector<int>& point_ids,
    int axis,
    float* left_bbox_max,
    float* right_bbox_min
) {
    int count = point_ids.size();

    // Sort points along axis
    std::sort(point_ids.begin(), point_ids.end(),
        [this, axis](int a, int b) {
            return points_[a].pos[axis] < points_[b].pos[axis];
        }
    );

    // Find median
    int median_idx = count / 2;
    float median_val = points_[point_ids[median_idx]].pos[axis];

    // Update bounding boxes
    left_bbox_max[axis] = median_val;
    right_bbox_min[axis] = median_val;

    return median_idx;
}
