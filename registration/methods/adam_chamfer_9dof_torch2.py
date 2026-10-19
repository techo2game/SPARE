"""Adam + Chamfer with 9 DOF (6D rotation + 3D translation) - Pure PyTorch."""
import torch


def compute_distance_oneside_trimmed(test, train, keep_ratio=0.5):
    """One-sided distance keeping only best matches (discard worst 50%).

    Robust to outliers and partial overlap.

    Args:
        test: (N, 3) test points
        train: (M, 3) train points
        keep_ratio: Ratio of best matches to keep (default 0.5 = 50%)

    Returns:
        mean distance of best matches only
    """
    from pytorch3d.ops import knn_points

    # Find k=1 nearest neighbor
    test_batch = test.unsqueeze(0)  # (1, N, 3)
    train_batch = train.unsqueeze(0)  # (1, M, 3)

    knn_result = knn_points(test_batch, train_batch, K=1)
    dists = knn_result.dists.squeeze().sqrt()  # (N,) actual distances

    # Keep only bottom 50% (best matches)
    num_keep = max(1, int(test.shape[0] * keep_ratio))
    best_dists, _ = torch.topk(dists, num_keep, largest=False)

    return best_dists.mean()

def compute_distance_oneside2(test, train):
    """Fast one-sided distance using PyTorch3D knn_points."""
    from pytorch3d.ops import knn_points

    # knn_points expects batched input: (B, N, 3)
    test_batch = test.unsqueeze(0)  # (1, N, 3)
    train_batch = train.unsqueeze(0)  # (1, M, 3)

    # Find k=1 nearest neighbor
    knn_result = knn_points(test_batch, train_batch, K=1)

    # knn_result.dists is (1, N, 1) - squared distances
    dists = knn_result.dists.squeeze()  # (N,)

    return dists.sqrt().mean()  # sqrt to get actual distance


def align_to_pca_axes(points):
    """Align point cloud so largest extent is along X-axis using PCA.

    Args:
        points: (N, 3) tensor on GPU

    Returns:
        aligned_points: (N, 3) tensor on GPU, PCA-aligned
        transform: (4, 4) transformation matrix applied
    """
    # Center points
    centroid = points.mean(dim=0, keepdim=True)
    centered = points - centroid

    # Compute covariance matrix
    cov = (centered.T @ centered) / (points.shape[0] - 1)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort by descending eigenvalues (largest variance first)
    idx = torch.argsort(eigenvalues, descending=True)
    pca_axes = eigenvectors[:, idx]  # (3, 3)

    # Align to standard axes: largest variance → X, second → Y, third → Z
    # R transforms from PCA space to standard space
    R = pca_axes.T  # Transpose to get rotation matrix

    # Apply rotation to centered points
    aligned = centered @ R.T

    # Build transformation matrix
    transform = torch.eye(4, device=points.device)
    transform[:3, :3] = R
    transform[:3, 3] = -centroid @ R.T

    return aligned, transform

def compute_chamfer(points1, points2):
    """Compute symmetric Chamfer distance (batched for GPU)."""
    dist_matrix = torch.cdist(points1, points2, p=2)
    dist_fwd = dist_matrix.min(dim=1)[0]
    dist_bwd = dist_matrix.min(dim=0)[0]
    return dist_fwd.mean() + dist_bwd.mean()

def compute_distance_oneside(test, train):
    """Compute simple mean squared distance between two point sets."""
    dist_matrix = torch.cdist(test, train, p=2)
    dist_fwd = dist_matrix.min(dim=1)[0]
    return dist_fwd.mean()

def adam_chamfer_9dof_torch(test, train, max_iter=200, num_points=2000, lr=0.2,
                      use_scheduler=True, num_restarts=24, patience=20):
    """Global registration using Adam optimizer with 9 DOF optimization.

    Optimizes both rotation (6D representation) and translation (3D).
    Does NOT assume centroids should align.

    Args:
        test: Test point cloud torch tensor (N, 3) on GPU
        train: Train point cloud torch tensor (M, 3) on GPU
        max_iter: Maximum iterations per restart
        num_points: Number of points to sample
        lr: Initial learning rate
        use_scheduler: Use cosine annealing scheduler
        num_restarts: Number of random restarts
        patience: Early stopping patience (default: 20)

    Returns:
        transformation: 4x4 transformation matrix (numpy) for global alignment
    """

    device = test.device


    def rotation_6d_to_matrix(d6):
        """Convert 6D representation to rotation matrix."""
        a1 = d6[:3]
        a2 = d6[3:]

        b1 = a1 / torch.norm(a1)
        b2 = a2 - (a2 @ b1) * b1
        b2 = b2 / torch.norm(b2)
        b3 = torch.linalg.cross(b1, b2)

        R = torch.stack([b1, b2, b3], dim=1)
        return R

    def fibonacci_sphere_rotation(i, n):
        """Generate rotation using Fibonacci sphere sampling - Pure PyTorch."""
        phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0, device=device)))
        y = torch.tensor(1 - (i / float(n - 1)) * 2, device=device)
        radius = torch.sqrt(1 - y * y)

        theta = phi * i
        x = torch.cos(theta) * radius
        z = torch.sin(theta) * radius

        axis = torch.stack([x, y, z])
        axis = axis / torch.norm(axis)

        angle = torch.tensor((i * 2.0 * torch.pi / n) if i > 0 else 0.0, device=device)

        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=device), -axis[2], axis[1]]),
            torch.stack([axis[2], torch.tensor(0.0, device=device), -axis[0]]),
            torch.stack([-axis[1], axis[0], torch.tensor(0.0, device=device)])
        ])
        R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    # Subsample test points
    if test.size(0) > num_points:
        idx = torch.randperm(test.size(0), device=device)[:num_points]
        test_sampled = test[idx]
    else:
        test_sampled = test

    # Subsample train points
    if train.size(0) > num_points:
        idx = torch.randperm(train.size(0), device=device)[:num_points]
        train_sampled = train[idx]
    else:
        train_sampled = train


    best_loss = float('inf')
    best_transform = None

    for restart in range(num_restarts):
        # Initialize rotation (6D)
        if restart == 0:
            R_init = torch.eye(3, device=device)
        else:
            R_init = fibonacci_sphere_rotation(restart, num_restarts)

        rot_6d = R_init[:, :2].flatten().clone()
        rot_6d.requires_grad = True

        # Initialize translation (3D) - start at zero
        trans = torch.zeros(3, device=device, requires_grad=True)

        # Optimize both rotation and translation
        optimizer = torch.optim.Adam([rot_6d, trans], lr=lr)

        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)

        for iteration in range(max_iter):
            # Convert 6D to rotation matrix
            R = rotation_6d_to_matrix(rot_6d)

            # Apply rotation + translation
            test_transformed = (test_sampled @ R.T) + trans

            # Compute Chamfer loss
            #loss = compute_chamfer(test_transformed, train_sampled)
            loss = compute_distance_oneside_trimmed(train_sampled, test_transformed,0.5) + compute_distance_oneside2(test_sampled, train_sampled)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_scheduler:
                scheduler.step()

        final_loss = loss.item()

        if final_loss < best_loss:
            best_loss = final_loss
            # Build transformation matrix on GPU
            with torch.no_grad():
                R_best = rotation_6d_to_matrix(rot_6d.detach())
                t_best = trans.detach()
                transform_gpu = torch.eye(4, device=device)
                transform_gpu[:3, :3] = R_best
                transform_gpu[:3, 3] = t_best
                best_transform = transform_gpu.clone()

    # Convert to numpy only at the very end
    transformation = best_transform

    return transformation