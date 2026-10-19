"""Adam + Chamfer with 9 DOF (6D rotation + 3D translation)."""
import numpy as np
import torch
import torch.nn.functional as F


def adam_chamfer_9dof(test, train, max_iter=200, num_points=4000, lr=0.01,
                      use_scheduler=True, num_restarts=1):
    """Global registration using Adam optimizer with 9 DOF optimization.

    Optimizes both rotation (6D representation) and translation (3D).
    Does NOT assume centroids should align.

    Args:
        test: Full test point cloud (Open3D PointCloud)
        train: Full train point cloud (Open3D PointCloud)
        max_iter: Maximum iterations per restart
        num_points: Number of points to sample
        lr: Initial learning rate
        use_scheduler: Use cosine annealing scheduler
        num_restarts: Number of random restarts

    Returns:
        transformation: 4x4 transformation matrix for global alignment
    """
    from sklearn.neighbors import NearestNeighbors
    lr=0.2
    num_restarts = 24
    max_iter = 200
    num_points = 2000

    # Get point arrays (NO centering - optimize translation directly)
    test_points = np.asarray(test.points, dtype=np.float32)
    train_points = np.asarray(train.points, dtype=np.float32)

    # Subsample
    if len(test_points) > num_points:
        test_sub = test_points[np.random.choice(len(test_points), num_points, replace=False)]
    else:
        test_sub = test_points

    if len(train_points) > num_points:
        train_sub = train_points[np.random.choice(len(train_points), num_points, replace=False)]
    else:
        train_sub = train_points

    # Convert to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_tensor = torch.from_numpy(test_sub).float().to(device)
    train_tensor = torch.from_numpy(train_sub).float().to(device)

    def compute_chamfer(points1, points2):
        """Compute symmetric Chamfer distance (batched for GPU)."""
        dist_matrix = torch.cdist(points1, points2, p=2)
        dist_fwd = dist_matrix.min(dim=1)[0]
        dist_bwd = dist_matrix.min(dim=0)[0]
        return dist_fwd.mean() + dist_bwd.mean()

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
        """Generate rotation using Fibonacci sphere sampling."""
        phi = np.pi * (3.0 - np.sqrt(5.0))
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y * y)

        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        axis = np.array([x, y, z])
        axis = axis / np.linalg.norm(axis)

        angle = (i * 2.0 * np.pi / n) if i > 0 else 0

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    best_loss = float('inf')
    best_rot_params = None
    best_trans_params = None

    for restart in range(num_restarts):
        # Initialize rotation (6D)
        if restart == 0:
            R_init = np.eye(3)
        else:
            R_init = fibonacci_sphere_rotation(restart, num_restarts)

        rot_6d = torch.from_numpy(R_init[:, :2].flatten().astype(np.float32)).to(device)
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
            test_transformed = (test_tensor @ R.T) + trans

            # Compute Chamfer loss
            loss = compute_chamfer(test_transformed, train_tensor)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_scheduler:
                scheduler.step()

        final_loss = loss.item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_rot_params = rot_6d.detach().clone()
            best_trans_params = trans.detach().clone()

    # Build transformation from best parameters
    with torch.no_grad():
        R_optimal = rotation_6d_to_matrix(best_rot_params).cpu().numpy()
        t_optimal = best_trans_params.cpu().numpy()

    transformation = np.eye(4)
    transformation[:3, :3] = R_optimal
    transformation[:3, 3] = t_optimal

    # Apply to full test cloud
    test.transform(transformation)

    # Evaluate
    test_aligned = np.asarray(test.points)
    train_full = np.asarray(train.points)

    nbrs = NearestNeighbors(n_neighbors=1).fit(train_full)
    distances, _ = nbrs.kneighbors(test_aligned)
    rmse = np.sqrt(np.mean(distances**2))

    return transformation