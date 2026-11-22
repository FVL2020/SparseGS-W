import numpy as np

from utils.graphics_utils import BasicPointCloud


def slerp(p0, p1, t):
    if np.all(p0 == 0) or np.all(p1 == 0):
        return p0 + (p1 - p0) * t

    p0_norm = p0 / np.linalg.norm(p0)
    p1_norm = p1 / np.linalg.norm(p1)

    dot_product = np.dot(p0_norm, p1_norm)
    if dot_product < 0.0:
        p1_norm = -p1_norm
        dot_product = -dot_product
    if dot_product > 1.0 or dot_product < -1.0:
        raise ValueError("dot product out of range [-1, 1]")

    omega = np.arccos(dot_product)
    so = np.sin(omega)
    if so < 1e-6:
        return p0 + (p1 - p0) * t

    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def interpolate_camera_poses(cam_pose1, cam_pose2, num_interpolations, random_interpolate=False):
    R1 = cam_pose1[:3, :3]
    R2 = cam_pose2[:3, :3]
    t1 = cam_pose1[:3, 3]
    t2 = cam_pose2[:3, 3]

    if random_interpolate:
        factors = np.sort(np.random.rand(num_interpolations))
    else:
        factors = np.linspace(0, 1, num_interpolations)

    interpolated_translations = [slerp(t1, t2, t) for t in factors]
    interpolated_rotations = [(R1 + (R2 - R1) * t) for t in factors]

    for i in range(len(interpolated_rotations)):
        u, _, vh = np.linalg.svd(interpolated_rotations[i])
        interpolated_rotations[i] = np.dot(u, vh)

    interpolated_poses = [np.eye(4, dtype=np.float32) for _ in range(num_interpolations)]
    for i in range(num_interpolations):
        interpolated_poses[i][:3, :3] = interpolated_rotations[i]
        interpolated_poses[i][:3, 3] = interpolated_translations[i]

    interpolated_poses.pop()
    return interpolated_poses


def interpolate_camera_poses_qvec(q1, t1, q2, t2, num_interpolations, random_interpolate=False):
    if random_interpolate:
        factors = np.sort(np.random.rand(num_interpolations))
    else:
        factors = np.linspace(0, 1, num_interpolations)

    interpolated_translations = [slerp(t1, t2, t) for t in factors]
    interpolated_rotations = [(q1 + (q2 - q1) * t) for t in np.linspace(0, 1, num_interpolations)]
    return interpolated_rotations, interpolated_translations


def generate_random_pcd(pcd, num_samples=10000):
    positions = pcd.points

    min_vals = np.min(positions, axis=0)
    max_vals = np.max(positions, axis=0)
    center = (min_vals + max_vals) / 2
    half_length = np.max(max_vals - min_vals) / 2

    random_points = np.random.uniform(
        low=center - half_length,
        high=center + half_length,
        size=(num_samples, 3),
    )
    random_colors = np.random.rand(num_samples, 3)
    random_normals = np.random.randn(num_samples, 3)
    random_normals /= np.linalg.norm(random_normals, axis=1, keepdims=True)

    return BasicPointCloud(points=random_points, colors=random_colors, normals=random_normals)
