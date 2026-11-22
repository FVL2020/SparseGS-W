import json
import os
import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.my_utils import interpolate_camera_poses, interpolate_camera_poses_qvec
from utils.sh_utils import SH2RGB

from natsort import natsorted


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class CameraInfoWithDepth(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    novel_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        return center.flatten(), np.max(dist)

    cam_centers = []
    for cam in cam_info:
        C2W = np.linalg.inv(getWorld2View2(cam.R, cam.T))
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    return {"translate": -center, "radius": diagonal * 1.1}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            raise AssertionError(
                "Colmap camera model not handled: only undistorted datasets "
                "(PINHOLE or SIMPLE_PINHOLE cameras) support!"
            )

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        if not os.path.exists(image_path):
            if image_path.split(".")[-1] == "jpg":
                image_path = image_path.replace(".jpg", ".png")
            if image_path.split(".")[-1] == "JPG":
                image_path = image_path.replace(".JPG", ".png")
        if not os.path.exists(image_path):
            prefixed = os.path.join(images_folder, "image_" + os.path.basename(image_path))
            if os.path.exists(prefixed):
                image_path = prefixed
        image = Image.open(image_path)

        cam_infos.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
            )
        )
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0/images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0/cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0/images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0/cameras.txt"))

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for c in cam_infos if "image" in c.image_name]
        test_cam_infos = [c for c in cam_infos if "image" not in c.image_name]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(os.path.join(path, "sparse/0/points3D.bin"))
        except Exception:
            xyz, rgb, _ = read_points3D_text(os.path.join(path, "sparse/0/points3D.txt"))
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        novel_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        for idx, frame in enumerate(contents["frames"]):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            FovY = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )
    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        novel_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


def create_new_infos_tanks(idx, cam_extrinsics, cam_intrinsics, num_inter=10, random_interpolate=False):
    new_qvec = []
    new_tvec = []
    cam_infos = []

    sorted_with_indices = sorted(cam_extrinsics.items(), key=lambda x: x[1].tvec[0])
    cam_extrinsics_sorted = [item[1] for item in sorted_with_indices]

    for i in range(0, len(idx) - 1):
        current_two = idx[i : i + 2]
        novel_qvec, novel_tvec = interpolate_camera_poses_qvec(
            cam_extrinsics_sorted[current_two[0]].qvec,
            cam_extrinsics_sorted[current_two[0]].tvec,
            cam_extrinsics_sorted[current_two[1]].qvec,
            cam_extrinsics_sorted[current_two[1]].tvec,
            num_interpolations=num_inter,
            random_interpolate=random_interpolate,
        )
        novel_qvec.pop(0)
        novel_qvec.pop(-1)
        novel_tvec.pop(0)
        novel_tvec.pop(-1)
        new_qvec.append(novel_qvec)
        new_tvec.append(novel_tvec)
    new_qvec = np.concatenate(new_qvec, axis=0)
    new_tvec = np.concatenate(new_tvec, axis=0)

    width = cam_intrinsics[1].width
    height = cam_intrinsics[1].height
    focal_length_x = cam_intrinsics[1].params[0]
    focal_length_y = cam_intrinsics[1].params[1]

    for index, qvec in enumerate(new_qvec):
        R = np.transpose(qvec2rotmat(qvec))
        T = np.array(new_tvec[index])
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_y, width)
        cam_infos.append(
            CameraInfoWithDepth(
                uid=index,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=None,
                image_path=None,
                image_name=None,
                width=width,
                height=height,
                depth=None,
            )
        )
    return cam_infos


def create_new_infos_original(idx, cam_extrinsics, cam_intrinsics, num_inter=10, random_interpolate=False):
    new_poses = []
    cam_infos = []

    sorted_with_indices = sorted(enumerate(cam_extrinsics), key=lambda x: x[1][1][:3, 3][0])
    cam_extrinsics_sorted = [item[1] for item in sorted_with_indices]
    sorted_indices = [item[0] for item in sorted_with_indices]
    cam_intrinsics_sorted = [cam_intrinsics[i] for i in sorted_indices]

    for i in range(0, len(idx) - 1):
        current_two = idx[i : i + 2]
        novel_poses_list = interpolate_camera_poses(
            cam_extrinsics_sorted[current_two[0]][1],
            cam_extrinsics_sorted[current_two[1]][1],
            num_interpolations=num_inter,
            random_interpolate=random_interpolate,
        )
        if i == len(idx) - 2:
            novel_poses_list.append(cam_extrinsics_sorted[current_two[1]][1])
        new_poses.append(np.array(novel_poses_list))
    new_poses = np.concatenate(new_poses, axis=0)

    width = int(np.mean(cam_intrinsics, axis=0)[1]) * 2
    height = int(np.mean(cam_intrinsics, axis=0)[2]) * 2
    for index, new_pose in enumerate(new_poses):
        w2c = np.linalg.inv(new_pose)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        if index % (num_inter - 1) == 0:
            focal = cam_intrinsics_sorted[index // (num_inter - 1)][0]
        else:
            p0 = cam_intrinsics_sorted[index // (num_inter - 1)][0]
            p1 = cam_intrinsics_sorted[index // (num_inter - 1) + 1][0]
            alpha = (index % (num_inter - 1)) / (num_inter - 1)
            focal = p0 + (p1 - p0) * alpha
        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)

        cam_infos.append(
            CameraInfoWithDepth(
                uid=index,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=None,
                image_path=None,
                image_name=None,
                width=width,
                height=height,
                depth=None,
            )
        )
    return cam_infos


def readNerfWCameras(cam_extrinsics, cam_intrinsics, root_folder):
    cam_infos = []
    images_folder = os.path.join(root_folder, "crop_images")
    for root, _, files in os.walk(images_folder):
        if files[0].find("image") != -1:
            try:
                files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            except Exception:
                files = natsorted(files)
        else:
            files.sort()

        for index, key in enumerate(cam_extrinsics):
            image_name = key[0]
            image_path = os.path.join(root, "image_" + image_name + ".png")
            image = Image.open(image_path)

            dp_folder = os.path.join(root_folder, "depth")
            if os.path.exists(dp_folder):
                dp = np.load(os.path.join(dp_folder, "depth_" + image_name + ".npy"))
            else:
                dp = None

            w2c = np.linalg.inv(key[1])
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            focal = cam_intrinsics[index][0]
            FovY = focal2fov(focal, image.height)
            FovX = focal2fov(focal, image.width)
            cam_infos.append(
                CameraInfoWithDepth(
                    uid=index,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.width,
                    height=image.height,
                    depth=dp,
                )
            )
    return cam_infos


def readNerfWSceneInfo(path, images, eval, tanks=10, N_sparse=-1, num_inter=10, dataset_name=None, arg=None):
    if dataset_name == "Tanks":
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "train_intrinsic.txt"))
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "train_c2w.txt"))
        cam_infos = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, "crop_images"),
        )
    else:
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "train_intrinsic.txt"))
        params_list = [camera.params for camera in cam_intrinsics.values()]
        cam_intrinsics = np.vstack(params_list)[:, 1:]

        with open(os.path.join(path, "train_c2w.pkl"), "rb") as f:
            cam_extrinsics = pickle.load(f)
        cam_infos = readNerfWCameras(
            cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, root_folder=path
        )

    if N_sparse != len(cam_infos):
        raise ValueError("N_sparse != len(training view)")

    cam_extrinsics_all = read_extrinsics_text(os.path.join(path, "test_c2w.txt"))
    cam_intrinsics_all = read_intrinsics_text(os.path.join(path, "test_intrinsic.txt"))
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics_all,
        cam_intrinsics=cam_intrinsics_all,
        images_folder=os.path.join(path, "test"),
    )
    cam_infos_test = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        if N_sparse > 0:
            train_cam_infos = cam_infos[0:N_sparse]
            idx_train = list(range(len(train_cam_infos)))
            if dataset_name == "Tanks":
                novel_cam_info = create_new_infos_tanks(
                    idx_train,
                    cam_extrinsics=cam_extrinsics,
                    cam_intrinsics=cam_intrinsics,
                    num_inter=num_inter,
                    random_interpolate=False,
                )
            elif dataset_name == "NerfW":
                novel_cam_info = create_new_infos_original(
                    idx_train,
                    cam_extrinsics=cam_extrinsics,
                    cam_intrinsics=cam_intrinsics,
                    num_inter=num_inter,
                    random_interpolate=False,
                )
            else:
                novel_cam_info = []
            test_cam_infos = cam_infos_test
        else:
            train_cam_infos = cam_infos
            test_cam_infos = cam_infos_test
            novel_cam_info = []
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        novel_cam_info = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if getattr(arg, "dust3r_pts", 1):
        candidates = [
            "dust3r_pc_with_sky_add_sky_my.ply",
            "dust3r_pc_with_sky_add_sky.ply",
            "dust3r_pc_with_sky.ply",
        ]
        ply_path = next(
            (os.path.join(path, name) for name in candidates if os.path.exists(os.path.join(path, name))),
            os.path.join(path, candidates[-1]),
        )
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    print(f"INFO: load initial point cloud: {ply_path}")

    pcd = None
    try:
        pcd = fetchPly(ply_path)
        if getattr(arg, "random_pts", 0):
            from utils.my_utils import generate_random_pcd

            pcd = generate_random_pcd(pcd, num_samples=10000)
            print("INFO: generate random point cloud")
    except Exception:
        pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        novel_cameras=novel_cam_info,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "NerfW": readNerfWSceneInfo,
}
