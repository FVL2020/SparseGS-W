import json
import logging
import os
import random

import numpy as np

from arguments import ModelParams
from scene.cameras import PseudoCamera
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import (
    cameraList_from_camInfos,
    camera_to_JSON,
    novelCameraList_from_camInfos,
)
from utils.my_utils import interpolate_camera_poses
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.novel_cameras = {}
        self.pseudo_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "crop_images")):
            print("Found crop_images, assuming Nerf-W data set!")
            scene_info = sceneLoadTypeCallbacks["NerfW"](
                args.source_path,
                args.images,
                args.eval,
                N_sparse=args.n_sparse,
                num_inter=args.novel_num_interp,
                dataset_name=args.dataset_name,
                arg=args,
            )
        else:
            raise AssertionError("Could not recognize scene type!")

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
            print("Loading Novel Cameras")
            self.novel_cameras[resolution_scale] = novelCameraList_from_camInfos(
                scene_info.novel_cameras, resolution_scale, args
            )

            pseudo_cams = []
            new_poses = []
            idx = list(range(len(self.train_cameras[resolution_scale])))
            K_list = [
                (
                    np.vstack(
                        [np.hstack((camera.R, camera.T.reshape(-1, 1))), np.array([0, 0, 0, 1])]
                    ),
                    camera.FoVx,
                    camera.FoVy,
                )
                for camera in self.train_cameras[resolution_scale]
            ]
            cam_extrinsics_sorted = sorted(K_list, key=lambda item: item[0][:3, 3][0])

            for i in range(0, len(idx) - 1):
                current_two = idx[i : i + 2]
                novel_poses_list = interpolate_camera_poses(
                    cam_extrinsics_sorted[current_two[0]][0],
                    cam_extrinsics_sorted[current_two[1]][0],
                    num_interpolations=args.pseudo_num_interp,
                    random_interpolate=False,
                )
                novel_poses_list.pop(0)
                new_poses.append(np.array(novel_poses_list))
            new_poses = np.concatenate(new_poses, axis=0)

            ave_image_height = int(
                np.mean([camera.image_height for camera in self.train_cameras[resolution_scale]])
            )
            ave_image_width = int(
                np.mean([camera.image_width for camera in self.train_cameras[resolution_scale]])
            )
            for i, pose in enumerate(new_poses):
                p0 = np.array(cam_extrinsics_sorted[i // (args.pseudo_num_interp - 2)][1:])
                p1 = np.array(cam_extrinsics_sorted[i // (args.pseudo_num_interp - 2) + 1][1:])
                alpha = (i % (args.pseudo_num_interp - 2) + 1) / (args.pseudo_num_interp - 1)
                focal = p0 + (p1 - p0) * alpha
                pseudo_cams.append(
                    PseudoCamera(
                        R=pose[:3, :3],
                        T=pose[:3, 3],
                        FoVx=focal[0],
                        FoVy=focal[1],
                        width=ave_image_width,
                        height=ave_image_height,
                    )
                )
            self.pseudo_cameras[resolution_scale] = pseudo_cams
            print(f"Create {len(pseudo_cams)} pseudo cams for train!")
            logging.info(f"Create {len(pseudo_cams)} pseudo cams for train!")

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.init_RT_seq(self.train_cameras)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getNovelCameras(self, scale=1.0):
        return self.novel_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        return self.pseudo_cameras[scale] if self.pseudo_cameras else [None]
