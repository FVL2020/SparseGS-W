import logging
import os
import subprocess
import time
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, render_opt_pose
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.pose_utils import get_tensor_from_camera


def make_video(render_path):
    ffmpeg_command = [
        "ffmpeg",
        "-framerate",
        "10",
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(render_path, "*.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "12",
        os.path.join(render_path, "novel_synthesis.mp4"),
    ]
    try:
        subprocess.run(ffmpeg_command, check=False)
    except FileNotFoundError:
        print("ffmpeg not found, skip video generation.")


def render_set_optimize(dataset, name, iteration, scene, gaussians, pipeline, background, args):
    logging.info(f"model path :{args.model_path}")
    logging.info(f"source path :{args.source_path}")

    test_views = scene.getTestCameras().copy()

    model_path = dataset.model_path
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    opt_pose_render = os.path.join(
        os.path.dirname(render_path), f"opt_pose_render{args.optim_test_pose_iter}"
    )
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(opt_pose_render, exist_ok=True)

    opt_poses_pth = os.path.join(
        model_path, name, "ours_{}".format(iteration), f"opt_poses_{args.optim_test_pose_iter}.pt"
    )
    if not os.path.exists(opt_poses_pth):
        opt_poses = []
        logging.info(f"optim test pose iterations :{args.optim_test_pose_iter}")
    else:
        print(f"INFO: already find opt pose of test view from :{opt_poses_pth}, Load it!")
        logging.info(f"Already find opt pose of test view from :{opt_poses_pth}, Load it!")
        opt_poses = torch.load(opt_poses_pth)

    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    for idx, view in enumerate(tqdm(test_views, desc="Post-Process progress")):
        if not os.path.exists(opt_poses_pth):
            num_iter = args.optim_test_pose_iter
            camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
            initial_camera_pose = camera_pose.clone().detach()

            camera_tensor_T = camera_pose[-3:].requires_grad_()
            camera_tensor_q = camera_pose[:4].requires_grad_()
            pose_optimizer = torch.optim.Adam(
                [
                    {"params": [camera_tensor_T], "lr": 0.0003},
                    {"params": [camera_tensor_q], "lr": 0.0001},
                ]
            )

            progress_bar = tqdm(
                range(num_iter), desc=f"Tracking Time Step: {idx}", disable=True
            )

            candidate_q = camera_tensor_q.clone().detach()
            candidate_T = camera_tensor_T.clone().detach()
            current_min_loss = float(1e20)
            gt = view.original_image[0:3, :, :]
            for iteration in range(num_iter):
                rendering = render_opt_pose(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]),
                )["render"]
                loss = torch.abs(gt - rendering).mean()
                if iteration % 10 == 0:
                    print(iteration, loss.item())
                    logging.info(
                        f"Image Name: {view.image_name} Iteration {iteration}/{num_iter}, "
                        f"Loss :{loss.item()}"
                    )
                loss.backward()

                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_q = camera_tensor_q.clone().detach()
                        candidate_T = camera_tensor_T.clone().detach()

                    progress_bar.update(1)

            camera_tensor_q = candidate_q
            camera_tensor_T = candidate_T
            progress_bar.close()
            opt_pose = torch.cat([camera_tensor_q, camera_tensor_T])
            print(opt_pose - initial_camera_pose)
            opt_poses.append(opt_pose)
        else:
            opt_pose = opt_poses[idx]
            gt = view.original_image[0:3, :, :]

        rendering_opt = render_opt_pose(
            view, gaussians, pipeline, background, camera_pose=opt_pose
        )["render"]
        torchvision.utils.save_image(
            rendering_opt, os.path.join(opt_pose_render, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))

    if not os.path.exists(opt_poses_pth):
        torch.save(opt_poses, opt_poses_pth)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
        if getattr(view, "original_image", None) is not None:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))


def render_sets(args, dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not args.skip_novel:
            render_set(
                dataset.model_path,
                "novel",
                scene.loaded_iter,
                scene.getNovelCameras(),
                gaussians,
                pipeline,
                background,
            )
            make_video(
                os.path.join(dataset.model_path, "novel", "ours_{}".format(scene.loaded_iter), "renders")
            )

        if not args.skip_pseudo:
            render_set(
                dataset.model_path,
                "pseudo_view",
                scene.loaded_iter,
                scene.getPseudoCameras(),
                gaussians,
                pipeline,
                background,
            )
            make_video(
                os.path.join(
                    dataset.model_path, "pseudo_view", "ours_{}".format(scene.loaded_iter), "renders"
                )
            )

    if not skip_test:
        render_set_optimize(dataset, "test", scene.loaded_iter, scene, gaussians, pipeline, background, args)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_novel", action="store_true")
    parser.add_argument("--skip_pseudo", action="store_true")
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    log_file_path = os.path.join(
        model.extract(args).model_path,
        "logs",
        f"({time.strftime('%Y-%m-%d_%H-%M-%S')})_render.log",
    )
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    safe_state(args.quiet)

    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test)
