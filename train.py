import copy
import logging
import os
import random
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from diffusion import StableDiffusion
from gaussian_renderer import render, render_opt_pose
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import get_camera_from_tensor

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import lpips
except ImportError:
    lpips = None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_pose(path, quat_pose):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colmap_poses = torch.stack([get_camera_from_tensor(q) for q in quat_pose]).detach().cpu().numpy()
    np.save(path, colmap_poses)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, args):
    seed_everything(dataset.seed)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    guidance_sd = None
    if dataset.lambda_diffusion:
        guidance_sd = StableDiffusion(device=device, args=dataset)
        print("[INFO] loaded SD!")
        logging.info("loaded SD!")

    lpips_criteria = None
    if args.use_lpips_loss:
        lpips_criteria = lpips.LPIPS(net="vgg").to(device)
        logging.info("use lpips loss!")

    id_list = None
    ellipse_list = None
    viewpoint_stack = scene.getTrainCameras().copy()

    noisy_latent_folder = os.path.join(scene.model_path, "pseudo_latents")
    os.makedirs(noisy_latent_folder, exist_ok=True)
    mid_results_path = os.path.join(scene.model_path, "mid_sd_results")
    os.makedirs(mid_results_path, exist_ok=True)
    mid_rendering_path = os.path.join(scene.model_path, "mid_rendering")
    os.makedirs(mid_rendering_path, exist_ok=True)

    easy_pseudo_list = None
    min_pseudo_loss = float("inf")
    min_delta = 1e-3
    counter = 0
    patience = 10
    counter_cams_list = 1

    easy_pseudo14_idx = []
    easy_pseudo6_idx = []
    if args.use_CL_training_strategy:
        logging.info("use_CL_training_strategy!")
        pseudo_cameras = scene.getPseudoCameras()
        num_cameras = len(pseudo_cameras)

        interval = int(num_cameras / (dataset.pseudo_num_interp - 2))
        segment_size = num_cameras // interval

        for rate_cams, index_list in ((14, easy_pseudo14_idx), (6, easy_pseudo6_idx)):
            for i in range(interval):
                segment_start = i * segment_size
                segment_end = segment_start + segment_size
                segment_indices = list(range(segment_start, segment_end))
                easy_view_indices = (
                    segment_indices[: segment_size // rate_cams]
                    + segment_indices[-segment_size // rate_cams :]
                )
                index_list.extend(easy_view_indices)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        if not id_list:
            id_list = list(range(len(viewpoint_stack)))
        img_idx = id_list.pop(randint(0, len(id_list) - 1))
        viewpoint_cam = viewpoint_stack[img_idx]

        pose = gaussians.get_RT(viewpoint_cam.uid)

        pick_diff_cam = (
            randint(1, 10) <= int(dataset.pseudo_view_probility * 10)
            and dataset.lambda_diffusion
            and iteration > dataset.pseudo_view_sd_iteration
        )
        if pick_diff_cam:
            if args.use_CL_training_strategy:
                if counter_cams_list <= 2:
                    try:
                        a = min_pseudo_loss - diffusion_loss.item()
                        if a > min_delta:
                            min_pseudo_loss = diffusion_loss.item()
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patience:
                                counter_cams_list += 1
                                logging.info(
                                    f"Iteration {iteration}/{opt.iterations}, CL strategy enter "
                                    f"next pseudo cams subset {counter_cams_list}"
                                )
                                counter = 0
                                min_pseudo_loss = float("inf")
                        logging.info(
                            f"Iteration {iteration}/{opt.iterations}, "
                            f"min_pseudo_loss - diffusion_loss: {a}"
                        )
                    except Exception:
                        pass

                    if counter_cams_list == 1:
                        if not easy_pseudo_list:
                            easy_pseudo_list = copy.deepcopy(easy_pseudo14_idx)
                        ellipse_idx = easy_pseudo_list.pop(randint(0, len(easy_pseudo_list) - 1))
                    else:
                        if not easy_pseudo_list:
                            easy_pseudo_list = copy.deepcopy(easy_pseudo6_idx)
                        ellipse_idx = easy_pseudo_list.pop(randint(0, len(easy_pseudo_list) - 1))
                else:
                    if not ellipse_list:
                        ellipse_list = list(range(len(scene.getPseudoCameras())))
                    ellipse_idx = ellipse_list.pop(randint(0, len(ellipse_list) - 1))
            else:
                if not ellipse_list:
                    ellipse_list = list(range(len(scene.getPseudoCameras())))
                ellipse_idx = ellipse_list.pop(randint(0, len(ellipse_list) - 1))

            ellipse_cam = scene.getPseudoCameras()[ellipse_idx]
            diff_render_pkg = render(ellipse_cam, gaussians, pipe, background)
            diff_image = diff_render_pkg["render"]

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_opt_pose(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if iteration % 300 == 0 or iteration < dataset.n_sparse + 1:
            torchvision.utils.save_image(
                image, os.path.join(mid_rendering_path, f"output_image_{iteration}.png")
            )
        if (
            dataset.pseudo_view_sd_iteration - dataset.n_sparse - 1
            <= iteration
            < dataset.pseudo_view_sd_iteration
        ):
            torchvision.utils.save_image(
                image, os.path.join(mid_rendering_path, f"output_image_{iteration}.png")
            )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if args.use_lpips_loss:
            loss += lpips_criteria(image, gt_image).mean() * args.lpips_loss_coef

        if pick_diff_cam:
            flag_densify = os.path.exists(
                os.path.join(noisy_latent_folder, f"noisy_latents_{ellipse_idx}.pt")
            )

            img_sd = guidance_sd.train_step(
                diff_image.unsqueeze(0),
                dataset.num_inference_steps,
                novelview_idx=ellipse_idx,
                noisy_latent_folder=noisy_latent_folder,
            )
            img_sd_enhan = img_sd.detach()
            diffusion_loss = (1.0 - opt.lambda_dssim) * l1_loss(
                diff_image.unsqueeze(0), img_sd_enhan
            ) + opt.lambda_dssim * (1.0 - ssim(diff_image.unsqueeze(0), img_sd_enhan))

            if randint(1, 10) <= 1:
                combined_image = torch.cat([diff_image, img_sd_enhan.squeeze(0)], dim=2)
                torchvision.utils.save_image(
                    combined_image, os.path.join(mid_results_path, f"sd_enhan_{iteration}.png")
                )

            loss += dataset.lambda_diffusion * diffusion_loss

            if args.use_lpips_loss:
                loss += lpips_criteria(diff_image.unsqueeze(0), img_sd_enhan).mean() * args.lpips_loss_coef

        loss.backward()

        iter_end.record()
        if iteration % 50 == 0:
            logging.info(f"Iteration {iteration}/{opt.iterations}, Loss: {loss}")

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if pick_diff_cam and tb_writer:
                tb_writer.add_scalar("train_loss_patches/diffusion_loss", diffusion_loss.item(), iteration)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(
                    os.path.join(scene.model_path, "pose", f"pose_{iteration}.npy"),
                    gaussians.P,
                )

            if dataset.use_densify and pick_diff_cam:
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if not flag_densify and iteration > opt.densify_from_iter and randint(1, 10) <= 1:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold
                        )
                        if tb_writer:
                            tb_writer.add_histogram(
                                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
                            )
                            tb_writer.add_scalar(
                                "total_points", scene.gaussians.get_xyz.shape[0], iteration
                            )
                        logging.info(
                            f"Iteration {iteration}/{opt.iterations}, densify! "
                            f"Guassian points:{scene.gaussians.get_xyz.shape[0]}"
                        )

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, f"chkpnt{iteration}.pth"),
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": scene.getTrainCameras()},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and idx < 5:
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 2_500, 5_000, 7_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 2_500, 5_000, 7_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--use_CL_training_strategy", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    log_file_path = os.path.join(
        lp.extract(args).model_path,
        "logs",
        f"({time.strftime('%Y-%m-%d_%H-%M-%S')})_Iteration({op.extract(args).iterations}).log",
    )
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Command line arguments:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    print("\nTraining complete.")
    logging.info("Training complete.")
