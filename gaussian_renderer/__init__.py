import math

import torch

# https://github.com/slothfulxtx/diff-gaussian-rasterization
from diff_gauss import GaussianRasterizationSettings
from diff_gauss import GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.pose_utils import get_camera_from_tensor, quadmultiply
from utils.sh_utils import eval_sh

# https://github.com/NVlabs/InstantSplat/tree/main/submodules
from diff_gs_instantsplat import GaussianRasterizationSettings as PoseRasterizationSettings
from diff_gs_instantsplat import GaussianRasterizer as PoseRasterizer


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None):
    screenspace_points = (
        torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, rendered_depth, _, rendered_alpha, radii, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
    )

    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def render_opt_pose(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                    scaling_modifier=1.0, override_color=None, camera_pose=None):
    screenspace_points = (
        torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    w2c = torch.eye(4).cuda()
    projmatrix = w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0)).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
    raster_settings = PoseRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = PoseRasterizer(raster_settings=raster_settings)

    rel_w2c = get_camera_from_tensor(camera_pose)
    gaussians_xyz = pc._xyz.clone()
    gaussians_rot = pc._rotation.clone()

    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    means3D = (rel_w2c @ xyz_homo.T).T[:, :3]
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = quadmultiply(camera_pose[:4], gaussians_rot)

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
