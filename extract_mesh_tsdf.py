import torch
from scene import Scene
import os
import sys
import resource
from os import makedirs
from gaussian_renderer import render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import open3d as o3d
import open3d.core as o3c
import math

def tsdf_fusion(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size,
                voxel_size=0.04, depth_max=20.0, alpha_thres=0.5, block_count=500000):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "tsdf")

    makedirs(render_path, exist_ok=True)
    o3d_device = o3d.core.Device("CPU:0")

    print(f"[TSDF] voxel_size={voxel_size}, depth_max={depth_max}, alpha_thres={alpha_thres}, block_count={block_count}", flush=True)

    vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=block_count,
            device=o3d_device)

    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            try:
                rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]

                depth = rendering[6:7, :, :]
                alpha = rendering[7:8, :, :]
                rgb = rendering[:3, :, :]

                if view.gt_alpha_mask is not None:
                    depth[(view.gt_alpha_mask < 0.5)] = 0

                depth[(alpha < alpha_thres)] = 0

                if idx == 0:
                    valid_depth = depth[depth > 0]
                    print(f"[DEBUG] Render channels: {rendering.shape[0]}, depth range: [{valid_depth.min().item():.4f}, {valid_depth.max().item():.4f}], "
                          f"valid pixels: {valid_depth.numel()}/{depth.numel()}, alpha range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]", flush=True)

                W = view.image_width
                H = view.image_height
                ndc2pix = torch.tensor([
                    [W / 2, 0, 0, (W-1) / 2],
                    [0, H / 2, 0, (H-1) / 2],
                    [0, 0, 0, 1]]).float().cuda().T
                intrins =  (view.projection_matrix @ ndc2pix)[:3,:3].T
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    width=W,
                    height=H,
                    cx = intrins[0,2].item(),
                    cy = intrins[1,2].item(),
                    fx = intrins[0,0].item(),
                    fy = intrins[1,1].item()
                )

                extrinsic = np.asarray((view.world_view_transform.T).cpu().numpy())

                o3d_color = o3d.t.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy(), order="C"))
                o3d_depth = o3d.t.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C"))
                o3d_color = o3d_color.to(o3d_device)
                o3d_depth = o3d_depth.to(o3d_device)

                intrinsic_t = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)
                extrinsic_t = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)

                frustum_block_coords = vbg.compute_unique_block_coordinates(
                    o3d_depth, intrinsic_t, extrinsic_t, 1.0, depth_max)

                vbg.integrate(frustum_block_coords, o3d_depth, o3d_color, intrinsic_t,
                              intrinsic_t, extrinsic_t, 1.0, depth_max)

                # Free GPU tensors explicitly
                del rendering, depth, alpha, rgb
                torch.cuda.empty_cache()

                if (idx + 1) % 50 == 0:
                    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                    print(f"[DEBUG] View {idx+1}/{len(views)} done, peak RSS: {rss_gb:.1f} GB", flush=True)
            except Exception as e:
                print(f"[WARN] View {idx} failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue

        print(f"[DEBUG] Extracting mesh from VBG...", flush=True)
        try:
            t_mesh = vbg.extract_triangle_mesh()
            print(f"[DEBUG] Got tensor mesh, converting to legacy...", flush=True)
            mesh = t_mesh.to_legacy()
            print(f"[DEBUG] Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles", flush=True)
        except Exception as e:
            print(f"[DEBUG] Error during mesh extraction: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # write mesh
        o3d.io.write_triangle_mesh(f"{render_path}/tsdf.ply", mesh)
        print(f"[DEBUG] Wrote mesh to {render_path}/tsdf.ply", flush=True)
            
            
def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams,
                 voxel_size=0.04, depth_max=20.0, block_count=500000):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        train_cameras = scene.getTrainCameras()

        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        cams = train_cameras
        tsdf_fusion(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size,
                    voxel_size=voxel_size, depth_max=depth_max, block_count=block_count)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=0.04, type=float)
    parser.add_argument("--depth_max", default=20.0, type=float)
    parser.add_argument("--block_count", default=500000, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args),
                 voxel_size=args.voxel_size, depth_max=args.depth_max, block_count=args.block_count)