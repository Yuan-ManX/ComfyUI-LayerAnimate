import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist
import os
from einops import rearrange
import imageio
import torchvision
from PIL import Image
import io
from matplotlib import pyplot as plt


RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6

COLORWHEEL = torch.zeros((RY + YG + GC + CB + BM + MR, 3))
col = 0

# RY
COLORWHEEL[0:RY, 0] = 255
COLORWHEEL[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
col = col + RY
# YG
COLORWHEEL[col:col + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
COLORWHEEL[col:col + YG, 1] = 255
col = col + YG
# GC
COLORWHEEL[col:col + GC, 1] = 255
COLORWHEEL[col:col + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
col = col + GC
# CB
COLORWHEEL[col:col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
COLORWHEEL[col:col + CB, 2] = 255
col = col + CB
# BM
COLORWHEEL[col:col + BM, 2] = 255
COLORWHEEL[col:col + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
col = col + BM
# MR
COLORWHEEL[col:col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
COLORWHEEL[col:col + MR, 0] = 255


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """
    name: full name of source para
    para_list: partial name of target para
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_images_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    os.makedirs(path, exist_ok=True)
    for time_idx, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        image = Image.fromarray(x)
        image.save(os.path.join(path, f"{time_idx:04d}.png"))

def save_image_with_mask(image: torch.Tensor, masks: torch.Tensor, path: str, rescale=False, alpha=0.6):
    # image: [C, H, W], mask: [N, H, W]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = rearrange(image, "c h w -> h w c")
    if rescale:
        image = (image + 1.0) / 2.0 # -1,1 -> 0,1
    image = (image * 255).numpy().astype(np.uint8)
    final_image = Image.fromarray(image).convert("RGBA")
    cmap = plt.get_cmap("tab20c")
    masks = masks.cpu().numpy().astype(np.float32)
    for i, img in enumerate(masks):
        mask_color = np.array([*cmap(i * 4 + 2)[:3], alpha])
        mask = img[:,:,None] * mask_color[None,None,:] * 255
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert("RGBA")
        final_image = Image.alpha_composite(final_image, mask)
    final_image.save(path)

def save_videos_with_heatmap(videos: torch.Tensor, trajectory: torch.Tensor, path: str, n_rows=6, fps=8):
    # use Image RGBA and alpha_composite to combine video and trajectory
    # use imageio to save video
    videos = rearrange(videos, "b c t h w -> t b c h w")
    trajectory = rearrange(trajectory, "b c t h w -> t b c h w")
    outputs = []
    for x, y in zip(videos, trajectory):
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x = (x * 255).numpy().astype(np.uint8)
        y = torchvision.utils.make_grid(y, nrow=6)
        y = y.transpose(0, 1).transpose(1, 2).squeeze(-1)
        y = torch.cat([y, torch.mean(y, dim=-1, keepdim=True)], dim=-1)
        y = (y * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x).convert("RGBA")
        y = Image.fromarray(y)
        x = Image.alpha_composite(x, y)
        outputs.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_videos_with_traj(videos: torch.Tensor, trajectory: torch.Tensor, path: str, rescale=False, fps=8, line_width=3, circle_radius=5):
    # videos: [C, F, H, W]
    # trajectory: [F, N, 2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    videos = rearrange(videos, "c f h w -> f h w c")
    if rescale:
        videos = (videos + 1) / 2
    videos = (videos * 255).numpy().astype(np.uint8)
    outputs = []
    for frame_idx, img in enumerate(videos):
        # img: [H, W, C], traj: [N, 2]
        # draw trajectory use cv2.line
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for traj_idx in range(trajectory.shape[1]):
            for history_idx in range(frame_idx):
                cv2.line(img, tuple(trajectory[history_idx, traj_idx].int().tolist()), tuple(trajectory[history_idx+1, traj_idx].int().tolist()), (0, 0, 255), line_width)
            cv2.circle(img, tuple(trajectory[frame_idx, traj_idx].int().tolist()), circle_radius, (100, 230, 160), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs.append(img)
    imageio.mimsave(path, outputs, fps=fps)

def save_layer_prompts_video(videos, layer_masks, motion_scores, flow_maps, path, alpha=0.6, fps=8, flow_step=10, flow_scale=1.0):
    # videos: [F, C, H, W]
    # layer_masks: [N, F, H, W]
    # motion_scores: [N, ]
    # flow_maps: [F, 2, H, W]
    frame_length = videos.shape[0]
    h, w = videos.shape[-2:]
    n_keyframes = layer_masks.shape[1]
    if n_keyframes == 1:
        keyframe_indices = [0]
    elif n_keyframes == 2:
        keyframe_indices = [0, frame_length - 1]
    else:
        keyframe_indices = list(range(n_keyframes))
    videos = rearrange(videos, "t c h w -> t h w c")
    videos = ((videos + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
    layer_masks = layer_masks.numpy()
    flow_maps = flow_maps.float().numpy()
    frame_list = []
    cmap = plt.get_cmap("tab10")
    for frame_idx in range(frame_length):
        output_frame = Image.new("RGBA", (w * 2, h * 2))
        frame = Image.fromarray(videos[frame_idx]).convert("RGBA")
        frame_mask = None
        output_frame.paste(frame, (0, 0))
        for layer_idx, layer_mask in enumerate(layer_masks):
            if frame_idx in keyframe_indices:
                layer_color = (np.array([*cmap(layer_idx)[:3], alpha]) * 255).astype(np.uint8)
                if frame_idx == frame_length - 1:
                    mask_with_color = Image.fromarray(layer_mask[-1, :, :, np.newaxis] * layer_color[np.newaxis, np.newaxis, :])
                else:
                    mask_with_color = Image.fromarray(layer_mask[frame_idx, :, :, np.newaxis] * layer_color[np.newaxis, np.newaxis, :])
            else:
                mask_with_color = Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8))
            frame = Image.alpha_composite(frame, mask_with_color)
            frame_mask = Image.alpha_composite(frame_mask, mask_with_color) if frame_mask is not None else mask_with_color
        output_frame.paste(frame, (w, 0))
        output_frame.paste(frame_mask, (0, h))
        flow_x = flow_maps[frame_idx, 0] * flow_scale
        flow_y = flow_maps[frame_idx, 1] * flow_scale
        x, y = np.arange(0, w, step=flow_step), np.arange(0, h, step=flow_step)
        X, Y = np.meshgrid(x, y)
        U, V = flow_x[::flow_step, ::flow_step], flow_y[::flow_step, ::flow_step]
        plt.figure()
        plt.gca().set_facecolor('white')
        plt.quiver(X, Y, U, V, color='black', angles='xy', scale_units='xy', scale=1)
        plt.xlim(0, w)
        plt.ylim(h, 0)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        flow = Image.open(buf).convert("RGBA")
        output_frame.paste(flow, (w, h))
        plt.close()
        frame_list.append(output_frame)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frame_list, fps=fps)

def flow_uv_to_colors(u, v, rad, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (torch.tensor): Input horizontal flow of shape [N,H,W]
        v (torch.tensor): Input vertical flow of shape [N,H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        torch.tensor: Flow visualization image of shape [N,3,H,W]
    """
    flow_image = torch.zeros((u.shape[0], 3, u.shape[1], u.shape[2]), dtype=torch.uint8, device=u.device)
    colorwheel = COLORWHEEL.to(u.device)
    ncols = colorwheel.shape[0]
    a = torch.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).int()
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, ch_idx, :, :] = torch.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Adapted from Tora: https://github.com/alibaba/Tora/blob/14db1b0a074284a6c265564eef07f5320911dc00/sat/utils/flow_utils.py#L120
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (torch.Tensor): Flow UV image of shape [N,2,H,W]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        torch.Tensor: Flow visualization image of shape [N,3,H,W]
    """
    if clip_flow is not None:
        flow_uv = torch.clamp(flow_uv, 0, clip_flow)
    u = flow_uv[:, 0]
    v = flow_uv[:, 1]
    rad = torch.sqrt(u**2 + v**2)
    rad_max = torch.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    flow_image = flow_uv_to_colors(u, v, rad, convert_to_bgr)
    return flow_image

def generate_gaussian_template(imgSize=200):
    """ Adapted from DragAnything: https://github.com/showlab/DragAnything/blob/79355363218a7eb9b3437a31b8604b6d436d9337/dataset/dataset.py#L110"""
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    # Guass Map
    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)

    # isotropicGrayscaleImage = cv2.resize(isotropicGrayscaleImage, (40, 40))
    return isotropicGrayscaleImage

def generate_gaussian_heatmap(tracks, width, height, layer_index, layer_capacity, side=20, offset=True):
    heatmap_template = generate_gaussian_template()
    num_frames, num_points = tracks.shape[:2]
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.cpu().numpy()
    if offset:
        offset_kernel = cv2.resize(heatmap_template / 255, (2 * side + 1, 2 * side + 1))
        offset_kernel /= np.sum(offset_kernel)
        offset_kernel /= offset_kernel[side, side]
    heatmaps = []
    for frame_idx in range(num_frames):
        if offset:
            layer_imgs = np.zeros((layer_capacity, height, width, 3), dtype=np.float32)
        else:
            layer_imgs = np.zeros((layer_capacity, height, width, 1), dtype=np.float32)
        layer_heatmaps = []
        for point_idx in range(num_points):
            x, y = tracks[frame_idx, point_idx]
            layer_id = layer_index[point_idx]
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            x1 = int(max(x - side, 0))
            x2 = int(min(x + side, width - 1))
            y1 = int(max(y - side, 0))
            y2 = int(min(y + side, height - 1))
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            temp_map = cv2.resize(heatmap_template, (x2-x1, y2-y1))
            layer_imgs[layer_id, y1:y2,x1:x2, 0] = np.maximum(layer_imgs[layer_id, y1:y2,x1:x2, 0], temp_map)
            if offset:
                if frame_idx < num_frames - 1:
                    next_x, next_y = tracks[frame_idx + 1, point_idx]
                else:
                    next_x, next_y = x, y
                layer_imgs[layer_id, int(y), int(x), 1] = next_x - x
                layer_imgs[layer_id, int(y), int(x), 2] = next_y - y
        for img in layer_imgs:
            if offset:
                img[:, :, 1:] = cv2.filter2D(img[:, :, 1:], -1, offset_kernel)
            else:
                img = cv2.cvtColor(img[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            layer_heatmaps.append(img)
        heatmaps.append(np.stack(layer_heatmaps, axis=0))
    heatmaps = np.stack(heatmaps, axis=0)
    return torch.from_numpy(heatmaps).permute(0, 1, 4, 2, 3).contiguous().float()   # [F, N_layer, C, H, W]
