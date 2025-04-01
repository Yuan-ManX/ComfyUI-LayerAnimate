import argparse
import sys
import datetime
import os
import json

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import spaces
from PIL import Image
from typing import List

from diffusers import DDIMScheduler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lvdm.models.unet import UNetModel
from lvdm.models.autoencoder import AutoencoderKL, AutoencoderKL_Dualref
from lvdm.models.condition import FrozenOpenCLIPEmbedder, FrozenOpenCLIPImageEmbedderV2, Resampler
from lvdm.models.layer_controlnet import LayerControlNet
from lvdm.pipelines.pipeline_animation import AnimationPipeline
from lvdm.utils import generate_gaussian_heatmap, save_videos_grid, save_videos_with_traj

from einops import rearrange
import cv2
import decord
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.interpolate import PchipInterpolator

class LayerAnimate:

    @spaces.GPU
    def __init__(self, args):
        if args.savedir is None:
            time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            savedir = f"samples/{time_str}"
        else:
            savedir = args.savedir
        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)

        self.weight_dtype  = torch.bfloat16
        self.device        = args.device
        self.text_encoder  = FrozenOpenCLIPEmbedder().eval()
        self.image_encoder = FrozenOpenCLIPImageEmbedderV2().eval()

        self.W = args.W
        self.H = args.H
        self.L = args.L
        self.layer_capacity = args.layer_capacity

        self.transforms = transforms.Compose([
            transforms.Resize(min(self.H, self.W)),
            transforms.CenterCrop((self.H, self.W)),
        ])
        self.pipeline = None
        self.generator = None
        # sample_grid is used to generate fixed trajectories to freeze static layers
        self.sample_grid = np.meshgrid(np.linspace(0, self.W - 1, 10, dtype=int), np.linspace(0, self.H - 1, 10, dtype=int))
        self.sample_grid = np.stack(self.sample_grid, axis=-1).reshape(-1, 1, 2)
        self.sample_grid = np.repeat(self.sample_grid, self.L, axis=1) # [N, F, 2]

    @spaces.GPU
    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.generator = torch.Generator(self.device).manual_seed(seed)

    @spaces.GPU
    def set_model(self, pretrained_model_path):
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        image_projector = Resampler.from_pretrained(pretrained_model_path, subfolder="image_projector").eval()
        vae, vae_dualref = None, None
        if "I2V" or "Mix" in pretrained_model_path:
            vae           = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").eval()
        if "Interp" or "Mix" in pretrained_model_path:
            vae_dualref   = AutoencoderKL_Dualref.from_pretrained(pretrained_model_path, subfolder="vae_dualref").eval()
        unet              = UNetModel.from_pretrained(pretrained_model_path, subfolder="unet").eval()
        layer_controlnet  = LayerControlNet.from_pretrained(pretrained_model_path, subfolder="layer_controlnet").eval()

        self.pipeline = AnimationPipeline(
            vae=vae, vae_dualref=vae_dualref, text_encoder=self.text_encoder, image_encoder=self.image_encoder, image_projector=image_projector,
            unet=unet, layer_controlnet=layer_controlnet, scheduler=scheduler
        ).to(device=self.device, dtype=self.weight_dtype)
        if "Interp" or "Mix" in pretrained_model_path:
            self.pipeline.vae_dualref.decoder.to(dtype=torch.float32)
        return pretrained_model_path

    def upload_image(self, image):
        image = self.transforms(image)
        return image

    def run(self, input_image, input_image_end, pretrained_model_path, seed,
            prompt, n_prompt, num_inference_steps, guidance_scale,
            *layer_args):
        self.set_seed(seed)
        global layer_tracking_points
        args_layer_tracking_points = [layer_tracking_points[i].value for i in range(self.layer_capacity)]

        args_layer_masks = layer_args[:self.layer_capacity]
        args_layer_masks_end = layer_args[self.layer_capacity : 2 * self.layer_capacity]
        args_layer_controls = layer_args[2 * self.layer_capacity : 3 * self.layer_capacity]
        args_layer_scores = list(layer_args[3 * self.layer_capacity : 4 * self.layer_capacity])
        args_layer_sketches = layer_args[4 * self.layer_capacity : 5 * self.layer_capacity]
        args_layer_valids = layer_args[5 * self.layer_capacity : 6 * self.layer_capacity]
        args_layer_statics = layer_args[6 * self.layer_capacity : 7 * self.layer_capacity]
        for layer_idx in range(self.layer_capacity):
            if args_layer_controls[layer_idx] != "score":
                args_layer_scores[layer_idx] = -1
            if args_layer_statics[layer_idx]:
                args_layer_scores[layer_idx] = 0

        mode = "i2v"
        image1 = F.to_tensor(input_image) * 2 - 1
        frame_tensor = image1[None].to(self.device) # [F, C, H, W]
        if input_image_end is not None:
            mode = "interpolate"
            image2 = F.to_tensor(input_image_end) * 2 - 1
            frame_tensor2 = image2[None].to(self.device)
            frame_tensor = torch.cat([frame_tensor, frame_tensor2], dim=0)
        frame_tensor = frame_tensor[None]

        if mode == "interpolate":
            layer_masks = torch.zeros((1, self.layer_capacity, 2, 1, self.H, self.W), dtype=torch.bool)
        else:
            layer_masks = torch.zeros((1, self.layer_capacity, 1, 1, self.H, self.W), dtype=torch.bool)
        for layer_idx in range(self.layer_capacity):
            if args_layer_masks[layer_idx] is not None:
                mask = F.to_tensor(args_layer_masks[layer_idx]) > 0.5
                layer_masks[0, layer_idx, 0] = mask
            if args_layer_masks_end[layer_idx] is not None and mode == "interpolate":
                mask = F.to_tensor(args_layer_masks_end[layer_idx]) > 0.5
                layer_masks[0, layer_idx, 1] = mask
        layer_masks = layer_masks.to(self.device)
        layer_regions = layer_masks * frame_tensor[:, None]
        layer_validity = torch.tensor([args_layer_valids], dtype=torch.bool, device=self.device)
        motion_scores = torch.tensor([args_layer_scores], dtype=self.weight_dtype, device=self.device)
        layer_static = torch.tensor([args_layer_statics], dtype=torch.bool, device=self.device)

        sketch = torch.ones((1, self.layer_capacity, self.L, 3, self.H, self.W), dtype=self.weight_dtype)
        for layer_idx in range(self.layer_capacity):
            sketch_path = args_layer_sketches[layer_idx]
            if sketch_path is not None:
                video_reader = decord.VideoReader(sketch_path)
                assert len(video_reader) == self.L, f"Input the length of sketch sequence should match the video length."
                video_frames = video_reader.get_batch(range(self.L)).asnumpy()
                sketch_values = [F.to_tensor(self.transforms(Image.fromarray(frame))) for frame in video_frames]
                sketch_values = torch.stack(sketch_values) * 2 - 1
                sketch[0, layer_idx] = sketch_values
        sketch = sketch.to(self.device)

        heatmap = torch.zeros((1, self.layer_capacity, self.L, 3, self.H, self.W), dtype=self.weight_dtype)
        heatmap[:, :, :, 0] -= 1
        trajectory = []
        traj_layer_index = []
        for layer_idx in range(self.layer_capacity):
            tracking_points = args_layer_tracking_points[layer_idx]
            if args_layer_statics[layer_idx]:
                # generate pseudo trajectory for static layers
                temp_layer_mask = layer_masks[0, layer_idx, 0, 0].cpu().numpy()
                valid_flag = temp_layer_mask[self.sample_grid[:, 0, 1], self.sample_grid[:, 0, 0]]
                valid_grid = self.sample_grid[valid_flag]    # [F, N, 2]
                trajectory.extend(list(valid_grid))
                traj_layer_index.extend([layer_idx] * valid_grid.shape[0])
            else:
                for temp_track in tracking_points:
                    if len(temp_track) > 1:
                        x = [point[0] for point in temp_track]
                        y = [point[1] for point in temp_track]
                        t = np.linspace(0, 1, len(temp_track))
                        fx = PchipInterpolator(t, x)
                        fy = PchipInterpolator(t, y)
                        t_new = np.linspace(0, 1, self.L)
                        x_new = fx(t_new)
                        y_new = fy(t_new)
                        temp_traj = np.stack([x_new, y_new], axis=-1).astype(np.float32)
                        trajectory.append(temp_traj)
                        traj_layer_index.append(layer_idx)
                    elif len(temp_track) == 1:
                        trajectory.append(np.array(temp_track * self.L))
                        traj_layer_index.append(layer_idx)

        trajectory = np.stack(trajectory)
        trajectory = np.transpose(trajectory, (1, 0, 2))
        traj_layer_index = np.array(traj_layer_index)
        heatmap = generate_gaussian_heatmap(trajectory, self.W, self.H, traj_layer_index, self.layer_capacity, offset=True)
        heatmap = rearrange(heatmap, "f n c h w -> (f n) c h w")
        graymap, offset = heatmap[:, :1], heatmap[:, 1:]
        graymap = graymap / 255.
        rad = torch.sqrt(offset[:, 0:1]**2 + offset[:, 1:2]**2)
        rad_max = torch.max(rad)
        epsilon = 1e-5
        offset = offset / (rad_max + epsilon)
        graymap = graymap * 2 - 1
        heatmap = torch.cat([graymap, offset], dim=1)
        heatmap = rearrange(heatmap, '(f n) c h w -> n f c h w', n=self.layer_capacity)
        heatmap = heatmap[None]
        heatmap = heatmap.to(self.device)

        sample = self.pipeline(
            prompt,
            self.L,
            self.H,
            self.W,
            frame_tensor,
            layer_masks             = layer_masks,
            layer_regions           = layer_regions,
            layer_static            = layer_static,
            motion_scores           = motion_scores,
            sketch                  = sketch,
            trajectory              = heatmap,
            layer_validity          = layer_validity,
            num_inference_steps     = num_inference_steps,
            guidance_scale          = guidance_scale,
            guidance_rescale        = 0.7,
            negative_prompt         = n_prompt,
            num_videos_per_prompt   = 1,
            eta                     = 1.0,
            generator               = self.generator,
            fps                     = 24,
            mode                    = mode,
            weight_dtype            = self.weight_dtype,
            output_type             = "tensor",
        ).videos
        output_video_path = os.path.join(self.savedir, "video.mp4")
        save_videos_grid(sample, output_video_path, fps=8)
        output_video_traj_path = os.path.join(self.savedir, "video_with_traj.mp4")
        vis_traj_flag = np.zeros(trajectory.shape[1], dtype=bool)
        for traj_idx in range(trajectory.shape[1]):
            if not args_layer_statics[traj_layer_index[traj_idx]]:
                vis_traj_flag[traj_idx] = True
        vis_traj = torch.from_numpy(trajectory[:, vis_traj_flag])
        save_videos_with_traj(sample[0], vis_traj, os.path.join(self.savedir, f"video_with_traj.mp4"), fps=8, line_width=7, circle_radius=10)
        return output_video_path, output_video_traj_path


def update_layer_region(image, layer_mask):
    if image is None or layer_mask is None:
        return None, False
    layer_mask_tensor = (F.to_tensor(layer_mask) > 0.5).float()
    image = F.to_tensor(image)
    layer_region = image * layer_mask_tensor
    layer_region = F.to_pil_image(layer_region)
    layer_region.putalpha(layer_mask)
    return layer_region, True

def control_layers(control_type):
    if control_type == "score":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif control_type == "trajectory":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def visualize_trajectory(tracking_points, first_frame, first_mask, last_frame, last_mask):
    first_mask_tensor = (F.to_tensor(first_mask) > 0.5).float()
    first_frame = F.to_tensor(first_frame)
    first_region = first_frame * first_mask_tensor
    first_region = F.to_pil_image(first_region)
    first_region.putalpha(first_mask)
    transparent_background = first_region.convert('RGBA')

    if last_frame is not None and last_mask is not None:
        last_mask_tensor = (F.to_tensor(last_mask) > 0.5).float()
        last_frame = F.to_tensor(last_frame)
        last_region = last_frame * last_mask_tensor
        last_region = F.to_pil_image(last_region)
        last_region.putalpha(last_mask)
        transparent_background_end = last_region.convert('RGBA')

    width, height = transparent_background.size
    transparent_layer = np.zeros((height, width, 4))
    for track in tracking_points:
        if len(track) > 1:
            for i in range(len(track)-1):
                start_point = np.array(track[i], dtype=np.int32)
                end_point = np.array(track[i+1], dtype=np.int32)
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = max(np.sqrt(vx**2 + vy**2), 1)
                if i == len(track)-2:
                    cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
        elif len(track) == 1:
            cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)
    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    if last_frame is not None and last_mask is not None:
        trajectory_map_end = Image.alpha_composite(transparent_background_end, transparent_layer)
    else:
        trajectory_map_end = None
    return trajectory_map, trajectory_map_end

def add_drag(layer_idx):
    global layer_tracking_points
    tracking_points = layer_tracking_points[layer_idx].value
    tracking_points.append([])
    return

def delete_last_drag(layer_idx, first_frame, first_mask, last_frame, last_mask):
    global layer_tracking_points
    tracking_points = layer_tracking_points[layer_idx].value
    tracking_points.pop()
    trajectory_map, trajectory_map_end = visualize_trajectory(tracking_points, first_frame, first_mask, last_frame, last_mask)
    return trajectory_map, trajectory_map_end

def delete_last_step(layer_idx, first_frame, first_mask, last_frame, last_mask):
    global layer_tracking_points
    tracking_points = layer_tracking_points[layer_idx].value
    tracking_points[-1].pop()
    trajectory_map, trajectory_map_end = visualize_trajectory(tracking_points, first_frame, first_mask, last_frame, last_mask)
    return trajectory_map, trajectory_map_end

def add_tracking_points(layer_idx, first_frame, first_mask, last_frame, last_mask, evt: gr.SelectData):  # SelectData is a subclass of EventData
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")
    global layer_tracking_points
    tracking_points = layer_tracking_points[layer_idx].value
    tracking_points[-1].append(evt.index)
    trajectory_map, trajectory_map_end = visualize_trajectory(tracking_points, first_frame, first_mask, last_frame, last_mask)
    return trajectory_map, trajectory_map_end

def reset_states(layer_idx, first_frame, first_mask, last_frame, last_mask):
    global layer_tracking_points
    layer_tracking_points[layer_idx].value = [[]]
    tracking_points = layer_tracking_points[layer_idx].value
    trajectory_map, trajectory_map_end = visualize_trajectory(tracking_points, first_frame, first_mask, last_frame, last_mask)
    return trajectory_map, trajectory_map_end

def upload_tracking_points(tracking_path, layer_idx, first_frame, first_mask, last_frame, last_mask):
    if tracking_path is None:
        layer_region, _ = update_layer_region(first_frame, first_mask)
        layer_region_end, _ = update_layer_region(last_frame, last_mask)
        return layer_region, layer_region_end

    global layer_tracking_points
    with open(tracking_path, "r") as f:
        tracking_points = json.load(f)
    layer_tracking_points[layer_idx].value = tracking_points
    trajectory_map, trajectory_map_end = visualize_trajectory(tracking_points, first_frame, first_mask, last_frame, last_mask)
    return trajectory_map, trajectory_map_end

def reset_all_controls():
    global args, layer_tracking_points
    outputs = []
    # Reset tracking points states
    for layer_idx in range(args.layer_capacity):
        layer_tracking_points[layer_idx].value = [[]]

    # Reset global components
    outputs.extend([
        "an anime scene.",  # text prompt
        "",                 # negative text prompt
        50,                 # inference steps
        7.5,                # guidance scale
        42,                 # seed
        None,               # input image
        None,               # input image end
        None,               # output video
        None,               # output video with trajectory
    ])
    # Reset layer controls visibility
    outputs.extend([None] * args.layer_capacity)    # layer masks
    outputs.extend([None] * args.layer_capacity)    # layer masks end
    outputs.extend([None] * args.layer_capacity)    # layer regions
    outputs.extend([None] * args.layer_capacity)    # layer regions end
    outputs.extend(["sketch"] * args.layer_capacity)    # layer controls
    outputs.extend([gr.update(visible=False, value=-1) for _ in range(args.layer_capacity)])    # layer score controls
    outputs.extend([gr.update(visible=False) for _ in range(4 * args.layer_capacity)])    # layer trajectory control 4 buttons
    outputs.extend([gr.update(visible=False, value=None) for _ in range(args.layer_capacity)])    # layer trajectory file
    outputs.extend([None] * args.layer_capacity)    # layer sketch controls
    outputs.extend([False] * args.layer_capacity)    # layer validity
    outputs.extend([False] * args.layer_capacity)    # layer statics
    return outputs


# ComfyUI Custom Node: Load Images
class LoadImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_paths": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "LayerAnimate"

    def load(self, image_paths):
        paths = image_paths.split(",")
        images = [Image.open(p).convert("RGBA") for p in paths if p.strip()]
        return (images,)


# ComfyUI Custom Node: Load Model
class LoadPretrainedModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_path": ("STRING", {"default": "None"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "LayerAnimate"

    def load(self, model_path):
        model = torch.load(model_path) if model_path != "None" else None
        return (model,)


# ComfyUI Custom Node: Animation Processing
class LayerAnimateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "input_image": ("IMAGE",),
                "input_image_end": ("IMAGE",),
                "text_prompt": ("STRING", {"default": "an anime scene."}),
                "negative_text_prompt": ("STRING", {"default": ""}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 7.5}),
                "seed": ("INT", {"default": 42}),
                "layer_masks": ("IMAGE",),
                "layer_masks_end": ("IMAGE",),
                "layer_controls": ("STRING",),
                "layer_score_controls": ("FLOAT",),
                "layer_sketch_controls": ("VIDEO",),
                "layer_valids": ("BOOL",),
                "layer_statics": ("BOOL",),
            }
        }

    RETURN_TYPES = ("VIDEO", "VIDEO")
    FUNCTION = "run"
    CATEGORY = "LayerAnimate"

    def run(self, model, input_image, input_image_end, text_prompt, negative_text_prompt, num_inference_steps,
            guidance_scale, seed, layer_masks, layer_masks_end, layer_controls, layer_score_controls,
            layer_sketch_controls, layer_valids, layer_statics):
        
        # TODO: Replace with actual model inference logic
        generated_video = "output_video.mp4"
        generated_video_traj = "output_video_traj.mp4"
        
        return (generated_video, generated_video_traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None)

    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=320)
    parser.add_argument("--layer_capacity", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

