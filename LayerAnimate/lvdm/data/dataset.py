import os, random
import cv2
import pandas as pd
import numpy as np
from einops import repeat
from decord import VideoReader
import gc
from contextlib import contextmanager
import pickle
from pycocotools.mask import decode
from einops import rearrange
from scipy.cluster.hierarchy import linkage

import torch
from torchvision.transforms import transforms, functional as F
from torch.utils.data.dataset import Dataset
from PIL import Image
from ..utils import flow_to_image


def get_random_keyframe_indices(shape, p_single_frame=0.5):
    # mask = 0 means the pixel is masked
    f, c, h, w = shape
    mask = torch.zeros(f, dtype=torch.bool)
    if np.random.rand() < p_single_frame:
        mask[0] = True
    else:
        mask[0] = True
        mask[-1] = True
    return torch.nonzero(mask).squeeze(-1)

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

class AnimeDataset(Dataset):
    def __init__(
        self,
        csv_path, video_folder=None,
        sample_size=256, sample_stride=3, sample_n_frames=16,
        length_drop_start=0.1, length_drop_end=0.9,
        p_single_frame=0.5, p_bisection=0.8, p_flip=0.5,
        return_types=[], max_motion_score=-1, static_motion_score=-1,
        layer_capacity=4, eta_s=3.0, binarize_sketch=False,
        in_object_ratio=0.8, visible_ratio=0.1, consider_visibility=False,
        max_traj_per_object=3, min_traj_per_object=1,
        max_traj_per_layer=-1, min_traj_per_layer=1,
        traj_sample_seed=42, traj_mode="heatmap", heatmap_offset=False,
    ):
        self.dataset = pd.read_csv(csv_path)
        self.length = len(self.dataset)

        self.video_folder       = video_folder
        self.sample_stride      = sample_stride     # specifically, it is the maximum sampling stride
        self.sample_n_frames    = sample_n_frames
        self.length_drop_start  = length_drop_start
        self.length_drop_end    = length_drop_end
        self.p_single_frame     = p_single_frame
        self.p_bisection        = p_bisection
        self.p_flip             = p_flip
        self.return_types       = return_types      # ["sketch", "layer", "trajectory"]
        self.layer_capacity     = layer_capacity
        self.binarize_sketch    = binarize_sketch
        self.eta_s              = eta_s
        self.max_motion_score   = max_motion_score
        self.static_motion_score = static_motion_score
        self.in_object_ratio    = in_object_ratio
        self.visible_ratio      = visible_ratio
        self.max_traj_per_object = max_traj_per_object
        self.min_traj_per_object = min_traj_per_object
        self.max_traj_per_layer = max_traj_per_layer
        self.min_traj_per_layer = min_traj_per_layer
        self.consider_visibility = consider_visibility
        self.traj_mode          = traj_mode
        self.heatmap_offset     = heatmap_offset
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(min(self.sample_size)),
            transforms.CenterCrop(self.sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        if "layer" in self.return_types:
            self.mask_transforms = transforms.Compose([
                transforms.Resize(min(self.sample_size)),
                transforms.CenterCrop(self.sample_size),
            ])
        if "trajectory" in self.return_types:
            if self.traj_mode == "heatmap":
                self.heatmap_template = self.generate_gaussian_template()
            elif self.traj_mode == "flowmap":
                self.flow_kernel = self.generate_gaussian_kernel()
            self.traj_sample_rng = torch.Generator().manual_seed(traj_sample_seed)

    def get_batch(self, idx):
        video_dict = self.dataset.iloc[idx]
        text = video_dict['text']
        video_name = video_dict['id']
        fps = np.round(video_dict['fps']).astype(int)
        video_dir = os.path.join(self.video_folder, "clip", video_dict['relpath'])

        if "layer" in self.return_types:
            with open(os.path.join(self.video_folder, "masklet_f48", f"{video_name}.pkl"), "rb") as f:
                masklets = pickle.load(f)
            keyframe_index = np.array(sorted(list(masklets.keys())))

        with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
            video_length = len(video_reader)
            if video_length == 0:
                raise ValueError(f"No Frames in video.")
            start_pos = int(video_length * self.length_drop_start)
            end_pos = int(video_length * self.length_drop_end)  # exclusive

            if self.sample_n_frames == 48:
                sample_stride = np.clip((end_pos - start_pos) // (self.sample_n_frames - 1), 1, self.sample_stride)
                clip_length = (self.sample_n_frames - 1) * sample_stride + 1
                start_idx = start_pos
                end_idx = min(start_idx + clip_length, end_pos) - 1
                batch_index = np.linspace(start_idx, end_idx, self.sample_n_frames, dtype=int)
            else:
                valid_index = keyframe_index[(keyframe_index >= start_pos) & (keyframe_index < end_pos)]
                batch_index = valid_index[np.linspace(0, len(valid_index) - 1, self.sample_n_frames, dtype=int)]

            mean_sample_stride = np.clip(np.diff(batch_index).mean(), a_min=0.1, a_max=self.sample_stride)
            fps = np.round(fps / mean_sample_stride).astype(int)

            try:
                pixel_values = video_reader.get_batch(batch_index).asnumpy()
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")
            origin_h, origin_w = pixel_values.shape[1:3]

        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.
        if np.random.rand() < self.p_flip:
            pixel_values = F.hflip(pixel_values)
            do_flip = True
        else:
            do_flip = False
        pixel_values = self.pixel_transforms(pixel_values)

        keyframe_indices = get_random_keyframe_indices(pixel_values.shape, self.p_single_frame)    # [f, ]
        assert len(keyframe_indices) in [1, 2], f"control images length should be 1 or 2, but got {len(keyframe_indices)}."
        if len(keyframe_indices) == 1:
            keyframe_indices = torch.cat([keyframe_indices, keyframe_indices], dim=0)

        batch = dict(
            pixel_values=pixel_values, keyframe_indices=keyframe_indices,
            text=text, fps=fps, index=idx
        )

        if "layer" in self.return_types:
            with open(os.path.join(self.video_folder, "masklet_f48_score", f"{video_name}.pkl"), "rb") as f:
                masklet_scores = pickle.load(f)
            object_masks = []
            object_indices = sorted(list(masklets[batch_index[0]].keys()))
            for frame_idx in batch_index:
                object_masks.append(torch.from_numpy(np.stack([decode(masklets[frame_idx][object_idx]) for object_idx in object_indices])) > 0)
            object_masks = torch.stack(object_masks, dim=1)     # [N_object, F, H, W]

            # complement the non-object background
            non_object_mask = ~torch.any(object_masks, dim=0, keepdim=True)
            object_masks = torch.cat([non_object_mask, object_masks], dim=0)  # [N_object+1, F, H, W]
            object_indices = np.array([-1] + object_indices)
            layer_masks, motion_scores = self.hierarchical_merging(object_masks, masklet_scores, object_indices)

            # keep the original masks for trajectory generation
            if "trajectory" in self.return_types:
                original_object_masks = object_masks.clone()
                original_layer_masks = layer_masks.clone()

            layer_masks = rearrange(layer_masks, 'n f h w -> (n f) h w')
            if do_flip:
                layer_masks = F.hflip(layer_masks)
            layer_masks = self.mask_transforms(layer_masks[:, None].float()).squeeze(1) > 0.5
            layer_masks = rearrange(layer_masks, '(n f) h w -> n f h w', n=self.layer_capacity)

            layer_static = (motion_scores < self.static_motion_score) & layer_masks.any(dim=(1, 2, 3))
            motion_scores[layer_static] = 0.0
            if keyframe_indices[0] == keyframe_indices[1]:
                layer_regions = layer_masks[:, [0, 0]].unsqueeze(2) * pixel_values[[0, 0]].unsqueeze(0)    # [N, 2, C, H, W]
            else:
                layer_regions = layer_masks[:, [0, -1]].unsqueeze(2) * pixel_values[[0, -1]].unsqueeze(0)    # [N, 2, C, H, W]

            # normalize motion scores
            if self.max_motion_score > 0:
                motion_scores = motion_scores.clamp_max(self.max_motion_score) / self.max_motion_score

            batch["layer_masks"] = layer_masks                  # [N, F, H, W]
            batch["motion_scores"] = motion_scores              # [N, ]
            batch["layer_regions"] = layer_regions              # [N, 2, C, H, W]
            batch["layer_static"] = layer_static                # [N, ]

        if "trajectory" in self.return_types:
            object_idx_map = torch.zeros(original_object_masks.shape[1:], dtype=torch.long)   # [F, H, W]
            for object_idx, mask in enumerate(original_object_masks):
                object_idx_map[mask] = object_idx
            layer_idx_map = torch.zeros(original_layer_masks.shape[1:], dtype=torch.long)    # [F, H, W]
            for layer_idx, mask in enumerate(original_layer_masks):
                layer_idx_map[mask] = layer_idx

            raw_trajectory = np.load(os.path.join(self.video_folder, "masklet_f48_track_60", f"{video_name}.npz"))["tracks"]    # [F, N_point, 3], 3: x, y, visibliity
            assert len(raw_trajectory) == len(masklets), f"trajectory length does not match masklet length."
            relative_index = np.searchsorted(keyframe_index, batch_index)
            trajectory = torch.from_numpy(raw_trajectory[relative_index])                   # [F, N_point, 3]
            traj_object_index = torch.zeros(trajectory.shape[:2], dtype=torch.long) - 1     # [F, N_point]

            out_of_bounds_mask = (trajectory[..., 0] < 0) | (trajectory[..., 1] < 0) | (trajectory[..., 0] >= origin_w) | (trajectory[..., 1] >= origin_h)   # [F, N_point]
            in_bounds_indices = torch.nonzero(~out_of_bounds_mask)   # [N_valid, 2], frame_idx, point_idx
            frame_indices, point_indices = in_bounds_indices[:, 0], in_bounds_indices[:, 1]
            in_bounds_x = trajectory[frame_indices, point_indices, 0].long()
            in_bounds_y = trajectory[frame_indices, point_indices, 1].long()
            traj_object_index[frame_indices, point_indices] = object_idx_map[frame_indices, in_bounds_y, in_bounds_x]

            # filter out points that are not in the same object
            initial_object_index = traj_object_index[0]
            ratio_mask = (traj_object_index == initial_object_index[None]).sum(dim=0) / (traj_object_index != -1).sum(dim=0)
            ratio_mask = ratio_mask > self.in_object_ratio
            ratio_mask = ratio_mask & (initial_object_index > -1)

            # filter out points that are not visible
            visibility_mask = trajectory[..., 2].mean(dim=0) > self.visible_ratio

            # sample trajectory for each object
            combined_mask = ratio_mask & visibility_mask        # [N_point, ]
            retained_points = torch.where(combined_mask)[0]
            retained_initial_object_index = initial_object_index[retained_points]
            unique_initial_object_index = torch.unique(retained_initial_object_index)
            sampled_indices = []
            for obj_id in unique_initial_object_index:
                temp_mask = retained_initial_object_index == obj_id
                candidates = retained_points[temp_mask]
                if self.max_traj_per_object > 0:
                    sample_num = np.random.randint(self.min_traj_per_object, self.max_traj_per_object + 1)
                    if len(candidates) > sample_num:
                        candidate_motion = (trajectory[1:, candidates, :2] - trajectory[:-1, candidates, :2]).norm(dim=-1).mean(dim=0)
                        selected = torch.multinomial(candidate_motion, sample_num, replacement=False, generator=self.traj_sample_rng)
                        candidates = candidates[selected]
                sampled_indices.append(candidates)

            final_indices = torch.cat(sampled_indices) if sampled_indices else torch.tensor([], dtype=torch.long)
            filtered_trajectory = trajectory[:, final_indices, :]
            traj_layer_index = layer_idx_map[0, filtered_trajectory[0, :, 1].long(), filtered_trajectory[0, :, 0].long()]  # [N_point, ]
            # sample trajectory for each layer
            if self.max_traj_per_layer > 0:
                sampled_traj = []
                for layer_idx in range(self.layer_capacity):
                    candidates = filtered_trajectory[:, traj_layer_index == layer_idx]
                    sample_num = np.random.randint(self.min_traj_per_layer, self.max_traj_per_layer + 1)
                    if candidates.shape[1] > sample_num:
                        candidate_motion = (candidates[1:, :, :2] - candidates[:-1, :, :2]).norm(dim=-1).mean(dim=0)
                        selected = torch.multinomial(candidate_motion, sample_num, replacement=False, generator=self.traj_sample_rng)
                        sampled_traj.append(candidates[:, selected])
                    else:
                        sampled_traj.append(candidates)
                filtered_trajectory = torch.cat(sampled_traj, dim=1)
                traj_layer_index = layer_idx_map[0, filtered_trajectory[0, :, 1].long(), filtered_trajectory[0, :, 0].long()]

            if self.traj_mode == "heatmap":
                heatmap = self.generate_gaussian_heatmap(filtered_trajectory, origin_w, origin_h, traj_layer_index, consider_visibility=self.consider_visibility, offset=self.heatmap_offset)
                heatmap = rearrange(heatmap, 'f n c h w -> (f n) c h w')
                if self.heatmap_offset:
                    graymap, offset = heatmap[:, :1], heatmap[:, 1:]
                    graymap = graymap / 255.
                    rad = torch.sqrt(offset[:, 0:1]**2 + offset[:, 1:2]**2)
                    rad_max = torch.max(rad)
                    epsilon = 1e-5
                    offset = offset / (rad_max + epsilon)
                    if do_flip:
                        graymap = F.hflip(graymap)
                        offset = F.hflip(offset)
                        offset[:, 0] = -offset[:, 0]
                    graymap = graymap * 2 - 1
                    heatmap = torch.cat([graymap, offset], dim=1)
                    heatmap = self.mask_transforms(heatmap)
                else:
                    heatmap = heatmap / 255.
                    if do_flip:
                        heatmap = F.hflip(heatmap)
                    heatmap = self.pixel_transforms(heatmap)
                heatmap = rearrange(heatmap, '(f n) c h w -> n f c h w', n=self.layer_capacity)
                batch["trajectory"] = heatmap
            elif self.traj_mode == "flowmap":
                filtered_trajectory[:, :, :2] = filtered_trajectory[:, :, :2] * filtered_trajectory.new_tensor(
                    [
                        (self.sample_size[1] - 1) / (origin_w - 1),
                        (self.sample_size[0] - 1) / (origin_h - 1)
                    ]
                )[None, None]
                if do_flip:
                    filtered_trajectory[:, :, 0] = self.sample_size[1] - 1 - filtered_trajectory[:, :, 0]
                flowmap = self.generate_flowmap(filtered_trajectory, self.sample_size[1], self.sample_size[0], traj_layer_index, consider_visibility=self.consider_visibility)
                flowmap = rearrange(flowmap, 'f n c h w -> (f n) c h w')
                flowmap = flow_to_image(flowmap) / 255. * 2 - 1
                flowmap = rearrange(flowmap, '(f n) c h w -> n f c h w', n=self.layer_capacity)
                batch["trajectory"] = flowmap

        return batch

    def hierarchical_merging(self, masklets, masklet_scores, object_indices):
        # masklets: [N_object, F, H, W]
        # masklet_scores (Dict): key: object id, value: motion score
        # object_indices: list of object ids
        F, H, W = masklets.shape[1:]
        object_scores = np.array([masklet_scores[object_idx] for object_idx in object_indices]).reshape(-1, 1)
        layer_indices = np.arange(object_scores.shape[0])

        if object_scores.shape[0] >= 2:
            z = linkage(object_scores, method="ward")
            pairs = z[:, :2].astype(int)
            dists = z[:, 2]
            n_group = len(layer_indices)
            for row_idx, (pair, dist) in enumerate(zip(pairs, dists)):
                if n_group <= self.layer_capacity and (dist > self.eta_s or n_group <= 2):
                    break
                else:
                    mask0 = layer_indices == pair[0]
                    mask1 = layer_indices == pair[1]
                    n_group -= 1
                    layer_indices[mask0] = row_idx + len(layer_indices)
                    layer_indices[mask1] = row_idx + len(layer_indices)
            # remap layer_id to 0 ~ n_group - 1
            new_layer_indices = np.zeros_like(layer_indices)
            for i, layer_idx in enumerate(np.unique(layer_indices)):
                new_layer_indices[layer_indices == layer_idx] = i
            layer_indices = new_layer_indices

        layer_masks = torch.zeros((self.layer_capacity, F, H, W), dtype=torch.bool)
        motion_scores = torch.zeros(self.layer_capacity, dtype=torch.float32)
        for layer_idx in range(self.layer_capacity):
            masklet_indices = (layer_indices == layer_idx)
            if masklet_indices.any():
                layer_masks[layer_idx] = torch.any(masklets[masklet_indices], dim=0)
                motion_scores[layer_idx] = object_scores[masklet_indices].mean()
        return layer_masks, motion_scores

    def generate_gaussian_heatmap(self, tracks, width, height, layer_index, side=20, consider_visibility=True, offset=False):
        num_frames, num_points = tracks.shape[:2]
        if isinstance(tracks, torch.Tensor):
            tracks = tracks.cpu().numpy()
        if offset:
            offset_kernel = cv2.resize(self.heatmap_template / 255, (2 * side + 1, 2 * side + 1))
            offset_kernel /= np.sum(offset_kernel)
            offset_kernel /= offset_kernel[side, side]
        heatmaps = []
        for frame_idx in range(self.sample_n_frames):
            if offset:
                layer_imgs = np.zeros((self.layer_capacity, height, width, 3), dtype=np.float32)
            else:
                layer_imgs = np.zeros((self.layer_capacity, height, width, 1), dtype=np.float32)
            layer_heatmaps = []
            for point_idx in range(num_points):
                x, y, visiblity = tracks[frame_idx, point_idx]
                layer_id = layer_index[point_idx]
                if x < 0 or y < 0 or x >= width or y >= height or (consider_visibility and visiblity < 0.5):
                    continue
                x1 = int(max(x - side, 0))
                x2 = int(min(x + side, width - 1))
                y1 = int(max(y - side, 0))
                y2 = int(min(y + side, height - 1))
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                temp_map = cv2.resize(self.heatmap_template, (x2-x1, y2-y1))
                layer_imgs[layer_id, y1:y2,x1:x2, 0] = np.maximum(layer_imgs[layer_id, y1:y2,x1:x2, 0], temp_map)
                if offset:
                    if frame_idx < self.sample_n_frames - 1:
                        next_x, next_y, _ = tracks[frame_idx + 1, point_idx]
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

    def generate_gaussian_template(self, imgSize=200):
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

    def generate_flowmap(self, tracks, width, height, layer_index, consider_visibility=True):
        num_frames, num_points = tracks.shape[:2]
        if isinstance(tracks, torch.Tensor):
            tracks = tracks.cpu().numpy()
        flowmaps = []
        for frame_idx in range(self.sample_n_frames - 1):
            flowmap = np.zeros((self.layer_capacity, height, width, 2), dtype=np.float32)
            for point_idx in range(num_points):
                x0, y0, visiblity0 = tracks[frame_idx, point_idx]
                x1, y1, visiblity1 = tracks[frame_idx + 1, point_idx]
                layer_id = layer_index[point_idx]
                if x0 < 0 or y0 < 0 or x0 >= width or y0 >= height or (consider_visibility and visiblity0 < 0.5):
                    continue
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                flowmap[layer_id, y0, x0, 0] = x1 - x0
                flowmap[layer_id, y0, x0, 1] = y1 - y0
            for i in range(self.layer_capacity):
                flowmap[i] = cv2.filter2D(flowmap[i], -1, self.flow_kernel)
            flowmaps.append(flowmap)
        flowmaps = np.stack(flowmaps, axis=0)
        flowmaps = np.concatenate([flowmaps, np.zeros_like(flowmaps[:1])], axis=0)
        return torch.from_numpy(flowmaps).permute(0, 1, 4, 2, 3).contiguous().float()

    def generate_gaussian_kernel(self, kernel_size=99, sig_x=10, sig_y=10):
        """
        Adapted from Tora: https://github.com/alibaba/Tora/blob/14db1b0a074284a6c265564eef07f5320911dc00/diffusers-version/tora/traj_utils.py#L69
        Generate a bivariate isotropic Gaussian kernel.
        """
        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        grid = np.hstack(
            (
                xx.reshape((kernel_size * kernel_size, 1)),
                yy.reshape(kernel_size * kernel_size, 1),
            )
        ).reshape(kernel_size, kernel_size, 2)
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
        kernel = kernel / np.sum(kernel)
        kernel = kernel / kernel[kernel_size // 2, kernel_size // 2]
        return kernel

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.get_batch(idx)
                break

            except Exception as e:
                print(f"Error in getting sample {idx}. Error is {e}.")
                idx = random.randint(0, self.length-1)

        return sample
