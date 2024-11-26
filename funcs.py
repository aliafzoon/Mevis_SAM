import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label
from torch.nn.functional import one_hot


def random_sum_to(n, num_terms=None):
    """
    generate num_tersm with sum as n
    """
    num_terms = (num_terms or r.randint(2, n)) - 1
    a = random.sample(range(1, n), num_terms) + [0, n]
    list.sort(a)
    return [a[i + 1] - a[i] for i in range(len(a) - 1)]


def generate_click_prompt(img, msk, pt_label=1) -> torch.Tensor:
    """
    img and mask shape : (B, C, H, W)
    Returns: pt(torch.Tensor): prompt of shape (B, 2), prompt mask
    """
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w = msk.size()
    msk = msk[:, 0, :, :]
    pt_list = []
    msk_list = []
    for j in range(b):
        msk_s = msk[j, :, :]
        indices = torch.nonzero(msk_s)
        if indices.size(0) == 0:
            # generate a random array between [0-h, 0-h]:
            random_index = torch.randint(0, h, (2,)).to(device=msk.device)
            new_s = msk_s
        else:
            random_index = random.choice(indices)
            label = msk_s[random_index[0], random_index[1]]
            new_s = torch.zeros_like(msk_s)
            # convert bool tensor to int
            new_s = (msk_s == label).to(dtype=torch.float)
            # new_s[msk_s == label] = 1
        pt_list.append(random_index)
        msk_list.append(new_s)

    pt = torch.stack(pt_list, dim=0)
    msk = torch.stack(msk_list, dim=0)
    msk = msk.unsqueeze(1)

    return img, pt, msk  # [b, 2], [b, c, h, w]


def get_first_prompt(
    mask_cls, dist_thre_ratio=0.3, prompt_num=5, max_prompt_num=15, region_type="random"
):
    """
    if region_type = random, we random select one region and generate prompt
    if region_type = all, we generate prompt at each object region
    if region_type = largest_k, we generate prompt at largest k region, k <10
    """
    if prompt_num == -1:
        prompt_num = random.randint(1, max_prompt_num)
    # Find all disconnected regions
    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    # print('num of regions found', region_ids)
    ratio_list, regionid_list = [], []
    for region_id in range(1, region_ids + 1):
        # find coordinates of points in the region
        binary_msk = np.where(label_msk == region_id, 1, 0)

        # clean some region that is abnormally small
        r = np.sum(binary_msk) / np.sum(mask_cls)
        # print('curr mask over all mask ratio', r)
        ratio_list.append(r)
        regionid_list.append(region_id)
    if len(ratio_list) > 0:
        ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
        regionid_list = regionid_list[::-1]

        if region_type == "random":
            prompt_num = 1
            regionid_list = [random.choice(regionid_list)]  # random choose 1 region
            prompt_num_each_region = [1]
        elif region_type[:7] == "largest":
            region_max_num = int(region_type.split("_")[-1])
            # print(region_max_num,prompt_num,len(regionid_list))
            valid_region = min(region_max_num, len(regionid_list))
            if valid_region < prompt_num:
                prompt_num_each_region = random_sum_to(prompt_num, valid_region)
            else:
                prompt_num_each_region = prompt_num * [1]
            regionid_list = regionid_list[: min(valid_region, prompt_num)]
            # print(prompt_num_each_region)
        else:
            prompt_num_each_region = len(regionid_list) * [1]

        prompt = []
        mask_curr = np.zeros_like(label_msk)

        for reg_id in range(len(regionid_list)):
            binary_msk = np.where(label_msk == regionid_list[reg_id], 1, 0)
            mask_curr = np.logical_or(binary_msk, mask_curr)

            padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), "constant"))
            dist_img = cv2.distanceTransform(
                padded_mask, distanceType=cv2.DIST_L2, maskSize=5
            ).astype(np.float32)[1:-1, 1:-1]

            # sort the distances
            dist_array = sorted(dist_img.copy().flatten())[::-1]
            dist_array = np.array(dist_array)
            # find the threshold:
            dis_thre = max(dist_array[int(dist_thre_ratio * np.sum(dist_array > 0))], 1)
            # print(np.max(dist_array))
            # print(dis_thre)
            cY, cX = np.where(dist_img >= dis_thre)
            while prompt_num_each_region[reg_id] > 0:
                # random select one prompt
                random_idx = np.random.randint(0, len(cX))
                cx, cy = int(cX[random_idx]), int(cY[random_idx])
                prompt.append((cx, cy, 1))
                prompt_num_each_region[reg_id] -= 1

        while len(prompt) < max_prompt_num:  # repeat prompt to ensure the same size
            prompt.append((cx, cy, 1))
    else:  # if this image doesn't have target object
        prompt = [(0, 0, -1)]
        mask_curr = np.zeros_like(label_msk)
        while len(prompt) < max_prompt_num:  # repeat prompt to ensure the same size
            prompt.append((0, 0, -1))
    prompt = np.array(prompt)
    mask_curr = np.array(mask_curr, dtype=int)
    return prompt, mask_curr


def get_top_boxes(
    mask_cls, dist_thre_ratio=0.10, prompt_num=15, region_type="largest_15"
):
    # Find all disconnected regions
    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
    # print('num of regions found', region_ids)
    ratio_list, regionid_list = [], []
    for region_id in range(1, region_ids + 1):
        # find coordinates of points in the region
        binary_msk = np.where(label_msk == region_id, 1, 0)

        # clean some region that is abnormally small
        r = np.sum(binary_msk) / np.sum(mask_cls)
        # print('curr mask over all mask ratio', r)
        ratio_list.append(r)
        regionid_list.append(region_id)
    if len(ratio_list) > 0:
        # sort the region from largest to smallest
        ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
        regionid_list = regionid_list[::-1]

        if region_type == "random":
            prompt_num = 1
            regionid_list = [random.choice(regionid_list)]  # random choose 1 region
        elif region_type[:7] == "largest":
            region_max_num = int(region_type.split("_")[-1])
            regionid_list = regionid_list[: min(region_max_num, len(regionid_list))]

        prompt = []
        mask_curr = np.zeros_like(label_msk)
        for reg_id in range(len(regionid_list)):
            binary_msk = np.where(label_msk == regionid_list[reg_id], 1, 0)
            mask_curr = np.logical_or(binary_msk, mask_curr)
            box = MaskToBoxSimple(binary_msk, dist_thre_ratio)
            prompt.append(box)

        while len(prompt) < prompt_num:  # repeat prompt to ensure the same size
            prompt.append(box)
        prompt = np.array(prompt)
        mask_curr = np.array(mask_curr, dtype=int)
    else:
        prompt = [[0, 0, 0, 0]]
        mask_curr = np.zeros_like(label_msk)
        while len(prompt) < prompt_num:
            prompt.append(prompt[0])
    return prompt, mask_curr


def MaskToBoxSimple(mask, random_thre=0.05):
    """
    random_thre, the randomness at each side of box
    """
    mask = mask.squeeze()

    y_max, x_max = mask.shape[0], mask.shape[1]

    # find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0, x0 = row.min(), col.min()
    y1, x1 = row.max(), col.max()

    y_thre = (y1 - y0) * random_thre
    x_thre = (x1 - x0) * random_thre

    x0 = max(0, x0 - x_thre * random.random())
    x1 = min(x_max, x1 + x_thre * random.random())

    y0 = max(0, y0 - y_thre * random.random())
    y1 = min(y_max, y1 + y_thre * random.random())

    return [x0, y0, x1, y1]


def min_max_normalize(tensor, p=0.01):
    p_min = torch.quantile(tensor, p)
    p_max = torch.quantile(tensor, 1 - p)
    tensor = torch.clamp(tensor, p_min, p_max)
    return tensor


# -------------
def sum_along_axis(volume, axis):
    """
    Sum all pixel values of a 3D image volume along a specified axis.
    Args:
        - volume (torch.Tensor): 3D tensor of shape (depth, height, width)
        - axis (int): Axis along which to sum (0 for depth, 1 for height, 2 for width)
    Returns:
        - torch.Tensor: Summed tensor
    """
    if axis not in [0, 1, 2]:
        raise ValueError(
            "Invalid axis. Choose from 0 (depth), 1 (height), or 2 (width)."
        )

    return torch.sum(volume, dim=axis)


def calculate_sensitivity_specificity(pred, target, class_index):
    """
    Calculates sensitivity and specificity for a given class in semantic segmentation.

    Args:
      pred: Predicted segmentation mask (B, C, H, W)
      target: Ground truth segmentation mask (B, C, H, W)
      class_index: Index of the class to evaluate

    Returns:
      sensitivity, specificity
    """

    # Convert to binary masks for the chosen class
    pred_binary = (pred == class_index).float()
    target_binary = (target == class_index).float()
    # print(pred_binary.size(), target_binary.size())

    # Flatten the tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    # Calculate confusion matrix
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)

    return sensitivity, specificity


def do_inference_on_batch(net, batch, prompt=False):
    net.eval()
    bs = batch["images"].shape[0]
    images = batch["images"]
    masks = batch.get("masks", None)
    atten_map = batch.get("atten_map", torch.zeros((bs, 64, 64))).unsqueeze(1)
    device = images.device
    with torch.no_grad():
        image_embeddings = net.image_encoder(images)
        if prompt and masks != None:
            images, point_coords, masks = generate_click_prompt(images, masks)
            point_labels = torch.ones(images.size(0))
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=device
            )
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
            coords_torch, labels_torch = (
                coords_torch[None, :, :],
                labels_torch[None, :],
            )
            pt = (coords_torch, labels_torch)
        else:
            pt = None
        sparse_embeddings, dense_embeddings = net.prompt_encoder(
            points=pt, boxes=None, masks=None
        )

        image_pe = torch.stack(
            [net.prompt_encoder.get_dense_pe() for i in range(bs)], dim=0
        ).squeeze(1)
        image_embeddings = net.attention_fusion(image_embeddings, atten_map)
        low_res_masks, iou_predictions = net.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
    return low_res_masks, iou_predictions


def weighted_average(metrics, n_slices):
    total = sum(n_slices)
    weighted_sum = 0
    for m, n in zip(metrics, n_slices):
        weighted_sum += m * n
    return weighted_sum / total


def get_n_slices(path: Path):
    [slice_1, slice_2] = path.stem.split("-")[1:3]
    return int(slice_2) - int(slice_1) + 1


def merge_metrics(data_dict):
    images = list(data_dict.keys())
    merged_data = {}
    for image in images:
        n_slices = []
        if "prediction_path" in data_dict[image].keys():
            for p in data_dict[image]["prediction_path"]:
                n_slices.append(get_n_slices(Path(p)))
            merged_data[image] = data_dict[image].copy()
            for key in merged_data[image].keys():
                if key == "prediction_path":
                    continue
                merged_data[image][key] = weighted_average(
                    merged_data[image][key], n_slices
                )
            # merged_data[image]["iou"] = weighted_average(
            #     merged_data[image]["iou"], n_slices
            # )
            # merged_data[image]["pixel_accuracy"] = weighted_average(
            #     merged_data[image]["pixel_accuracy"], n_slices
            # )
        else:
            continue

    return merged_data


def mask_sacrum_volume(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Modify the image based on the mask. Pixels above the uppermost cls pixel
    and below the lowermost cls pixel in the mask are set to zero.
    Args:
        - image (torch.Tensor):  The image of shape (BxCxHxW) or (BxHxW) to be masked with 0 along Height
        - mask (torch.Tensor): The mask of shape (BxCxHxW) or (BxHxW) to use along Depth
    Returns:
        - torch.Tensor: Modified image
    """
    modified_image = image.clone()
    modified_mask = mask.clone()
    if len(image.shape) == 3:
        modified_mask = modified_mask.unsqueeze(1)
        modified_image = modified_image.unsqueeze(1)
    mask_summed = sum_along_axis(modified_mask, axis=0)
    class_indices = torch.nonzero(mask_summed, as_tuple=True)
    # If there are no class pixels, return the original image
    if len(class_indices[0]) == 0:
        return image
    else:
        # Find the uppermost and lowermost indices where mask has class
        uppermost_idx = torch.min(class_indices[1]).item()
        lowermost_idx = torch.max(class_indices[1]).item()
        # Find the leftmost and rightmost indices where mask has class
        leftmost_idx = torch.min(class_indices[2]).item()
        rightmost_idx = torch.max(class_indices[2]).item()

        if lowermost_idx - uppermost_idx >= rightmost_idx - leftmost_idx:
            # The volume shows vertical vertebrae
            modified_image[:, :, :uppermost_idx, :] = 0
            modified_image[:, :, lowermost_idx + 1 :, :] = 0
        else:
            # The volume shows horizontal vertebrae
            modified_image[:, :, :, :leftmost_idx] = 0
            modified_image[:, :, :, rightmost_idx + 1 :] = 0

        if len(image.shape) == 3:
            modified_image = modified_image.squeeze(1)

        return modified_image
