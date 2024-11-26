import sys

sys.path.append("../")
import cv2
import monai
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.measure import label
from torchvision import transforms

import cfg
from funcs import *
from models.sam.modeling.prompt_encoder import Attention_Fusion

args = cfg.parse_args()


def drawContour(m, s, RGB, size, a=0.8):
    """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'"""
    # Fill contour "c" with white, make all else black

    # ratio = int(255/np.max(s))
    # s = np.uint(s*ratio)

    # Find edges of this contour and make into Numpy array
    contours, _ = cv2.findContours(np.uint8(s), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    m_old = m.copy()
    # Paint locations of found edges in color "RGB" onto "main"
    cv2.drawContours(m, contours, -1, RGB, size)
    m = cv2.addWeighted(np.uint8(m), a, np.uint8(m_old), 1 - a, 0)
    return m


def IOU(pm, gt):
    a = np.sum(np.bitwise_and(pm, gt))
    b = np.sum(pm) + np.sum(gt) - a + 1e-8
    return a / b


def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def remove_small_objects(array_2d, min_size=30):
    """
    Removes small objects from a 2D array using only NumPy.

    :param array_2d: Input 2D array.
    :param min_size: Minimum size of objects to keep.
    :return: 2D array with small objects removed.
    """
    # Label connected components
    structure = np.ones((3, 3), dtype=int)  # Define connectivity
    labeled, ncomponents = label(array_2d, structure)

    # Iterate through labeled components and remove small ones
    for i in range(1, ncomponents + 1):
        locations = np.where(labeled == i)
        if len(locations[0]) < min_size:
            array_2d[locations] = 0

    return array_2d


def create_box_mask(boxes, imgs):
    b, _, w, h = imgs.shape
    box_mask = torch.zeros((b, w, h))
    for k in range(b):
        k_box = boxes[k]
        for box in k_box:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            box_mask[k, y1:y2, x1:x2] = 1
    return box_mask


def resize_torch_volume(
    tensor, new_size: tuple[int, int, int] = (64, 64, 64), mode="trilinear"
):
    """
    Resize a 3D volume using PyTorch.

    Args:
    - volume (torch.Tensor): The input volume tensor of shape (D, H, W) or (B, D, H, W).
    - new_size (list or tuple): The desired output size as (D, H, W).
    - mode (str): The interpolation mode. Default is 'trilinear'.

    Returns:
    - torch.Tensor: The resized volume tensor of shape (D, H, W) or (B, D, H, W).
    """
    is_batched = True
    if len(tensor.shape) == 3:
        is_batched = False
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension
    elif len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(1)  # Add channel dimension
    # Resize the volume
    resized_tensor = F.interpolate(
        tensor, size=new_size, mode=mode, align_corners=False
    ).squeeze(1)
    # Remove the batch dimension if it was added
    if not is_batched:
        resized_tensor = resized_tensor.squeeze(0)
    return resized_tensor


# Calculate the percentile values
def torch_percentile(tensor, percentile):
    k = 1 + round(0.01 * float(percentile) * (tensor.numel() - 1))
    return tensor.reshape(-1).kthvalue(k).values.item()


def pred_attention(tensor, vnet, slice_id, device):
    """
    Edited: Switched from tio to torch.tensor
            input is 3d instead of 4d
            depth is at axis 0 instead of 2
    """

    class Normalize3D:
        """Normalize a tensor to a specified mean and standard deviation."""

        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, x):
            # Normalize x
            return (x - self.mean) / self.std

    def prob_rescale(prob, x_thres=0.05, y_thres=0.8, eps=1e-3):
        grad_1 = y_thres / x_thres
        grad_2 = (1 - y_thres) / (1 - x_thres)

        mask_eps = prob <= eps
        mask_1 = (eps < prob) & (prob <= x_thres)
        mask_2 = prob > x_thres
        prob[mask_1] = prob[mask_1] * grad_1
        prob[mask_2] = (prob[mask_2] - x_thres) * grad_2 + y_thres
        prob[mask_eps] = 0
        return prob

    def view_attention_2d(mask_volume, axis=0, eps=0.1):
        mask_eps = mask_volume <= eps
        mask_volume[mask_eps] = 0
        attention = np.sum(mask_volume, axis=axis)
        return (attention) / (np.max(attention) + 1e-8)

    norm_transform = transforms.Compose([Normalize3D(0.5, 0.5)])
    depth_image = tensor.shape[0]
    tensor = resize_torch_volume(tensor, (64, 64, 64))
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # add channel and batch dim
    tensor = norm_transform(tensor).float().to(device)
    with torch.set_grad_enabled(False):
        pred_mask = vnet(tensor)
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = pred_mask.detach().cpu().numpy()

    # the slice id after rescale to 64*64*64
    slice_id_reshape = int(slice_id * 64 / depth_image)
    slice_min = max(slice_id_reshape - 8, 0)
    slice_max = min(slice_id_reshape + 8, 64)
    return prob_rescale(
        view_attention_2d(np.squeeze(pred_mask[:, :, slice_min:slice_max, :, :]))
    )


def evaluate_1_volume_withattention(
    image_vol, model, device, slice_id=None, target_spacing=None, atten_map=None
):
    image_vol.data = image_vol.data / (image_vol.data.max() * 1.0)
    voxel_spacing = image_vol.spacing
    if target_spacing and (voxel_spacing != target_spacing):
        resample = tio.Resample(target_spacing, image_interpolation="nearest")
        image_vol = resample(image_vol)
    image_vol = image_vol.data[0]
    slice_num = image_vol.shape[2]
    if slice_id is not None:
        if slice_id > slice_num:
            slice_id = -1
    else:
        slice_id = slice_num // 2
    img_arr = image_vol[:, :, slice_id]
    img_arr = np.array(
        (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 0.00001) * 255,
        dtype=np.uint8,
    )
    img_3c = np.tile(img_arr[:, :, None], [1, 1, 3])
    img = Image.fromarray(img_3c, "RGB")
    Pil_img = img.copy()
    img = transforms.Resize((1024, 1024))(img)
    transform_img = transforms.Compose([transforms.ToTensor()])
    img = transform_img(img)
    img = min_max_normalize(img)
    if img.mean() < 0.1:
        img = monai.transforms.AdjustContrast(gamma=0.8)(img)
    imgs = torch.unsqueeze(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            img
        ),
        0,
    ).to(device)

    with torch.no_grad():
        img_emb = model.image_encoder(imgs)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        if not atten_map is None:
            # fuse the depth direction attention
            img_emb = model.attention_fusion(img_emb, atten_map)
        pred, _ = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        pred = pred[:, 1, :, :]
    ori_img = inverse_normalize(imgs.cpu()[0])
    return ori_img, pred, voxel_spacing, Pil_img, slice_id
