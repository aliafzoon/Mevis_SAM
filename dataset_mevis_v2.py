import json
import random
from pathlib import Path
import warnings

import monai
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from funcs import min_max_normalize, generate_click_prompt, mask_sacrum_volume
from models.sam.utils.transforms import ResizeLongestSide


class MRI_dataset_batched(Dataset):
    def __init__(
        self,
        args,
        data_file: Path,
        batch_size: int = 20,
        phase: str = "train",
        operation_mode: str = "queue",  # ["queue" or "single"]
        mask_out_size: int = 256,
        attention_size: int = 64,
        crop: bool = False,
        crop_size: int = 1024,
        cls: int = 1,
        if_prompt: bool = False,
        prompt_type: str = "points",
        if_attention_map: bool | None = None,
        device: str = "cpu",
    ):
        # TODO:
        super().__init__()
        self.batch_size = batch_size
        self.phase = phase
        self.operation_mode = operation_mode
        self.mask_out_size = mask_out_size
        self.attention_size = attention_size
        self.crop = crop
        self.crop_size = crop_size
        self.args = args
        self.cls = cls
        self.if_prompt = if_prompt
        self.prompt_type = prompt_type
        self.if_attention_map = if_attention_map
        self.device = device

        # available prompt types
        if if_prompt:
            assert prompt_type in ["points"], "Prompt type unknown. Skipping prompts..."
        assert Path(data_file).is_file(), "Data file does not exist."    

        self.data_dict = json.load(open(data_file, "r"))
        self.images_list = list(self.data_dict.keys())
        # self.transform_normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.queue = self._create_queue()

        if phase == "train":
            self.aug_img = [
                transforms.RandomEqualize(p=0.1),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                ),
                transforms.RandomAdjustSharpness(0.5, p=0.5),
            ]
            self.transform_spatial = transforms.Compose(
                [
                    transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.2)),
                    transforms.RandomRotation(45),
                ]
            )
            transform_img = [transforms.ToTensor()]
        else:
            transform_img = [transforms.ToTensor()]
        self.transform_img = transforms.Compose(transform_img)

    def __len__(self):
        if self.operation_mode == "queue":
            # return  based on queue size
            data_len = len(self.queue) // self.batch_size
            if len(self.queue) % self.batch_size != 0:
                data_len += 1
        else:
            # return number of image volumes
            data_len = len(self.images_list)

        return data_len

    def get_que_batch(self, index: int):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        if end_index >= len(self.queue):
            batch_tuples = self.queue[start_index:]
        else:
            batch_tuples = self.queue[start_index:end_index]

        grouped = {}
        for name, slice in batch_tuples:
            if name in grouped:
                grouped[name].append(slice)
            else:
                grouped[name] = [slice]
        volume_data = []
        for name, slices in grouped.items():
            the_dict = self.data_dict[name].copy()
            the_dict["slices"] = slices
            volume_data.append(the_dict)

        if len(batch_tuples) == 0:
            print(
                f"mri index overshoot, calling {index} start {start_index}, end {end_index}, total {len(self.queue)}"
            )

        for i, vol in enumerate(volume_data):
            data = self._process_volume(vol)
            if i == 0:
                loaded_batch = data
                # start of each image volume if concatenated
                loaded_batch["cat_indexes"] = [0]
            else:
                loaded_batch["cat_indexes"].append(loaded_batch["images"].shape[0])
                for key in data.keys():
                    if key in ["images", "masks", "atten_maps"]:
                        loaded_batch[key] = torch.cat([loaded_batch[key], data[key]], dim=0)
                    else:
                        loaded_batch[key] = loaded_batch[key] + data[key]

        return loaded_batch

    def _create_queue(self) -> list[tuple]:
        queue = []
        for img in self.images_list:
            for slice_num in self.data_dict[img]["slices"]:
                queue.append((img, slice_num))
        return queue

    def _process_volume(self, sample: dict) -> dict:
        out_dict = {}
        img_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])
        slices = [int(x) for x in sample["slices"]]  # all slices for this image
        attentions = sample["attention_path"]
        if self.if_attention_map:
            assert all([Path(x).is_file() for x in attentions.values()])
        if img_path.is_file():
            img_vol = np.array(
                sitk.GetArrayFromImage(sitk.ReadImage(img_path)), dtype=float
            )
            img_vol = np.array(
                (img_vol - img_vol.min())
                / (img_vol.max() - img_vol.min() + 1e-8)
                * 255,
                dtype=np.uint8,
            )
        else:
            print(f"Error reading the image {img_path}")

        if mask_path.is_file():
            msk = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        else:
            print(f"Error reading the mask {mask_path}")

        out_dict["slices"] = slices
        out_dict["original_size"] = [sample["original_size"]]
        out_dict["image_name"] = [img_path.name]

        output_attention = torch.zeros(
            (len(slices), self.attention_size, self.attention_size)
        )
        smallest_dimension_index = np.argmin(img_vol.shape)
        img_vol = (
            torch.from_numpy(
                np.take(img_vol, indices=slices, axis=smallest_dimension_index)
            )
            .to(self.device)
            .to(torch.float32)
        )
        msk = torch.from_numpy(
            np.take(msk, indices=slices, axis=smallest_dimension_index)
        ).to(self.device)
        if smallest_dimension_index != 0:
            img_vol = torch.moveaxis(img_vol, smallest_dimension_index, 0)
            msk = torch.moveaxis(msk, smallest_dimension_index, 0)
        img_vol = img_vol.unsqueeze(1).repeat(1, 3, 1, 1)
        msk = msk.unsqueeze(1)
        img_vol = ResizeLongestSide(
            target_length=self.args.image_size
        ).apply_image_torch(img_vol)
        img_vol = img_vol.to(torch.uint8)
        msk = self.resize_mask(msk, size=(img_vol.shape[2], img_vol.shape[3]))
        img_vol = self.pad(img_vol, size=self.args.image_size)
        msk = self.pad(msk, size=self.args.image_size)

        if self.crop:
            img_vol = self.pad(img_vol, size=self.args.crop_size)
            msk = self.pad(msk, size=self.args.crop_size)
            for i in range(img_vol.shape[0]):
                t, l, h, w = transforms.RandomCrop.get_params(
                    img_vol[i], (self.crop_size, self.crop_size)
                )
                img_vol[i] = transforms.functional.crop(img_vol[i], t, l, h, w)
                msk[i] = transforms.functional.crop(msk[i], t, l, h, w)

        if self.phase == "train":
            # add random optimazition
            for i in range(img_vol.shape[0]):
                aug_img_fuc = transforms.RandomChoice(self.aug_img)
                img_vol[i] = aug_img_fuc(img_vol[i])
                random_transform = monai.transforms.OneOf(
                    [
                        monai.transforms.RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
                        monai.transforms.RandKSpaceSpikeNoise(
                            prob=0.5, intensity_range=None, channel_wise=True
                        ),
                        monai.transforms.RandBiasField(degree=3),
                        monai.transforms.RandGibbsNoise(prob=0.5, alpha=(0.0, 1.0)),
                    ],
                    weights=[0.3, 0.3, 0.2, 0.2],
                )
                img_vol[i] = random_transform(img_vol[i]).as_tensor()
        else:
            img_vol = img_vol.to(torch.float32)
            adjust_contrast = monai.transforms.AdjustContrast(gamma=0.8)
            for i in range(img_vol.shape[0]):
                if img_vol[i].float().mean() < 0.05:
                    img_vol[i] = min_max_normalize(img_vol[i])
                    img_vol[i] = adjust_contrast(img_vol[i])
        
        msk = msk.to(torch.int)
        mask_cls = (msk == self.cls).to(torch.float32)
        img_vol = img_vol.to(torch.float32)

        img_vol = (img_vol - img_vol.min()) / (img_vol.max() - img_vol.min() + 1e-8)
        img_vol = mask_sacrum_volume(image=img_vol, mask=mask_cls)

        if self.if_attention_map:
            for i in range(len(slices)):
                output_attention[i] = torch.from_numpy(
                    np.load(attentions[str(slices[i])])
                )
            output_attention = output_attention.to(self.device)

        if self.phase == "train" and (not self.if_attention_map == None):
            mask_cls = mask_cls.repeat(1, 3, 1, 1)
            attention_large = output_attention.unsqueeze(1).repeat(1, 3, 1, 1)
            attention_large = transforms.functional.resize(
                attention_large, self.args.image_size
            )
            for i in range(img_vol.shape[0]):
                # both_targets = torch.cat((img_vol[i].unsqueeze(0), mask_cls[i].unsqueeze(0)), 0)
                all_targets = torch.cat(
                    (
                        img_vol[i : i + 1],
                        mask_cls[i : i + 1],
                        attention_large[i : i + 1],
                    ),
                    0,
                )
                transformed_targets = self.transform_spatial(all_targets)
                img_vol[i] = transformed_targets[0]
                mask_cls[i] = transformed_targets[1].to(torch.int).to(torch.float32)
                output_attention[i] = transforms.functional.resize(
                    transformed_targets[2], self.attention_size
                )[0, :, :]
            mask_cls = mask_cls[:, 0:1, :, :]
        # transform normalize
        img_vol = (img_vol - self.mean) / self.std

        out_dict["images"] = img_vol
        out_dict["masks"] = mask_cls
        out_dict["atten_maps"] = output_attention

        return out_dict
    
    def add_point_prompt(self, batch):
        images = batch["images"] 
        masks = batch["masks"]
        mask_size = masks.shape[-1]
        if mask_size != images.shape[-1]:
            masks = self.resize_mask(masks, images.shape[-2:])
        _, points, masks = generate_click_prompt(images, masks)
        point_labels = torch.ones(masks.size(0), 1, dtype=torch.int, device=self.device)
        if mask_size != images.shape[-1]:
            masks = self.resize_mask(masks, (mask_size, mask_size))
        batch["points"] = points.unsqueeze(1)
        batch["point_labels"] = point_labels
        batch["masks"] = masks
        return batch

    def inverse_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.mul(self.std).add(self.mean)

    def resize_mask(self, mask_tensor: torch.Tensor, size: tuple) -> torch.Tensor:
        """
            Resizes a batch of CxHxW tensors using the nearest neighbor method.
        Args:
            mask_tensor (torch.Tensor): Input tensor of shape (batch, channel, height, width).
            size (tuple): Target size as a tuple (new_height, new_width).
        Returns:
            torch.Tensor: Resized batch of tensor of shape (batch, channel, height, width).
        """
        new_height, new_width = size
        batch, channel, original_height, original_width = mask_tensor.shape
        # Calculate the ratio of the old dimensions to the new dimensions
        row_ratio, col_ratio = original_height / new_height, original_width / new_width
        # Create the new index arrays for the resized image
        row_indices = (
            torch.arange(new_height, device=mask_tensor.device) * row_ratio
        ).long()
        col_indices = (
            torch.arange(new_width, device=mask_tensor.device) * col_ratio
        ).long()
        row_indices = row_indices.unsqueeze(1).repeat(1, new_width)
        col_indices = col_indices.unsqueeze(0).repeat(new_height, 1)
        resized_mask_tensor = mask_tensor[:, :, row_indices, col_indices]

        return resized_mask_tensor

    def pad(self, images: torch.Tensor, size: int) -> torch.Tensor:
        """
            Pads a batch of CxHxW torch.Tensors to a specified size.
        Args:
            images (torch.Tensor): Input tensor of shape (batch, channel, height, width).
            size (int): Target size to pad the images to.
        Returns:
            torch.Tensor: Padded batch of images.
        """
        batch, channel, im_h, im_w = images.shape
        diff_w = max(0, size - im_w)
        diff_h = max(0, size - im_h)

        padding = (
            diff_w // 2,
            diff_h // 2,
            diff_w - diff_w // 2,
            diff_h - diff_h // 2,
        )
        padded_images = torch.zeros(
            (batch, channel, size, size), dtype=images.dtype, device=images.device
        )
        # Calculate the indices for placing the original images in the padded tensor
        padded_images[
            :, :, padding[1] : padding[1] + im_h, padding[0] : padding[0] + im_w
        ] = images

        return padded_images

    def __getitem__(self, index) -> dict:
        # Generate random click points
        if self.operation_mode == "queue":
            batch = self.get_que_batch(index)
        else:
            sample = self.data_dict[self.images_list[index]]
            batch = self._process_volume(sample)

        if self.if_prompt and torch.bernoulli(torch.tensor([self.args.prompt_probability])).bool().item():
            if self.prompt_type == "points":
                batch = self.add_point_prompt(batch)
        if batch["masks"].shape[-1] != self.mask_out_size:
            batch["masks"] = self.resize_mask(batch["masks"], (self.mask_out_size, self.mask_out_size))

        return batch

    def on_epoch_end(self):
        random.shuffle(self.images_list)
        self.queue = self._create_queue()
