# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder, Attention_Fusion
from .tiny_vit_sam import TinyViT
from .transformer import TwoWayTransformer


class Sam_Mevis(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"  # information

    def __init__(
        self,
        args,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        attention_fusion: Attention_Fusion,
        pixel_mean: list[float] = [123.675, 116.28, 103.53],
        pixel_std: list[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.args = args
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.attention_fusion = attention_fusion
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        if args.if_LST_CNN:
            self.alpha = nn.Parameter(torch.zeros(1))

    @property
    def device(self) -> str:
        return self.pixel_mean.device

    # @torch.no_grad()
    def forward(
        self,
        batched_input: dict,
        multimask_output: bool = True,
        if_attention=False,
    ) -> dict[str, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
            batched_input (dict): A dictionary with the following keys. A prompt key can be
            excluded if it is not present.
                'images': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
                'points': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
                'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
                'atten_maps": (optional)(torch.Tensor) Batched attention maps, with
                shape Bx64x64.
                'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
                'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
            multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
            (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
                'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
        """
        with torch.no_grad():
            image_embeddings = self.image_encoder(batched_input["images"])
            if if_attention:
                image_embeddings = self.attention_fusion(
                    image_embeddings, batched_input["atten_maps"].unsqueeze(1)
                )

            if "points" in list(batched_input.keys()):
                points = (batched_input["points"], batched_input["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )
            # pe = torch.stack(
            #     [self.prompt_encoder.get_dense_pe() for _ in range(image_embeddings.shape[0])],
            #     dim=0,
            # ).squeeze(1)
            pe = self.prompt_encoder.get_dense_pe()
            
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return low_res_masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x