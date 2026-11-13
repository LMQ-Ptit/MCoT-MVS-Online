import os
# Ensure headless matplotlib backend to avoid display errors in notebook/server
os.environ["MPLBACKEND"] = "Agg"

import urllib.request
from io import BytesIO
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

from sam_utils import SAMModel
from ram_utils import RAMModel
from dino_utils import DINOModel
from visualize_utils import draw_mask_and_segmentaion


class MaskGenerator:
    """
    Unified Segment Pipeline for MCoT-MVS (RAM++ + Grounding DINO + SAM 2.1).
    Supports batch dataset mask generation as well as single-image inference and CLIP feature pooling.
    """
    def __init__(
        self,
        ram_model: RAMModel,
        dino_model: DINOModel,
        sam_model: SAMModel,
        clip_model: Optional[Any] = None,
        workers: int = 4,
    ):
        self.ram = ram_model
        self.dino = dino_model
        self.sam = sam_model
        self.clip_model = clip_model
        self.workers = workers

    def generate_masks(self, image_path_list: List[str], save_dir: str, is_draw: bool = False):
        """Batch dataset processing (from original get_masks.py)"""
        batch_num = len(image_path_list) // self.workers + 1
        for i in tqdm(range(batch_num), desc="Generating Masks"):
            batch_image_path_list = image_path_list[
                i * self.workers : (i + 1) * self.workers
            ]
            if len(batch_image_path_list) == 0:
                break
            self.batch_generate_masks(batch_image_path_list, save_dir, is_draw)

    def batch_generate_masks(self, image_path_list: List[str], save_dir: str, is_draw: bool = False):
        pil_images = []
        for image_path in image_path_list:
            pil_image = Image.open(image_path).convert("RGB")
            pil_images.append(pil_image)

        # 1. RAM++ Tagging
        pred_tags = self.ram.get_batch_tags(pil_image_list=pil_images)

        # 2. Build Grounding Prompts (safe fallback to 'clothing.' if empty)
        grounding_texts = []
        for taglist in pred_tags:
            res = ""
            for tag in taglist:
                res += f"{tag}. "
            if not res.strip():
                res = "clothing."
            grounding_texts.append(res)

        # 3. Grounding DINO Bounding Boxes
        grounding_results = self.dino.get_batch_groundings(
            grounding_texts=grounding_texts, pil_image_list=pil_images
        )

        input_boxes_list = []
        for grounding_result in grounding_results:
            input_boxes = grounding_result["boxes"].cpu().numpy()
            input_boxes_list.append(input_boxes)

        # 4. SAM 2.1 Instance Masks
        masks_list, scores_list, logits_list = self.sam.get_batch_masks(
            input_boxes_list=input_boxes_list, pil_image_list=pil_images
        )

        # 5. Save outputs
        for i in range(len(image_path_list)):
            image_name = os.path.basename(image_path_list[i]).split(".")[0]
            output_dir = os.path.join(save_dir, "masks", image_name)
            os.makedirs(output_dir, exist_ok=True)

            draw_mask_and_segmentaion(
                ori_image_path=image_path_list[i],
                output_dir=output_dir,
                masks=masks_list[i],
                scores=scores_list[i],
                grounding_result=grounding_results[i],
                is_draw=is_draw,
            )
            saved_mask = masks_list[i].astype(np.int16)
            np.savez_compressed(os.path.join(output_dir, "masks.npz"), array=saved_mask)

    def pool_mask_features(self, pil_image: Image.Image, masks: np.ndarray) -> torch.Tensor:
        """
        Pools patch tokens from OpenCLIP ViT-H-14 based on binary instance masks.
        Returns tensor of shape (N, 1024), identical to extract_seg_feature_*.py.
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not attached to MaskGenerator. Pass clip_model in __init__.")

        if masks is None or len(masks) == 0:
            return torch.empty((0, 1024))

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        device = next(self.clip_model.parameters()).device
        cls, tokens = self.clip_model.visual_out(self.clip_model.preprocess_image(pil_image))

        features = []
        for i in range(masks.shape[0]):
            m = masks[i]
            if m.ndim == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            seg_mask = torch.from_numpy(m).to(dtype=torch.float32, device=device)
            image_mask = seg_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            H, W = image_mask.shape[-2:]
            stride_h = H // 16
            stride_w = W // 16

            if stride_h > 0 and stride_w > 0:
                pooled_mask = F.max_pool2d(
                    image_mask,
                    kernel_size=(stride_h, stride_w),
                    stride=(stride_h, stride_w),
                )
            else:
                pooled_mask = image_mask

            if pooled_mask.shape[-2:] != (16, 16):
                pooled_mask = F.interpolate(pooled_mask, size=(16, 16), mode="nearest")

            mask_patches = pooled_mask.reshape(1, -1, 1).long()
            non_zero_num = mask_patches.sum()
            image_feature = (tokens * mask_patches).sum(dim=1) / (non_zero_num + 1e-2)
            features.append(image_feature.squeeze(0))

        if len(features) == 0:
            features = [tokens.mean(dim=1).squeeze(0)]
        return torch.stack(features)  # (N, 1024)

    def warmup(self):
        """
        Pre-warm GPU kernels for RAM++, Grounding DINO, SAM 2.1, and OpenCLIP
        to eliminate initial forward-pass latency.
        """
        print("⏳ Pre-warming Segment Pipeline GPU kernels...")
        try:
            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            with torch.inference_mode():
                self.process_image(
                    image_input=dummy_img,
                    save_dir=None,
                    is_draw=False,
                    compute_clip_features=self.clip_model is not None,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("✅ Segment Pipeline warmup completed! Ready for real-time inference.")
        except Exception as e:
            print(f"⚠️ Segment warmup notice: {e}")

    def process_image(
        self,
        image_input: Union[str, Image.Image],
        save_dir: Optional[str] = None,
        is_draw: bool = True,
        compute_clip_features: bool = True
    ) -> Dict[str, Any]:
        """
        Single-image end-to-end inference method.
        Supports file path, web URL, or PIL.Image.
        """
        # Load PIL image
        if isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                req = urllib.request.Request(image_input, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    pil_img = Image.open(BytesIO(resp.read())).convert("RGB")
                img_name = "online_image"
                img_path = os.path.join(save_dir, f"{img_name}.jpg") if save_dir else None
                if img_path:
                    os.makedirs(save_dir, exist_ok=True)
                    pil_img.save(img_path)
            else:
                pil_img = Image.open(image_input).convert("RGB")
                img_name = os.path.basename(image_input).split(".")[0]
                img_path = image_input
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            img_name = "pil_image"
            img_path = os.path.join(save_dir, f"{img_name}.jpg") if save_dir else None
            if img_path:
                os.makedirs(save_dir, exist_ok=True)
                pil_img.save(img_path)
        else:
            raise ValueError("Input must be a file path, URL, or PIL Image.")

        # 1. RAM++
        pred_tags = self.ram.get_batch_tags([pil_img])[0]

        # 2. Grounding DINO
        grounding_text = ". ".join(pred_tags) + "." if pred_tags else "clothing."
        grounding_res = self.dino.get_batch_groundings([grounding_text], [pil_img])[0]
        boxes = grounding_res["boxes"].cpu().numpy()

        # 3. SAM 2.1
        masks_list, scores_list, _ = self.sam.get_batch_masks([boxes], [pil_img])
        masks = masks_list[0].astype(np.int16)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        scores = scores_list[0]

        # 4. Optional CLIP Feature Pooling
        seg_features = None
        if compute_clip_features and self.clip_model is not None:
            seg_features = self.pool_mask_features(pil_img, masks)

        # 5. Visualization & Saving
        annotated_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if not img_path:
                img_path = os.path.join(save_dir, f"{img_name}.jpg")
                pil_img.save(img_path)

            draw_mask_and_segmentaion(
                ori_image_path=img_path,
                output_dir=save_dir,
                masks=masks,
                scores=scores,
                grounding_result=grounding_res,
                is_draw=is_draw,
            )
            np.savez_compressed(os.path.join(save_dir, "masks.npz"), array=masks)
            if seg_features is not None:
                torch.save(seg_features.cpu(), os.path.join(save_dir, "seg_feature.pt"))

            annotated_path = os.path.join(save_dir, "grounded_sam2_annotated_image_with_mask.jpg")

        return {
            "tags": pred_tags,
            "boxes": boxes,
            "masks": masks,
            "scores": scores,
            "seg_features": seg_features,
            "annotated_image_path": annotated_path,
        }
