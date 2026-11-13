import os
os.environ["MPLBACKEND"] = "Agg"
import json
from typing import Optional
from sam_utils import SAMModel
from ram_utils import RAMModel
from dino_utils import DINOModel
from pipeline import MaskGenerator


def load_segment_pipeline(
    device: str = "cuda:0",
    ram_pretrained: Optional[str] = None,
    ram_tag_path: Optional[str] = None,
    dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
    sam_checkpoint_dir: Optional[str] = None,
    load_clip: bool = False,
    workers: int = 1,
    warmup: bool = True,
) -> MaskGenerator:
    """
    Factory function to initialize and return a complete MaskGenerator pipeline.
    Resolves model checkpoint paths automatically relative to segment/ directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Resolve paths
    if not ram_pretrained:
        ram_pretrained = os.path.join(current_dir, "ram_models", "ram_plus_swin_large_14m.pth")
    if not ram_tag_path:
        ram_tag_path = os.path.join(current_dir, "ram_models", "ram_tag_list.txt")
    if not sam_checkpoint_dir:
        sam_checkpoint_dir = os.path.join(current_dir, "sam2", "checkpoints")

    # 2. Load models
    print(f"⏳ Loading Segment Pipeline on [{device}]...")
    ram = RAMModel(
        pretrained=ram_pretrained,
        tag_file_path=ram_tag_path,
        device=device,
    )
    dino = DINOModel(model_id=dino_model_id, device=device)
    sam = SAMModel(sam_abs_dir_path=sam_checkpoint_dir, device=device)

    # 3. Optional OpenCLIP ViT-H-14 for feature extraction
    clip_model = None
    if load_clip:
        print("⏳ Loading OpenCLIP ViT-H-14 for mask feature pooling...")
        from extract_seg_feature_fiq import Clip
        clip_model = Clip().to(device).eval()

    pipeline = MaskGenerator(
        ram_model=ram,
        dino_model=dino,
        sam_model=sam,
        clip_model=clip_model,
        workers=workers,
    )

    # 4. Optional Pre-warming
    if warmup:
        pipeline.warmup()

    print("✅ Segment Pipeline loaded successfully!")
    return pipeline


def cirr():
    image_path_list = []
    cirr_base_path = "../data/cirr/cirr_dataset"
    with open(os.path.join(cirr_base_path, 'cirr/image_splits/split.rc2.test1.json'), 'r') as f:
        test_split = json.load(f)
    with open(os.path.join(cirr_base_path, 'cirr/image_splits/split.rc2.val.json'), 'r') as f:
        val_split = json.load(f)
    with open(os.path.join(cirr_base_path, 'cirr/image_splits/split.rc2.train.json'), 'r') as f:
        train_split = json.load(f)
    res = {}
    res.update(test_split)
    res.update(val_split)
    res.update(train_split)
    for value in res.values():
        image_path_list.append(os.path.join(cirr_base_path, value))

    maskGenerator = load_segment_pipeline(device="cuda", workers=1)
    maskGenerator.generate_masks(image_path_list=image_path_list, save_dir='../data/cirr/segment/masks', is_draw=False)


def fiq():
    image_path_list = []
    fiq_base_path = "../data/fiq/fashionIQ_dataset"
    image_name_list = []
    dress_types = ['dress', 'shirt', 'toptee']
    splits = ['val', 'train']
    for dress_type in dress_types:
        for split in splits:
            with open(os.path.join(fiq_base_path, f'image_splits/split.{dress_type}.{split}.json'), 'r') as f:
                data = json.load(f)
                image_name_list.extend(data)
    image_name_list = list(set(image_name_list))

    for image_name in image_name_list:
        image_path_list.append(os.path.join(fiq_base_path, 'images', image_name + '.png'))

    maskGenerator = load_segment_pipeline(device="cuda", workers=1)
    maskGenerator.generate_masks(image_path_list=image_path_list, save_dir='../data/fiq/segment/masks', is_draw=False)


if __name__ == "__main__":
    cirr()
    fiq()
