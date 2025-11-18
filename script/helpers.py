import logging
import random
import sys
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def collate_images(batch: List[dict]) -> dict:
    """Collate for Stage1 image-only datasets.
    - Pads images in a batch to the same H,W if needed and stacks to (B,C,H,W).
    - Returns `images` tensor and `cond_ids` (from `cond_id` or `type_id`).
    """
    # filter out None entries
    batch = [b for b in batch if b is not None]
    if not batch:
        return {"images": torch.empty(0), "cond_ids": torch.empty(0, dtype=torch.long)}

    imgs = [b["image"] for b in batch]
    # If all shapes equal, stack directly
    same_size = all(img.shape == imgs[0].shape for img in imgs)
    if same_size:
        images = torch.stack(imgs)
    else:
        # pad to max H,W
        h_max = max(img.shape[-2] for img in imgs)
        w_max = max(img.shape[-1] for img in imgs)
        padded = []
        for img in imgs:
            _, h, w = img.shape
            pad_h = h_max - h
            pad_w = w_max - w
            pad = (0, pad_w, 0, pad_h)
            padded.append(F.pad(img, pad, value=0.0))
        images = torch.stack(padded)

    # cond ids
    if "cond_id" in batch[0]:
        cond_ids = torch.tensor([b["cond_id"] for b in batch], dtype=torch.long)
    else:
        cond_ids = torch.tensor([b.get("type_id", 0) for b in batch], dtype=torch.long)

    return {"images": images, "cond_ids": cond_ids}
