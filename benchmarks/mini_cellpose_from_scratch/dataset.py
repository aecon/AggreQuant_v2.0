"""PyTorch Dataset for training mini Cellpose.

Loads FarRed images and cellpose_cyto3 pseudo-GT masks, computes flow
targets on first access (cached to disk), and applies augmentations.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import tifffile

from dynamics import masks_to_flows


class CellFlowDataset(Dataset):
    """Dataset that pairs images with flow targets computed from masks.

    On first use, flow targets are computed via heat diffusion and cached
    as .npy files. Subsequent loads read from cache.

    Args:
        image_dir: Path to image directory (with category subdirectories).
        mask_dir: Path to cellpose_cyto3 mask directory (flat).
        crop_size: Size of random crops for training.
        augment: Whether to apply augmentations.
        flow_cache_dir: Where to cache computed flow targets.
    """

    def __init__(self, image_dir, mask_dir, crop_size=256, augment=True,
                 flow_cache_dir=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.crop_size = crop_size
        self.augment = augment

        if flow_cache_dir is None:
            flow_cache_dir = self.mask_dir.parent / "flow_cache"
        self.flow_cache_dir = Path(flow_cache_dir)
        self.flow_cache_dir.mkdir(parents=True, exist_ok=True)

        self.pairs = self._discover_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image-mask pairs found. Check paths:\n"
                f"  images: {self.image_dir}\n  masks: {self.mask_dir}"
            )

    def _discover_pairs(self):
        """Find matching image-mask pairs.

        Walks category subdirectories for FarRed images, matches to flat mask dir.
        """
        pairs = []
        for category_dir in sorted(self.image_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for img_path in sorted(category_dir.glob("*wv 631 - FarRed*.tif")):
                mask_path = self.mask_dir / img_path.name
                if mask_path.exists():
                    pairs.append({
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "name": img_path.stem,
                    })
        return pairs

    def _get_or_compute_flows(self, idx):
        """Load or compute flow targets for a given sample."""
        name = self.pairs[idx]["name"]
        cache_path = self.flow_cache_dir / f"{name}.npy"

        if cache_path.exists():
            data = np.load(cache_path)
            flows = data[:2]
            cellprob = data[2]
        else:
            mask = tifffile.imread(str(self.pairs[idx]["mask_path"]))
            mask = mask.astype(np.int32)
            flows = masks_to_flows(mask)
            cellprob = (mask > 0).astype(np.float32)
            # Save as (3, H, W): [flow_y, flow_x, cellprob]
            data = np.concatenate([flows, cellprob[None]], axis=0)
            np.save(cache_path, data.astype(np.float32))

        return flows, cellprob

    def _normalize_image(self, img):
        """Percentile normalization: 1st-99.8th percentile stretch to [0, 1]."""
        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99.8)
        if p_high - p_low < 1e-6:
            return np.zeros_like(img, dtype=np.float32)
        img = (img.astype(np.float32) - p_low) / (p_high - p_low)
        return np.clip(img, 0, 1)

    def _random_crop(self, img, flows, cellprob):
        """Extract a random crop_size x crop_size patch."""
        _, H, W = img.shape
        cs = self.crop_size

        if H < cs or W < cs:
            # Pad if image is smaller than crop size
            pad_h = max(cs - H, 0)
            pad_w = max(cs - W, 0)
            img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)))
            flows = np.pad(flows, ((0, 0), (0, pad_h), (0, pad_w)))
            cellprob = np.pad(cellprob, ((0, pad_h), (0, pad_w)))
            _, H, W = img.shape

        y0 = np.random.randint(0, H - cs + 1)
        x0 = np.random.randint(0, W - cs + 1)

        img = img[:, y0:y0 + cs, x0:x0 + cs]
        flows = flows[:, y0:y0 + cs, x0:x0 + cs]
        cellprob = cellprob[y0:y0 + cs, x0:x0 + cs]

        return img, flows, cellprob

    def _augment_sample(self, img, flows, cellprob):
        """Random flips and 90-degree rotations with correct flow transforms."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()
            flows = flows[:, :, ::-1].copy()
            cellprob = cellprob[:, ::-1].copy()
            flows[1] = -flows[1]  # negate flow_x

        # Random vertical flip
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()
            flows = flows[:, ::-1, :].copy()
            cellprob = cellprob[::-1, :].copy()
            flows[0] = -flows[0]  # negate flow_y

        # Random 90-degree rotation (0, 1, 2, or 3 times)
        k = np.random.randint(0, 4)
        if k > 0:
            # np.rot90 on the last two dims
            img = np.rot90(img, k, axes=(1, 2)).copy()
            flows = np.rot90(flows, k, axes=(1, 2)).copy()
            cellprob = np.rot90(cellprob, k, axes=(0, 1)).copy()
            # Rotation transforms for flows:
            # 90 CW: (fy, fx) -> (fx, -fy)
            # 180:    (fy, fx) -> (-fy, -fx)
            # 270 CW: (fy, fx) -> (-fx, fy)
            if k == 1:
                flows = np.stack([flows[1], -flows[0]])
            elif k == 2:
                flows = np.stack([-flows[0], -flows[1]])
            elif k == 3:
                flows = np.stack([-flows[1], flows[0]])

        return img, flows, cellprob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return (image, target) tensors.

        image: (1, crop_size, crop_size) float32
        target: (3, crop_size, crop_size) float32 = [flow_y, flow_x, cellprob]
        """
        # Load image
        img = tifffile.imread(str(self.pairs[idx]["image_path"]))
        img = self._normalize_image(img)
        img = img[None]  # (1, H, W)

        # Load flows
        flows, cellprob = self._get_or_compute_flows(idx)

        # Crop
        img, flows, cellprob = self._random_crop(img, flows, cellprob)

        # Augment
        if self.augment:
            img, flows, cellprob = self._augment_sample(img, flows, cellprob)

        # Stack target
        target = np.concatenate([flows, cellprob[None]], axis=0)  # (3, H, W)

        return torch.from_numpy(img), torch.from_numpy(target)
