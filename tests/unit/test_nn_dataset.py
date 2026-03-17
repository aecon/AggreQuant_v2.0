"""Tests for nn/datatools/dataset.py — extract_patches, PatchDataset, create_dataloaders."""

import numpy as np
import pytest
import tifffile
import torch
import torchvision.transforms.v2 as v2

from aggrequant.nn.datatools.dataset import (
    extract_patches,
    PatchDataset,
    create_dataloaders,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_patch_dir(tmp_path, n_patches=10, patch_size=32):
    """Create a fake patch directory with images/ and masks/ subdirs."""
    img_dir = tmp_path / "patches" / "images"
    mask_dir = tmp_path / "patches" / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for i in range(n_patches):
        img = np.random.randint(0, 1000, (patch_size, patch_size), dtype=np.uint16)
        mask = np.random.choice([0, 1], (patch_size, patch_size)).astype(np.uint8)
        name = f"patch_{i:03d}.tif"
        tifffile.imwrite(img_dir / name, img)
        tifffile.imwrite(mask_dir / name, mask)

    return tmp_path / "patches"


def _make_image_mask_dirs(tmp_path, images, patch_size=128):
    """Create image_dir and mask_dir with synthetic image/mask pairs.

    Arguments:
        images: list of (stem, height, width) tuples describing each image.
    """
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    for stem, h, w in images:
        img = np.random.randint(0, 1000, (h, w), dtype=np.uint16)
        mask = np.random.choice([0, 1, 2], (h, w)).astype(np.uint16)
        tifffile.imwrite(img_dir / f"{stem}.tif", img)
        tifffile.imwrite(mask_dir / f"{stem}_mask.tif", mask)

    return img_dir, mask_dir


# ===========================================================================
# extract_patches
# ===========================================================================


class TestExtractPatches:
    """Tests for extract_patches."""

    def test_basic_extraction(self, tmp_path):
        """Two 256x256 images → correct patch count and directory structure."""
        img_dir, mask_dir = _make_image_mask_dirs(
            tmp_path,
            [("img_001", 256, 256), ("img_002", 256, 256)],
        )
        out_dir = tmp_path / "patches"

        n = extract_patches(img_dir, mask_dir, out_dir, patch_size=128)

        # 256/128 = 2 rows x 2 cols = 4 patches per image, 2 images = 8
        assert n == 8
        assert (out_dir / "images").is_dir()
        assert (out_dir / "masks").is_dir()
        assert len(list((out_dir / "images").glob("*.tif"))) == 8
        assert len(list((out_dir / "masks").glob("*.tif"))) == 8

    def test_patch_filenames(self, tmp_path):
        """Patch filenames encode source image and position."""
        img_dir, mask_dir = _make_image_mask_dirs(
            tmp_path, [("img_001", 256, 256)]
        )
        out_dir = tmp_path / "patches"
        extract_patches(img_dir, mask_dir, out_dir, patch_size=128)

        names = sorted(f.name for f in (out_dir / "images").glob("*.tif"))
        expected = [
            "img_001_y000_x000.tif",
            "img_001_y000_x001.tif",
            "img_001_y001_x000.tif",
            "img_001_y001_x001.tif",
        ]
        assert names == expected

    def test_incomplete_edge_patches_skipped(self, tmp_path):
        """Image not divisible by patch_size → only complete patches saved."""
        img_dir, mask_dir = _make_image_mask_dirs(
            tmp_path, [("img_001", 300, 200)]
        )
        out_dir = tmp_path / "patches"

        n = extract_patches(img_dir, mask_dir, out_dir, patch_size=128)

        # 300//128=2 rows, 200//128=1 col → 2 patches
        assert n == 2

    def test_mask_suffix_matching(self, tmp_path):
        """Finds mask via suffix convention (img_001 → img_001_mask.tif)."""
        img_dir, mask_dir = _make_image_mask_dirs(
            tmp_path, [("img_001", 128, 128)]
        )
        out_dir = tmp_path / "patches"

        n = extract_patches(img_dir, mask_dir, out_dir, patch_size=128)
        assert n == 1

    def test_mask_same_name_fallback(self, tmp_path):
        """Falls back to same-name matching when suffix doesn't exist."""
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        img = np.zeros((128, 128), dtype=np.uint16)
        mask = np.zeros((128, 128), dtype=np.uint8)
        tifffile.imwrite(img_dir / "img_001.tif", img)
        # No _mask suffix — same filename
        tifffile.imwrite(mask_dir / "img_001.tif", mask)

        out_dir = tmp_path / "patches"
        n = extract_patches(img_dir, mask_dir, out_dir, patch_size=128)
        assert n == 1

    def test_missing_mask_skipped(self, tmp_path):
        """Image without a mask is skipped (no error)."""
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        # Image with a matching mask
        tifffile.imwrite(img_dir / "good.tif", np.zeros((128, 128), dtype=np.uint16))
        tifffile.imwrite(mask_dir / "good_mask.tif", np.zeros((128, 128), dtype=np.uint8))

        # Image without any mask
        tifffile.imwrite(img_dir / "orphan.tif", np.zeros((128, 128), dtype=np.uint16))

        out_dir = tmp_path / "patches"
        n = extract_patches(img_dir, mask_dir, out_dir, patch_size=128)
        assert n == 1  # only 'good' extracted

    def test_empty_directory_raises(self, tmp_path):
        """No images matching pattern → ValueError."""
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        with pytest.raises(ValueError, match="No images found"):
            extract_patches(img_dir, mask_dir, tmp_path / "out", patch_size=128)


# ===========================================================================
# PatchDataset
# ===========================================================================


class TestPatchDataset:
    """Tests for PatchDataset."""

    def test_len_and_getitem(self, tmp_path):
        """Dataset length matches file count; items are (image, mask) tensors."""
        patch_dir = _make_patch_dir(tmp_path, n_patches=5)
        image_files = sorted((patch_dir / "images").glob("*.tif"))

        ds = PatchDataset(image_files, mask_dir=patch_dir / "masks")

        assert len(ds) == 5
        img, mask = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert img.shape == (1, 32, 32)
        assert mask.shape == (1, 32, 32)
        assert img.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_normalization_on(self, tmp_path):
        """With normalize=True, image values are in [0, 1]."""
        patch_dir = _make_patch_dir(tmp_path, n_patches=1)
        image_files = sorted((patch_dir / "images").glob("*.tif"))

        ds = PatchDataset(image_files, mask_dir=patch_dir / "masks", normalize=True)
        img, _ = ds[0]

        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_normalization_off(self, tmp_path):
        """With normalize=False, image is raw float32 (not clipped to [0, 1])."""
        patch_dir = _make_patch_dir(tmp_path, n_patches=1)
        image_files = sorted((patch_dir / "images").glob("*.tif"))

        ds = PatchDataset(image_files, mask_dir=patch_dir / "masks", normalize=False)
        img, _ = ds[0]

        assert img.dtype == torch.float32
        # uint16 data with values up to ~1000 → raw float should exceed 1.0
        assert img.max() > 1.0

    def test_mask_binarization(self, tmp_path):
        """Mask with multiple label values is binarized to {0.0, 1.0}."""
        img_dir = tmp_path / "patches" / "images"
        mask_dir = tmp_path / "patches" / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        tifffile.imwrite(img_dir / "p.tif", np.ones((32, 32), dtype=np.uint16))
        # Mask with labels 0, 1, 2, 5
        mask = np.array([[0, 1], [2, 5]], dtype=np.uint16)
        mask = np.tile(mask, (16, 16))  # 32x32
        tifffile.imwrite(mask_dir / "p.tif", mask)

        ds = PatchDataset([img_dir / "p.tif"], mask_dir=mask_dir)
        _, mask_t = ds[0]

        unique = torch.unique(mask_t)
        assert set(unique.tolist()) == {0.0, 1.0}

    def test_empty_file_list_raises(self):
        """Empty file list → ValueError."""
        with pytest.raises(ValueError, match="No image files"):
            PatchDataset([], mask_dir="/tmp")

    def test_with_transform(self, tmp_path):
        """Transform is applied to both image and mask."""
        patch_dir = _make_patch_dir(tmp_path, n_patches=1)
        image_files = sorted((patch_dir / "images").glob("*.tif"))

        # Deterministic horizontal flip
        transform = v2.Compose([v2.RandomHorizontalFlip(p=1.0)])

        ds_no_tf = PatchDataset(image_files, mask_dir=patch_dir / "masks")
        ds_tf = PatchDataset(
            image_files, mask_dir=patch_dir / "masks", transform=transform
        )

        img_orig, mask_orig = ds_no_tf[0]
        img_flip, mask_flip = ds_tf[0]

        assert img_flip.shape == img_orig.shape
        # Flipped image should equal torch.flip of original
        assert torch.allclose(img_flip, torch.flip(img_orig, dims=[-1]))
        assert torch.allclose(mask_flip, torch.flip(mask_orig, dims=[-1]))


# ===========================================================================
# create_dataloaders
# ===========================================================================


class TestCreateDataloaders:
    """Tests for create_dataloaders."""

    def test_split_correctness(self, tmp_path):
        """10 patches, val_split=0.2 → 2 val + 8 train, no overlap."""
        patch_dir = _make_patch_dir(tmp_path, n_patches=10)

        train_loader, val_loader = create_dataloaders(
            patch_dir, val_split=0.2, batch_size=4, num_workers=0,
        )

        assert len(train_loader.dataset) == 8
        assert len(val_loader.dataset) == 2

        train_names = {f.name for f in train_loader.dataset.image_files}
        val_names = {f.name for f in val_loader.dataset.image_files}
        assert train_names.isdisjoint(val_names)
        assert len(train_names | val_names) == 10

    def test_reproducible_split(self, tmp_path):
        """Same seed → same split; different seed → different split."""
        patch_dir = _make_patch_dir(tmp_path, n_patches=20)

        _, val1 = create_dataloaders(patch_dir, seed=42, num_workers=0)
        _, val2 = create_dataloaders(patch_dir, seed=42, num_workers=0)
        _, val3 = create_dataloaders(patch_dir, seed=99, num_workers=0)

        names1 = {f.name for f in val1.dataset.image_files}
        names2 = {f.name for f in val2.dataset.image_files}
        names3 = {f.name for f in val3.dataset.image_files}

        assert names1 == names2
        assert names1 != names3

    def test_returns_functional_dataloaders(self, tmp_path):
        """Can iterate one batch from each loader with correct tensor shapes."""
        patch_dir = _make_patch_dir(tmp_path, n_patches=10, patch_size=32)

        train_loader, val_loader = create_dataloaders(
            patch_dir, batch_size=4, num_workers=0,
        )

        imgs, masks = next(iter(train_loader))
        assert imgs.shape == (4, 1, 32, 32)
        assert masks.shape == (4, 1, 32, 32)

        # Val loader may have fewer than batch_size samples
        imgs_v, masks_v = next(iter(val_loader))
        assert imgs_v.shape[1:] == (1, 32, 32)
        assert masks_v.shape[1:] == (1, 32, 32)

    def test_empty_patch_dir_raises(self, tmp_path):
        """No patches in directory → ValueError."""
        empty_dir = tmp_path / "empty"
        (empty_dir / "images").mkdir(parents=True)
        (empty_dir / "masks").mkdir(parents=True)

        with pytest.raises(ValueError, match="No image files"):
            create_dataloaders(empty_dir, num_workers=0)
