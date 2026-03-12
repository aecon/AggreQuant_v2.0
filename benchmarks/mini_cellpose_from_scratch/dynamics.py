"""Flow-based dynamics for cell instance segmentation.

Implements the Cellpose algorithm:
  1. Heat diffusion on ground-truth masks to generate flow targets
  2. Euler integration to follow predicted flows at inference
  3. Histogram-based clustering to recover instance masks
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import find_objects, maximum_filter


# ---------------------------------------------------------------------------
# Flow target generation (training time)
# ---------------------------------------------------------------------------

def masks_to_flows(masks):
    """Convert a label mask to normalized flow fields via heat diffusion.

    For each labeled instance:
      1. Find center (pixel closest to mean of all mask pixels)
      2. Run heat diffusion from center within the mask
      3. Compute spatial gradients of the heat map
      4. Normalize gradient vectors to unit length

    Args:
        masks: (H, W) int array, 0=background, 1..N=instance labels.

    Returns:
        flows: (2, H, W) float32 array of [flow_y, flow_x], unit-normalized.
    """
    H, W = masks.shape
    flows = np.zeros((2, H, W), dtype=np.float32)

    if masks.max() == 0:
        return flows

    # Pad masks by 1 on each side (so gradient computation doesn't go OOB)
    masks_padded = np.pad(masks, 1, mode="constant", constant_values=0)
    slices = find_objects(masks)

    for i, slc in enumerate(slices):
        if slc is None:
            continue
        label = i + 1
        # Extract the bounding box with 1-pixel padding
        sr, sc = slc
        y0, y1 = sr.start, sr.stop
        x0, x1 = sc.start, sc.stop

        # Work in padded coordinates (+1 offset)
        py0, py1 = y0, y1 + 2  # +2 because pad added 1 on each side
        px0, px1 = x0, x1 + 2

        # Local mask region
        local_mask = masks_padded[py0:py1, px0:px1]
        is_this = (local_mask == label)

        # Find pixels belonging to this instance
        ys, xs = np.nonzero(is_this)
        if len(ys) == 0:
            continue

        # Find center: pixel closest to the mean coordinate
        ymean, xmean = ys.mean(), xs.mean()
        dists = (ys - ymean) ** 2 + (xs - xmean) ** 2
        imin = dists.argmin()
        ymed, xmed = ys[imin], xs[imin]

        # Heat diffusion
        ly, lx = local_mask.shape
        niter = 2 * (ly + lx)
        T = _heat_diffusion(ys, xs, ymed, xmed, lx, niter)

        # Reshape T back to 2D
        T_2d = np.zeros((ly, lx), dtype=np.float64)
        T_2d[ys, xs] = T

        # Spatial gradients (central differences)
        dy = T_2d[2:, 1:-1] - T_2d[:-2, 1:-1]
        dx = T_2d[1:-1, 2:] - T_2d[1:-1, :-2]

        # These gradients are in the interior of the local box (excluding
        # the 1-pixel border we already have from padding). Map back to
        # the original image coordinates.
        # The local box [py0:py1, px0:px1] maps to original [y0:y1, x0:x1]
        # after removing the pad offset. dy/dx have shape (ly-2, lx-2),
        # which corresponds exactly to the original mask region [y0:y1, x0:x1].
        region_mask = masks[y0:y1, x0:x1] == label
        flows[0, y0:y1, x0:x1] += dy * region_mask
        flows[1, y0:y1, x0:x1] += dx * region_mask

    # Normalize to unit length
    mag = np.sqrt(flows[0] ** 2 + flows[1] ** 2) + 1e-20
    flows[0] /= mag
    flows[1] /= mag

    # Zero out background
    bg = masks == 0
    flows[0][bg] = 0
    flows[1][bg] = 0

    return flows


def _heat_diffusion(y, x, ymed, xmed, lx, niter):
    """Run heat diffusion from center pixel within a single mask instance.

    At each iteration: add 1 to the center, then average each mask pixel
    with its 8 neighbors (only within-mask pixels, via flat indexing).

    Args:
        y, x: Pixel coordinates within the local bounding box.
        ymed, xmed: Center coordinates.
        lx: Width of the local bounding box (for flat indexing).
        niter: Number of diffusion iterations.

    Returns:
        T_vals: Array of diffusion values at the mask pixel locations.
    """
    n_pixels = len(y)
    # Flat indices
    idx = y * lx + x
    center_idx = ymed * lx + xmed

    # Neighbor offsets for 8-connected grid (plus self)
    offsets = np.array([-lx - 1, -lx, -lx + 1,
                        -1,       0,        1,
                         lx - 1,  lx,  lx + 1])

    # Build a set of valid flat indices for fast lookup
    idx_set = set(idx.tolist())

    # For each pixel, find which of its 9 neighbors are inside the mask
    # Pre-compute neighbor indices
    all_neighbors = idx[:, None] + offsets[None, :]  # (n_pixels, 9)

    # Boolean mask: is each neighbor valid?
    is_valid = np.zeros_like(all_neighbors, dtype=bool)
    for j in range(9):
        for k in range(n_pixels):
            is_valid[k, j] = all_neighbors[k, j] in idx_set

    # Count valid neighbors per pixel
    n_valid = is_valid.sum(axis=1).astype(np.float64)
    n_valid[n_valid == 0] = 1  # avoid division by zero

    # Create a mapping from flat index to position in T_vals
    idx_to_pos = {}
    for k, flat_idx in enumerate(idx):
        idx_to_pos[flat_idx] = k

    # Map neighbor indices to positions (or -1 if invalid)
    neighbor_pos = np.full_like(all_neighbors, -1)
    for k in range(n_pixels):
        for j in range(9):
            if is_valid[k, j]:
                neighbor_pos[k, j] = idx_to_pos[all_neighbors[k, j]]

    # Find center position
    center_pos = idx_to_pos[center_idx]

    # Run diffusion
    T_vals = np.zeros(n_pixels, dtype=np.float64)
    for _ in range(niter):
        T_vals[center_pos] += 1
        T_new = np.zeros(n_pixels, dtype=np.float64)
        for k in range(n_pixels):
            total = 0.0
            count = 0
            for j in range(9):
                if is_valid[k, j]:
                    total += T_vals[neighbor_pos[k, j]]
                    count += 1
            T_new[k] = total / count if count > 0 else 0.0
        T_vals = T_new

    return T_vals


# ---------------------------------------------------------------------------
# Euler integration (inference time)
# ---------------------------------------------------------------------------

def follow_flows(dP, cellprob, niter=200, cellprob_threshold=0.0, device=None):
    """Euler-integrate flow field to find pixel convergence points.

    Uses torch grid_sample for GPU-accelerated bilinear interpolation.

    Args:
        dP: (2, H, W) float32 flow field [flow_y, flow_x].
        cellprob: (H, W) float32 cell probability (after sigmoid).
        niter: Number of Euler integration steps.
        cellprob_threshold: Threshold for foreground mask.
        device: Torch device (defaults to cuda if available, else cpu).

    Returns:
        p_final: (2, N) int tensor of final [y, x] positions for N foreground pixels.
        inds: Tuple of (y_inds, x_inds) arrays for the foreground pixels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fg_mask = cellprob > cellprob_threshold
    inds = np.nonzero(fg_mask)
    if len(inds[0]) == 0:
        return torch.zeros((2, 0), dtype=torch.int, device=device), inds

    shape = np.array(dP.shape[1:], dtype=np.float32)  # [H, W]

    # Mask flows to foreground and scale down (matches Cellpose convention)
    dP_masked = dP * fg_mask[None] / 5.0

    # Set up tensors for grid_sample
    # grid_sample expects: input (1, C, H, W), grid (1, 1, N, 2) in [-1, 1]
    # grid coordinates are (x, y) not (y, x)
    im = torch.zeros((1, 2, int(shape[0]), int(shape[1])),
                      dtype=torch.float32, device=device)
    # flow_y -> im[0, 1], flow_x -> im[0, 0] (reversed for grid_sample x,y order)
    im[0, 0] = torch.from_numpy(dP_masked[1]).to(device)  # dx
    im[0, 1] = torch.from_numpy(dP_masked[0]).to(device)  # dy

    # Normalize flows: grid_sample works in [-1, 1], so scale flows accordingly
    im[0, 0] *= 2.0 / (shape[1] - 1)
    im[0, 1] *= 2.0 / (shape[0] - 1)

    # Initialize pixel positions
    n_pts = len(inds[0])
    pt = torch.zeros((1, 1, n_pts, 2), dtype=torch.float32, device=device)
    # grid_sample grid: last dim is (x, y)
    pt[0, 0, :, 0] = torch.from_numpy(inds[1].astype(np.float32)).to(device)  # x
    pt[0, 0, :, 1] = torch.from_numpy(inds[0].astype(np.float32)).to(device)  # y

    # Normalize to [-1, 1]
    pt[0, 0, :, 0] = pt[0, 0, :, 0] / (shape[1] - 1) * 2 - 1
    pt[0, 0, :, 1] = pt[0, 0, :, 1] / (shape[0] - 1) * 2 - 1

    # Euler integration
    for _ in range(niter):
        dPt = F.grid_sample(im, pt, align_corners=True, padding_mode="border")
        # dPt shape: (1, 2, 1, N) -> squeeze to get (2, N)
        pt[0, 0, :, 0] = torch.clamp(pt[0, 0, :, 0] + dPt[0, 0, 0, :], -1, 1)
        pt[0, 0, :, 1] = torch.clamp(pt[0, 0, :, 1] + dPt[0, 1, 0, :], -1, 1)

    # Convert back from [-1, 1] to pixel coordinates
    pt_final = pt[0, 0]  # (N, 2) with (x, y)
    x_coords = (pt_final[:, 0] + 1) / 2 * (shape[1] - 1)
    y_coords = (pt_final[:, 1] + 1) / 2 * (shape[0] - 1)

    p_final = torch.stack([y_coords, x_coords], dim=0).int()  # (2, N)
    return p_final, inds


# ---------------------------------------------------------------------------
# Histogram-based mask recovery (inference time)
# ---------------------------------------------------------------------------

def get_masks(p_final, inds, shape, rpad=20, peak_threshold=10,
              extend_threshold=2, max_size_fraction=0.4):
    """Create instance masks from pixel convergence points.

    1. Build 2D histogram of final pixel positions (with padding).
    2. Find peaks: local maxima (5x5) with count > peak_threshold.
    3. Expand peaks by dilating, keeping where count > extend_threshold.
    4. Map each foreground pixel to its label.

    Args:
        p_final: (2, N) int tensor of final [y, x] positions.
        inds: Tuple of (y_inds, x_inds) for the foreground pixels.
        shape: (H, W) of the original image.
        rpad: Histogram edge padding.
        peak_threshold: Minimum histogram count for a seed.
        extend_threshold: Minimum count for extending a seed.
        max_size_fraction: Max fraction of image a single mask can occupy.

    Returns:
        masks: (H, W) uint16 label mask.
    """
    device = p_final.device
    H, W = shape

    if p_final.shape[1] == 0:
        return np.zeros(shape, dtype=np.uint16)

    # Add padding and clamp
    pt = p_final.clone().long()
    pt += rpad
    pt[0] = torch.clamp(pt[0], 0, H + 2 * rpad - 1)
    pt[1] = torch.clamp(pt[1], 0, W + 2 * rpad - 1)

    # Build histogram via sparse tensor
    padded_shape = (H + 2 * rpad, W + 2 * rpad)
    coo = torch.sparse_coo_tensor(
        pt, torch.ones(pt.shape[1], device=device, dtype=torch.int), padded_shape
    )
    h = coo.to_dense()
    del coo

    # Find local maxima (5x5 max pool)
    h_np = h.cpu().numpy().astype(np.float64)
    hmax = maximum_filter(h_np, size=5)
    seeds = np.nonzero((np.abs(h_np - hmax) < 1e-6) & (h_np > peak_threshold))

    if len(seeds[0]) == 0:
        return np.zeros(shape, dtype=np.uint16)

    # Sort seeds by count (ascending — later seeds overwrite earlier ones)
    counts_at_seeds = h_np[seeds]
    order = np.argsort(counts_at_seeds)
    seeds = (seeds[0][order], seeds[1][order])
    n_seeds = len(seeds[0])

    # For each seed, extract an 11x11 patch from the histogram and dilate
    M = np.zeros(padded_shape, dtype=np.int32)
    for k in range(n_seeds):
        sy, sx = seeds[0][k], seeds[1][k]
        # Extract 11x11 region around seed
        y0, y1 = sy - 5, sy + 6
        x0, x1 = sx - 5, sx + 6

        # Clip to bounds
        cy0, cy1 = max(y0, 0), min(y1, padded_shape[0])
        cx0, cx1 = max(x0, 0), min(x1, padded_shape[1])

        h_patch = h_np[cy0:cy1, cx0:cx1]
        seed_mask = np.zeros_like(h_patch)
        # Place seed at center (adjusted for clipping)
        seed_mask[sy - cy0, sx - cx0] = 1

        # Dilate 5 times with 3x3 max filter, constrained by histogram > threshold
        for _ in range(5):
            seed_mask = maximum_filter(seed_mask, size=3)
            seed_mask *= (h_patch > extend_threshold)

        # Write to global mask
        region = M[cy0:cy1, cx0:cx1]
        region[seed_mask > 0] = k + 1

    # Map each foreground pixel to its label using its final position
    pt_np = pt.cpu().numpy()
    labels = M[pt_np[0], pt_np[1]]

    # Write into output mask
    masks = np.zeros(shape, dtype=np.uint16)
    masks[inds] = labels.astype(np.uint16)

    # Remove oversized masks
    if masks.max() > 0:
        uniq, counts = np.unique(masks, return_counts=True)
        big = H * W * max_size_fraction
        for u, c in zip(uniq, counts):
            if u > 0 and c > big:
                masks[masks == u] = 0

        # Relabel consecutively
        _relabel_consecutive(masks)

    return masks


def _relabel_consecutive(masks):
    """Relabel mask IDs to be consecutive 1..N, in-place."""
    uniq = np.unique(masks)
    uniq = uniq[uniq > 0]
    for new_label, old_label in enumerate(uniq, start=1):
        if old_label != new_label:
            masks[masks == old_label] = new_label


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

def compute_masks(flows, cellprob, niter=200, cellprob_threshold=0.0,
                  min_size=15, device=None):
    """Full pipeline: predicted flows + cellprob -> instance masks.

    Args:
        flows: (2, H, W) predicted flow field.
        cellprob: (H, W) predicted cell probability (after sigmoid).
        niter: Euler integration steps.
        cellprob_threshold: Foreground threshold.
        min_size: Remove masks smaller than this (in pixels).
        device: Torch device.

    Returns:
        masks: (H, W) uint16 label mask.
    """
    shape = cellprob.shape

    if (cellprob > cellprob_threshold).sum() == 0:
        return np.zeros(shape, dtype=np.uint16)

    # Follow flows
    p_final, inds = follow_flows(
        flows, cellprob, niter=niter,
        cellprob_threshold=cellprob_threshold, device=device
    )

    # Get masks from convergence points
    masks = get_masks(p_final, inds, shape)

    # Remove small masks
    if min_size > 0 and masks.max() > 0:
        uniq, counts = np.unique(masks, return_counts=True)
        for u, c in zip(uniq, counts):
            if u > 0 and c < min_size:
                masks[masks == u] = 0
        _relabel_consecutive(masks)

    return masks
