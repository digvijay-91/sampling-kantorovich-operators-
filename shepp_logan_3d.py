"""
shepp_logan_3d.py
-----------------
Generates a 3D synthetic volume from the 2D Shepp-Logan phantom and
defines the anatomically-motivated Regions of Interest (ROIs) used in:

  Singh & Singh (2026) — arXiv:2508.14043

Volume specification
--------------------
  - Base phantom : 2D Shepp-Logan (128 × 128)
  - 3D volume    : 64 slices × 128 × 128  (bicubic resize per slice)
  - Mid-slice    : volume[32, :, :]  — used for 2D operator evaluation

ROI definitions (mid-slice coordinates)
----------------------------------------
  1. White Matter (WM)      — rows 50–70,  cols 50–70
  2. Tumor ROI              — rows 30–50,  cols 80–100
  3. CSF                    — rows 10–30,  cols 10–30
  4. Liver Parenchyma       — rows 70–90,  cols 30–50
  5. Kidney Edge            — rows 80–100, cols 90–110
  6. Aorta                  — rows 40–60,  cols 10–30
"""

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.util import img_as_float


# ──────────────────────────────────────────────
# ROI definitions — (row_start, row_end, col_start, col_end)
# ──────────────────────────────────────────────

ROIS = {
    'White Matter':    (50,  70,  50,  70),
    'Tumor ROI':       (30,  50,  80, 100),
    'CSF':             (10,  30,  10,  30),
    'Liver Parenchyma':(70,  90,  30,  50),
    'Kidney Edge':     (80, 100,  90, 110),
    'Aorta':           (40,  60,  10,  30),
}

# ROI display colors (for visualisation)
ROI_COLORS = {
    'White Matter':     'cyan',
    'Tumor ROI':        'red',
    'CSF':              'blue',
    'Liver Parenchyma': 'green',
    'Kidney Edge':      'orange',
    'Aorta':            'purple',
}


def generate_phantom_2d(size: int = 128) -> np.ndarray:
    """
    Generate a 2D Shepp-Logan phantom of given size.

    Parameters
    ----------
    size : int — output image dimensions (size × size)

    Returns
    -------
    np.ndarray, shape (size, size), dtype float64, values in [0, 1]
    """
    phantom = shepp_logan_phantom()          # 400 × 400 by default
    phantom = resize(phantom, (size, size),
                     order=3,                # bicubic
                     anti_aliasing=True,
                     mode='reflect')
    return img_as_float(phantom)


def generate_volume_3d(n_slices: int = 64, size: int = 128) -> np.ndarray:
    """
    Build a 3D volume by stacking slightly varied 2D phantoms.

    Each slice is a rescaled version of the base phantom, simulating
    axial MRI/CT slices — consistent with the 64×128×128 volume
    described in Singh & Singh (2026).

    Parameters
    ----------
    n_slices : int — number of axial slices
    size     : int — in-plane resolution (size × size)

    Returns
    -------
    np.ndarray, shape (n_slices, size, size), dtype float64
    """
    base = generate_phantom_2d(size)
    volume = np.zeros((n_slices, size, size), dtype=np.float64)

    for i in range(n_slices):
        # Simulate slight contrast variation across slices
        scale = 0.85 + 0.30 * np.sin(np.pi * i / n_slices)
        volume[i] = np.clip(base * scale, 0, 1)

    return volume


def get_mid_slice(volume: np.ndarray) -> np.ndarray:
    """
    Return the mid-axial slice of the volume (volume[32, :, :]).

    Parameters
    ----------
    volume : np.ndarray, shape (n_slices, H, W)

    Returns
    -------
    np.ndarray, shape (H, W)
    """
    mid = volume.shape[0] // 2
    return volume[mid]


def extract_roi(image: np.ndarray, roi_name: str) -> np.ndarray:
    """
    Extract a named ROI from a 2D image.

    Parameters
    ----------
    image    : 2D array
    roi_name : one of the keys in ROIS

    Returns
    -------
    np.ndarray — cropped ROI region
    """
    if roi_name not in ROIS:
        raise ValueError(f"Unknown ROI '{roi_name}'. "
                         f"Available: {list(ROIS.keys())}")
    r0, r1, c0, c1 = ROIS[roi_name]
    return image[r0:r1, c0:c1]


def extract_all_rois(image: np.ndarray) -> dict:
    """
    Extract all defined ROIs from a 2D image.

    Parameters
    ----------
    image : 2D array

    Returns
    -------
    dict mapping ROI name → np.ndarray
    """
    return {name: extract_roi(image, name) for name in ROIS}


def add_gaussian_noise(image: np.ndarray,
                       sigma: float = 0.05,
                       seed: int = 42) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to simulate real measurement conditions.

    f*(x) = f(x) + ε*(x),   ε*(x) ~ N(0, σ²)

    This matches the noisy observation model in the PSK operator framework.

    Parameters
    ----------
    image : clean image array
    sigma : noise standard deviation (default 0.05 — ~5% of signal range)
    seed  : random seed for reproducibility

    Returns
    -------
    np.ndarray — noisy image, clipped to [0, 1]
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 1)


if __name__ == '__main__':
    # Quick sanity check
    vol = generate_volume_3d()
    mid = get_mid_slice(vol)
    print(f"Volume shape : {vol.shape}")
    print(f"Mid-slice    : {mid.shape}, range [{mid.min():.3f}, {mid.max():.3f}]")
    for name, coords in ROIS.items():
        roi = extract_roi(mid, name)
        print(f"  ROI '{name}': shape {roi.shape}, "
              f"mean={roi.mean():.4f}, std={roi.std():.4f}")
