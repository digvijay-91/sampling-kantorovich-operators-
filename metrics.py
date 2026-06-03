"""
metrics.py
----------
Quantitative image quality metrics used in:

  Singh & Singh (2026) — "A comparative study of some wavelet and sampling
  operators on various features of an image", arXiv:2508.14043

Metrics implemented
-------------------
Standard reconstruction quality:
  - PSNR  : Peak Signal-to-Noise Ratio          (higher = better)
  - SSIM  : Structural Similarity Index          (closer to 1 = better)
  - MAE   : Mean Absolute Error                  (lower = better)

Speckle / noise analysis (for ROI evaluation):
  - SI    : Speckle Index                        (lower = less noise)
  - SSI   : Speckle Suppression Index            (lower = better suppression)
  - SMPI  : Speckle Mean Preservation Index      (closer to 1 = better)
  - ENL   : Equivalent Number of Looks           (higher = smoother region)
"""

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


# ──────────────────────────────────────────────
# Standard reconstruction metrics
# ──────────────────────────────────────────────

def psnr(original: np.ndarray, reconstructed: np.ndarray,
         data_range: float = None) -> float:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 10 · log10( MAX² / MSE )

    Parameters
    ----------
    original       : reference image array
    reconstructed  : approximated image array
    data_range     : maximum possible pixel value (default: max of original)

    Returns
    -------
    float  — PSNR in decibels; +inf if images are identical
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')

    if data_range is None:
        data_range = original.max() - original.min()
        if data_range == 0:
            data_range = 1.0

    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(original: np.ndarray, reconstructed: np.ndarray,
         data_range: float = None) -> float:
    """
    Structural Similarity Index (Wang et al., 2004).

    Measures perceived quality across luminance, contrast, and structure.
    Range: [-1, 1], where 1 = perfect similarity.

    Parameters
    ----------
    original       : reference image array (2D or 3D)
    reconstructed  : approximated image array
    data_range     : pixel value range (default: max - min of original)
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if data_range is None:
        data_range = original.max() - original.min()
        if data_range == 0:
            data_range = 1.0

    return float(skimage_ssim(original, reconstructed,
                              data_range=data_range))


def mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Mean Absolute Error.

    MAE = (1/N) · Σ |f(i,j) - f̂(i,j)|

    Parameters
    ----------
    original       : reference image array
    reconstructed  : approximated image array
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    return float(np.mean(np.abs(original - reconstructed)))


# ──────────────────────────────────────────────
# ROI speckle / noise metrics
# ──────────────────────────────────────────────

def speckle_index(region: np.ndarray) -> float:
    """
    Speckle Index (SI).

    SI = σ_I / μ_I

    Measures noise level relative to mean intensity within a region.
    Lower SI indicates less noise.

    Parameters
    ----------
    region : 2D array representing an image region of interest (ROI)
    """
    region = np.asarray(region, dtype=np.float64)
    mu = np.mean(region)
    sigma = np.std(region)
    if mu == 0:
        return float('nan')
    return float(sigma / mu)


def speckle_suppression_index(original_roi: np.ndarray,
                               filtered_roi: np.ndarray) -> float:
    """
    Speckle Suppression Index (SSI).

    SSI = SI_filtered / SI_original

    Ratio of filtered to original speckle index.
    SSI → 0 means excellent noise suppression.

    Parameters
    ----------
    original_roi : ROI from the original (noisy) image
    filtered_roi : same ROI from the filtered/reconstructed image
    """
    si_orig = speckle_index(original_roi)
    si_filt = speckle_index(filtered_roi)
    if si_orig == 0 or np.isnan(si_orig):
        return float('nan')
    return float(si_filt / si_orig)


def speckle_mean_preservation_index(original_roi: np.ndarray,
                                     filtered_roi: np.ndarray) -> float:
    """
    Speckle Mean Preservation Index (SMPI).

    SMPI = μ_filtered / μ_original

    Measures how well mean intensity is preserved after filtering.
    SMPI → 1 means mean intensity is faithfully retained.

    Parameters
    ----------
    original_roi : ROI from the original image
    filtered_roi : same ROI from the filtered/reconstructed image
    """
    original_roi = np.asarray(original_roi, dtype=np.float64)
    filtered_roi = np.asarray(filtered_roi, dtype=np.float64)
    mu_orig = np.mean(original_roi)
    mu_filt = np.mean(filtered_roi)
    if mu_orig == 0:
        return float('nan')
    return float(mu_filt / mu_orig)


def equivalent_number_of_looks(region: np.ndarray) -> float:
    """
    Equivalent Number of Looks (ENL).

    ENL = μ² / σ²

    Assesses smoothness in uniform regions.
    Higher ENL indicates a smoother, more homogeneous region.

    Parameters
    ----------
    region : 2D array representing an image region of interest (ROI)
    """
    region = np.asarray(region, dtype=np.float64)
    mu = np.mean(region)
    var = np.var(region)
    if var == 0:
        return float('inf')
    return float((mu ** 2) / var)


# ──────────────────────────────────────────────
# Convenience: compute all metrics at once
# ──────────────────────────────────────────────

def all_reconstruction_metrics(original: np.ndarray,
                                reconstructed: np.ndarray,
                                data_range: float = None) -> dict:
    """
    Compute PSNR, SSIM, and MAE together.

    Returns
    -------
    dict with keys: 'PSNR', 'SSIM', 'MAE'
    """
    return {
        'PSNR': psnr(original, reconstructed, data_range),
        'SSIM': ssim(original, reconstructed, data_range),
        'MAE':  mae(original, reconstructed),
    }


def all_roi_metrics(original_roi: np.ndarray,
                    filtered_roi: np.ndarray) -> dict:
    """
    Compute SI, SSI, SMPI, ENL for a single ROI pair.

    Returns
    -------
    dict with keys: 'SI', 'SSI', 'SMPI', 'ENL'
    """
    return {
        'SI':   speckle_index(filtered_roi),
        'SSI':  speckle_suppression_index(original_roi, filtered_roi),
        'SMPI': speckle_mean_preservation_index(original_roi, filtered_roi),
        'ENL':  equivalent_number_of_looks(filtered_roi),
    }
