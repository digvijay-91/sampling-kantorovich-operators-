"""
operators_comparison.py
-----------------------
Implements the four approximation operators compared in:

  Singh & Singh (2026) — "A comparative study of some wavelet and sampling
  operators on various features of an image", arXiv:2508.14043

Operators
---------
1. Gaussian operator  G_σ         — uniform convolution smoothing
2. Kantorovich operator  S_w       — local averaging (SK operator)
3. Bilateral operator  B_σ         — edge-preserving smoothing
4. Wavelet operator  W_T           — multiresolution thresholding

Theoretical convergence (all four)
-----------------------------------
  ‖T_n f − f‖_{L^p} → 0  as  n → ∞

under appropriate kernel conditions (proved in Singh & Singh 2026,
Theorems 1–4). Differences appear only in the RATE and local
STRUCTURE of convergence — not in the guarantee itself.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral, denoise_wavelet
import pywt


# ──────────────────────────────────────────────
# 1. Gaussian Operator   G_σ f
# ──────────────────────────────────────────────

def gaussian_operator(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian smoothing operator.

    (G_σ f)(x) = ∫ f(y) φ_σ(x − y) dy

    where  φ_σ(t) = (2πσ²)^{-1/2} exp(−t²/2σ²)

    As σ → 0:  ‖G_σ f − f‖_{L^p} → 0  (modifier convolution → identity)

    Behaviour
    ---------
    + Uniform smoothing across entire image
    + Simple, fast, well-understood
    − Fails to preserve edges
    − MSE improves gradually with σ

    Parameters
    ----------
    image : 2D float array, values in [0, 1]
    sigma : Gaussian standard deviation (controls smoothing width)

    Returns
    -------
    np.ndarray — smoothed image
    """
    return gaussian_filter(image.astype(np.float64), sigma=sigma)


# ──────────────────────────────────────────────
# 2. Kantorovich (Sampling) Operator   S_w f
# ──────────────────────────────────────────────

def kantorovich_operator(image: np.ndarray,
                          n_iter: int = 3,
                          sigma: float = 1.0) -> np.ndarray:
    """
    Sampling Kantorovich operator — local averaging version.

    (S_w f)(x) = Σ_k  w_k · f(k/n) · u_k(x)

    Physical interpretation: replaces the unknown pointwise value f(k/w)
    with a local average over a small cell [k/w, (k+1)/w].
    This is exactly what sensors and cameras do.

    General form (Bardaro et al., 2004):
      (S_w f)(x) = Σ_{k∈Z} χ(wx − k) · [w ∫_{k/w}^{(k+1)/w} f(u) du]

    Convergence (Butzer & Stens, 1991):
      ‖S_w f − f‖_{L¹} → 0  as  w → ∞

    Behaviour
    ---------
    + Stable, uniform approximation
    + Best mean preservation (high SMPI)
    + High ENL in smooth regions
    − Lacks adaptive local sharpness (over-smooths edges)

    Implementation note
    -------------------
    We implement the discrete analogue via iterated Gaussian blur
    (approximating the box-kernel averaging), consistent with the
    3-iteration Gaussian blur used in Singh & Singh (2026) Table MSE results.

    Parameters
    ----------
    image  : 2D float array
    n_iter : number of averaging iterations (default 3)
    sigma  : per-iteration blur width

    Returns
    -------
    np.ndarray — Kantorovich-approximated image
    """
    result = image.astype(np.float64).copy()
    for _ in range(n_iter):
        result = gaussian_filter(result, sigma=sigma)
    return result


# ──────────────────────────────────────────────
# 3. Bilateral Operator   B_σ f
# ──────────────────────────────────────────────

def bilateral_operator(image: np.ndarray,
                        sigma_color: float = 0.05,
                        sigma_spatial: float = 1.0) -> np.ndarray:
    """
    Bilateral filtering operator (Tomasi & Manduchi, 1998).

    (B_σ f)(x) = (1/Z) Σ_y f(y) · G_σ(x−y) · G_r(|f(x)−f(y)|)

    Combines spatial proximity AND intensity similarity — smooths
    uniform regions while preserving sharp edges.

    Convergence (adapted):
      ‖B_σ f − f‖_{L^p} → 0  as  σ, r → 0

    Behaviour
    ---------
    + Best edge preservation (highest SSIM at edges)
    + Lowest MSE at all 3D resolutions
    + Superior in kidney edge, aorta ROIs  (SSI ≈ 0.99)
    − Higher SI than Kantorovich in smooth interiors

    Parameters
    ----------
    image        : 2D float array, values in [0, 1]
    sigma_color  : range (intensity) kernel width  (σ_r)
    sigma_spatial: spatial kernel width              (σ_s)

    Returns
    -------
    np.ndarray — bilaterally filtered image
    """
    image_f = np.asarray(image, dtype=np.float64)
    # skimage bilateral expects float in [0,1]
    result = denoise_bilateral(image_f,
                                sigma_color=sigma_color,
                                sigma_spatial=sigma_spatial,
                                channel_axis=None)
    return result.astype(np.float64)


# ──────────────────────────────────────────────
# 4. Wavelet Operator   W_T f
# ──────────────────────────────────────────────

def wavelet_operator(image: np.ndarray,
                      wavelet: str = 'db4',
                      mode: str = 'soft',
                      rescale_sigma: bool = True) -> np.ndarray:
    """
    Wavelet thresholding operator (multiresolution analysis).

    W(f) = {c_j, T_λ(d_j^{(l)})}
    Reconstructed:  S_W f = W^{-1}(c_j, T_λ(d_j^{(l)}))

    where T_λ is soft or hard thresholding and d_j^{(l)} are the
    wavelet detail coefficients at scale j, subband l.

    Theoretical bound (Singh & Singh 2026, Theorem 4):
      ‖W_T f − f‖_{L^p} → 0  as  j → ∞
      if  ‖Wf‖_{ℓ^p} ≤ C · 2^{j(s+1/2)} · ‖f‖_{W^{s,p}}

    Behaviour
    ---------
    + Perfect speckle suppression (SSI = 1.00 in ALL ROIs)
    + Best structure retention
    − Thresholding artefacts at low resolution (j small)
    − Not edge-adaptive

    Parameters
    ----------
    image         : 2D float array
    wavelet       : wavelet family (default 'db4' — Daubechies 4)
    mode          : 'soft' or 'hard' thresholding
    rescale_sigma : if True, estimate noise from image for threshold

    Returns
    -------
    np.ndarray — wavelet-denoised image
    """
    image_f = np.asarray(image, dtype=np.float64)
    result = denoise_wavelet(image_f,
                              wavelet=wavelet,
                              mode=mode,
                              rescale_sigma=rescale_sigma,
                              channel_axis=None)
    return result.astype(np.float64)


# ──────────────────────────────────────────────
# Convenience: apply all four operators at once
# ──────────────────────────────────────────────

def apply_all_operators(image: np.ndarray,
                         gaussian_sigma: float = 1.0,
                         kantorovich_iter: int = 3,
                         bilateral_color: float = 0.05,
                         bilateral_spatial: float = 1.0,
                         wavelet: str = 'db4') -> dict:
    """
    Apply all four operators to a single image.

    Parameters
    ----------
    image             : 2D float array (clean or noisy)
    gaussian_sigma    : σ for Gaussian operator
    kantorovich_iter  : iterations for Kantorovich operator
    bilateral_color   : σ_color for Bilateral operator
    bilateral_spatial : σ_spatial for Bilateral operator
    wavelet           : wavelet type for Wavelet operator

    Returns
    -------
    dict mapping operator name → reconstructed image array
    """
    return {
        'Gaussian':     gaussian_operator(image, sigma=gaussian_sigma),
        'Kantorovich':  kantorovich_operator(image, n_iter=kantorovich_iter),
        'Bilateral':    bilateral_operator(image,
                                           sigma_color=bilateral_color,
                                           sigma_spatial=bilateral_spatial),
        'Wavelet':      wavelet_operator(image, wavelet=wavelet),
    }


if __name__ == '__main__':
    # Quick self-test on a small random image
    rng = np.random.default_rng(0)
    test_img = rng.random((64, 64))

    results = apply_all_operators(test_img)
    for name, out in results.items():
        print(f"{name:15s}  shape={out.shape}  "
              f"range=[{out.min():.3f}, {out.max():.3f}]")
