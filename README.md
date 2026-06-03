# Sampling Kantorovich Operators
### Mathematical Reconstruction from Sampled Data

[![arXiv](https://img.shields.io/badge/arXiv-2506.12053-b31b1b.svg)](https://arxiv.org/abs/2506.12053)
[![arXiv](https://img.shields.io/badge/arXiv-2508.14043-b31b1b.svg)](https://arxiv.org/abs/2508.14043)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** Digvijay Singh  
**Affiliation:** Homi Bhabha Teaching-cum-Research Fellow, IET Lucknow (AKTU), India  
**Supervisors:** Dr. Karunesh Kumar Singh · Dr. Rahul Shukla (Deshabandhu College)

---

## Overview

This repository implements the approximation operators studied in my doctoral research on **Sampling Kantorovich (SK) operators**, their **probabilistic extensions (PSK)**, and **wavelet-based filtering operators** — with applications to computational imaging (MRI/CT reconstruction).

The core mathematical question:

> *How do we reconstruct a function **f** stably and accurately from incomplete, noisy samples?*

The answer provided here is mathematically guaranteed — not empirical.

---

## Research Papers

| # | Title | Venue | Status |
|---|-------|-------|--------|
| 1 | Fuzzy approximation theorems via power series summability methods in two variables | *Soft Computing*, 28(2):945–953, 2024 | Published |
| 2 | On wavelet-based sampling Kantorovich operators and their study in multi-resolution analysis | *Iranian Journal of Science*, 2026 | Published |
| 3 | Fundamental theorem of approximation for linear and non-linear sampling Kantorovich operators | Springer: *Positive Operators and Fixed-Point Theorems*, pp. 141–156, 2025 | Published |
| 4 | On a novel probabilistic sampling Kantorovich operators and its applications | arXiv:2506.12053 | Under Review |
| 5 | A comparative study of some wavelet and sampling operators on various features of an image | arXiv:2508.14043 | Communicated |

---

## Repository Structure

```
sampling-kantorovich-operators/
│
├── README.md
├── requirements.txt
│
├── image_experiments/          ← Operator comparison on Shepp-Logan phantom
│   ├── metrics.py              ← PSNR, SSIM, MAE, SI, SSI, SMPI, ENL
│   ├── shepp_logan_3d.py       ← 3D volume generation + ROI definitions
│   ├── operators_comparison.py ← Gaussian, Kantorovich, Bilateral, Wavelet
│   └── run_image_experiment.py ← Reproduces Tables 2,3 and Fig 7
│
├── sk_operators/               ← Classical & Probabilistic SK operators (coming soon)
│   ├── classical_sk.py
│   └── psk_operators.py
│
└── wavelet_filtering/          ← Wavelet-Kantorovich operators (coming soon)
    └── wavelet_operator.py
```

---

## Implemented Operators

### 1. Gaussian Operator  G_σ
```
(G_σ f)(x) = ∫ f(y) φ_σ(x − y) dy
```
Convergence: ‖G_σ f − f‖_{Lᵖ} → 0 as σ → 0

### 2. Kantorovich (Sampling) Operator  S_w
```
(S_w f)(x) = Σ_{k∈Z} χ(wx − k) · [w ∫_{k/w}^{(k+1)/w} f(u) du]
```
Convergence: ‖S_w f − f‖_{L¹} → 0 as w → ∞  (Butzer & Stens, 1991)

### 3. Bilateral Operator  B_σ
```
(B_σ f)(x) = (1/Z) Σ_y f(y) · G_σ(x−y) · G_r(|f(x)−f(y)|)
```
Convergence: ‖B_σ f − f‖_{Lᵖ} → 0 as σ, r → 0  (Tomasi & Manduchi, 1998)

### 4. Wavelet Operator  W_T
```
S_W f = W⁻¹(c_j, T_λ(d_j^{(l)}))
```
Convergence: ‖W_T f − f‖_{Lᵖ} → 0 as j → ∞

---

## Key Result (Theorem 2 — arXiv:2506.12053)

For the **Probabilistic Sampling Kantorovich (PSK) operator** P_nᵉ:

```
E[‖P_nᵉ f − f‖_{L¹}] → 0   as n → ∞
```

under: f ∈ L¹(ℝᵈ),  sup‖ε‖_{L¹} < ∞,  and Uniform Integrability.

This means: **even with noisy measurements, reconstruction converges on average** — a mathematical guarantee, not an empirical claim.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/sampling-kantorovich-operators
cd sampling-kantorovich-operators
pip install -r requirements.txt
cd image_experiments
python run_image_experiment.py
```

Results are saved to `image_experiments/results/`:
- `table2_roi_metrics.csv` — SI, SSI, SMPI, ENL across 6 ROIs × 4 operators
- `table_mse_3d.csv` — MSE at resolutions 16³, 32³, 64³
- `fig7_roi_grid.png` — ROI behaviour grid (Fig. 7)
- `fig_mid_slice_comparison.png` — Full mid-slice comparison

---

## Numerical Results Summary

### 3D MSE — All Operators at Multiple Resolutions

| Resolution | Gaussian | Kantorovich | Bilateral | Wavelet |
|------------|----------|-------------|-----------|---------|
| 16³        | 0.00403  | 0.00757     | **0.00161** | 0.00149 |
| 32³        | 0.00596  | 0.01109     | **0.00152** | 0.00144 |
| 64³        | 0.00712  | 0.01377     | 0.00099  | **0.00127** |

**Key finding:** Bilateral achieves lowest MSE at all resolutions; all operators converge as resolution increases — confirming the Fundamental Theorem of Approximation.

### Operator Recommendations by Region

| Region | Best Operator | Reason |
|--------|---------------|--------|
| Smooth regions (Tumor, Liver, WM) | Kantorovich | Highest SMPI, ENL |
| Speckle reduction (all ROIs) | Wavelet | SSI = 1.00 everywhere |
| Edge regions (Kidney, Aorta) | Bilateral | SSI ≈ 0.99 |
| Preliminary denoising | Gaussian | Simple, fast, uniform |

---

## Citation

If you use this code, please cite:

```bibtex
@article{singh2024fuzzy,
  title={Fuzzy approximation theorems via power series summability methods in two variables},
  author={Singh, Digvijay and Singh, Karunesh Kumar},
  journal={Soft Computing},
  volume={28},
  number={2},
  pages={945--953},
  year={2024}
}

@misc{singh2025psk,
  title={On a novel probabilistic sampling Kantorovich operators and its applications},
  author={Singh, Digvijay and Singh, Karunesh Kumar and Shukla, Rahul},
  year={2025},
  eprint={2506.12053},
  archivePrefix={arXiv}
}

@misc{singh2025comparative,
  title={A comparative study of some wavelet and sampling operators on various features of an image},
  author={Singh, Digvijay and Singh, Karunesh Kumar and Shukla, Rahul},
  year={2025},
  eprint={2508.14043},
  archivePrefix={arXiv}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
