"""
run_image_experiment.py
-----------------------
Reproduces the numerical results and figures from:

  Singh & Singh (2026) — "A comparative study of some wavelet and
  sampling operators on various features of an image"
  arXiv:2508.14043

Experiments
-----------
1. Table 2 — Denoising metrics (SI, SSI, SMPI, ENL) across all
             ROIs × all operators on the mid-slice of the 3D
             Shepp-Logan phantom.

2. Table MSE — Mean Squared Error comparison of all four operators
               at resolutions n=0,1,2 (16³, 32³, 64³ voxels).

3. Figure 7  — ROI behaviour grid: original vs 4 operators across
               6 anatomical regions.

Usage
-----
  python run_image_experiment.py

Output files (saved to ./results/)
------------------------------------
  table2_roi_metrics.csv
  table_mse_3d.csv
  fig7_roi_grid.png
  fig_mid_slice_comparison.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from shepp_logan_3d import (
    generate_volume_3d, get_mid_slice,
    add_gaussian_noise, extract_roi, ROIS, ROI_COLORS
)
from operators_comparison import apply_all_operators, gaussian_operator
from metrics import all_roi_metrics, all_reconstruction_metrics

# ── output directory ──────────────────────────
os.makedirs('results', exist_ok=True)

OPERATORS = ['Gaussian', 'Kantorovich', 'Bilateral', 'Wavelet']


# ══════════════════════════════════════════════
# EXPERIMENT 1 — Table 2: ROI denoising metrics
# ══════════════════════════════════════════════

def run_table2():
    """
    Compute SI, SSI, SMPI, ENL for each operator × each ROI.
    Matches Table 2 in Singh & Singh (2026).
    """
    print("\n── Generating phantom volume …")
    volume = generate_volume_3d(n_slices=64, size=128)
    mid    = get_mid_slice(volume)
    noisy  = add_gaussian_noise(mid, sigma=0.05, seed=42)

    print("── Applying all operators …")
    reconstructed = apply_all_operators(noisy)

    rows = []
    for roi_name in ROIS:
        orig_roi = extract_roi(mid, roi_name)
        for op_name in OPERATORS:
            filt_roi = extract_roi(reconstructed[op_name], roi_name)
            m = all_roi_metrics(orig_roi, filt_roi)
            rows.append({
                'ROI':      roi_name,
                'Operator': op_name,
                'SI':       round(m['SI'],   4),
                'SSI':      round(m['SSI'],  4),
                'SMPI':     round(m['SMPI'], 4),
                'ENL':      round(m['ENL'],  4),
            })

    df = pd.DataFrame(rows)
    df.to_csv('results/table2_roi_metrics.csv', index=False)
    print("── Table 2 saved → results/table2_roi_metrics.csv")
    print(df.to_string(index=False))
    return df, mid, noisy, reconstructed


# ══════════════════════════════════════════════
# EXPERIMENT 2 — 3D MSE at multiple resolutions
# ══════════════════════════════════════════════

def run_mse_3d():
    """
    Compute MSE for all operators at resolutions n=0,1,2
    (volumes of 16³, 32³, 64³). Matches the MSE table in
    Singh & Singh (2026) — Fig 18-style analysis.
    """
    print("\n── Running 3D MSE experiment …")
    resolutions = [(0, 16), (1, 32), (2, 64)]
    rows = []

    for n, size in resolutions:
        volume  = generate_volume_3d(n_slices=size, size=size)
        mid     = get_mid_slice(volume)
        noisy   = add_gaussian_noise(mid, sigma=0.05, seed=42)
        recon   = apply_all_operators(noisy)

        row = {'n': n, 'Resolution': f'{size}³'}
        for op_name in OPERATORS:
            mse = np.mean((mid - recon[op_name]) ** 2)
            row[op_name] = round(mse, 5)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv('results/table_mse_3d.csv', index=False)
    print("── MSE table saved → results/table_mse_3d.csv")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════
# EXPERIMENT 3 — Figure 7: ROI behaviour grid
# ══════════════════════════════════════════════

def plot_roi_grid(mid, reconstructed):
    """
    Reproduce Fig. 7 — ROI Behaviour: All Operators Across
    All Anatomical Regions.

    Rows    : White Matter, Tumor, CSF, Liver, Kidney Edge, Aorta
    Columns : Original | Gaussian | Kantorovich | Bilateral | Wavelet
    """
    print("\n── Generating ROI grid figure …")
    roi_names = list(ROIS.keys())
    col_labels = ['Original'] + OPERATORS
    n_rows = len(roi_names)
    n_cols = len(col_labels)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.5, n_rows * 2.2))
    fig.patch.set_facecolor('#0a0a1a')

    col_colors = {
        'Original':    '#ffffff',
        'Gaussian':    '#4fc3f7',
        'Kantorovich': '#ffb74d',
        'Bilateral':   '#ef5350',
        'Wavelet':     '#ab47bc',
    }

    for c, col in enumerate(col_labels):
        axes[0, c].set_title(col, color=col_colors[col],
                              fontsize=10, fontweight='bold', pad=6)

    for r, roi_name in enumerate(roi_names):
        # Row label
        axes[r, 0].set_ylabel(roi_name.replace(' ', '\n'),
                               color=ROI_COLORS[roi_name],
                               fontsize=8, rotation=0,
                               labelpad=50, va='center')

        # Original ROI
        orig_roi = extract_roi(mid, roi_name)
        axes[r, 0].imshow(orig_roi, cmap='gray', vmin=0, vmax=1)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_edgecolor(ROI_COLORS[roi_name])
            spine.set_linewidth(2)

        # Operator outputs
        for c, op in enumerate(OPERATORS, start=1):
            filt_roi = extract_roi(reconstructed[op], roi_name)
            axes[r, c].imshow(filt_roi, cmap='gray', vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_edgecolor(col_colors[op])
                spine.set_linewidth(1.5)

        for ax in axes[r]:
            ax.set_facecolor('#0a0a1a')

    fig.suptitle(
        'Fig. 7 — ROI Behaviour: All Operators Across All Anatomical Regions\n'
        'Kantorovich: best mean preservation  |  '
        'Bilateral & Wavelet: best edge preservation',
        color='white', fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig('results/fig7_roi_grid.png', dpi=150,
                bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print("── Fig 7 saved → results/fig7_roi_grid.png")


# ══════════════════════════════════════════════
# EXPERIMENT 4 — Mid-slice full comparison
# ══════════════════════════════════════════════

def plot_midslice_comparison(mid, noisy, reconstructed):
    """
    Full mid-slice comparison: Original | Noisy | 4 Operators.
    Includes PSNR, SSIM, MAE in subtitle of each panel.
    """
    print("\n── Generating mid-slice comparison figure …")
    images  = {'Original': mid, 'Noisy Input': noisy, **reconstructed}
    n_imgs  = len(images)

    fig, axes = plt.subplots(1, n_imgs, figsize=(n_imgs * 3, 3.5))
    fig.patch.set_facecolor('#0a0a1a')

    panel_colors = {
        'Original':    'white',
        'Noisy Input': '#ef9a9a',
        'Gaussian':    '#4fc3f7',
        'Kantorovich': '#ffb74d',
        'Bilateral':   '#ef5350',
        'Wavelet':     '#ab47bc',
    }

    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_facecolor('#0a0a1a')
        ax.set_xticks([]); ax.set_yticks([])

        if name not in ('Original', 'Noisy Input'):
            m = all_reconstruction_metrics(mid, img)
            subtitle = (f"PSNR={m['PSNR']:.2f} dB\n"
                        f"SSIM={m['SSIM']:.4f}  MAE={m['MAE']:.4f}")
            ax.set_xlabel(subtitle, color='#cccccc', fontsize=7)

        ax.set_title(name, color=panel_colors.get(name, 'white'),
                     fontsize=9, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_edgecolor(panel_colors.get(name, 'white'))
            spine.set_linewidth(1.5)

    fig.suptitle(
        'Mid-slice reconstruction: Shepp-Logan phantom (64×128×128 volume)\n'
        'Gaussian noise σ=0.05 | All operators applied to noisy input',
        color='white', fontsize=10
    )
    plt.tight_layout()
    plt.savefig('results/fig_mid_slice_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print("── Mid-slice comparison saved → results/fig_mid_slice_comparison.png")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("Sampling Kantorovich Operators — Image Experiment")
    print("Singh & Singh (2026), arXiv:2508.14043")
    print("=" * 60)

    df_table2, mid, noisy, reconstructed = run_table2()
    df_mse = run_mse_3d()
    plot_roi_grid(mid, reconstructed)
    plot_midslice_comparison(mid, noisy, reconstructed)

    print("\n✓ All experiments complete. Results in ./results/")
