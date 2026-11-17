"""4E-1: audit PSF concentration, field variation and aberration scaling.
Place beside aapg_4a_reference.py in AAPG_Project/scripts.
Does not modify models, checkpoints, datasets, or physics implementation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import aapg_4a_reference as ref

COEFFICIENTS = ('defocus_max', 'astig_max', 'coma_max', 'spherical_center', 'spherical_edge_add')
POSITIONS = {'center': (0.0, 0.0), 'mid_field': (0.5, 0.0), 'corner': (1.0, 1.0)}

def scaled_param(p, factor):
    q = dict(p)
    for k in COEFFICIENTS:
        q[k] = float(q[k]) * factor
    return q

def stats(psf):
    a = psf.detach().cpu().numpy().astype(np.float64)
    a /= a.sum()
    h, w = a.shape
    yy, xx = np.indices(a.shape)
    cy, cx = (h-1)/2, (w-1)/2
    radius = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    order = np.argsort(radius.ravel())
    rr = radius.ravel()[order]
    cumulative = np.cumsum(a.ravel()[order])
    def enclosed(q):
        return float(rr[min(np.searchsorted(cumulative, q), len(rr)-1)])
    y0, x0 = int(cy), int(cx)
    return {
        'sum': float(a.sum()),
        'center_pixel_energy': float(a[y0, x0]),
        'center_3x3_energy': float(a[y0-1:y0+2, x0-1:x0+2].sum()),
        'radius_50_px': enclosed(0.5),
        'radius_90_px': enclosed(0.9),
        'rms_radius_px': float(np.sqrt((a * radius**2).sum())),
        'peak_energy': float(a.max()),
    }, a

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, default=ref.PARAM_CSV)
    parser.add_argument('--out', type=Path, default=ref.PROJECT_ROOT/'figures/fig17_psf_physics_audit')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 100, 1000, 3000])
    parser.add_argument('--factors', type=float, nargs='+', default=[0, 1, 2, 3, 5, 8])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    params = ref.load_csv(args.params)
    if not params:
        parser.error('Parameter CSV is empty')
    indices = sorted(set(i for i in args.indices if 0 <= i < len(params)))
    if not indices:
        parser.error('No valid parameter indices')
    rows, panels = [], []
    for idx in indices:
        for factor in args.factors:
            p = scaled_param(params[idx], factor)
            for position, (fx, fy) in POSITIONS.items():
                psf = ref.generate_psf(p, fx, fy, ref.PUPIL_SIZE, ref.PSF_SIZE)
                s, arr = stats(psf)
                rows.append({'parameter_index': idx, 'factor': factor, 'position': position, **s})
                if idx == indices[0]:
                    panels.append((factor, position, arr))
                print(f'parameter={idx} factor={factor:g} position={position} '
                      f'center={s["center_pixel_energy"]:.4f} '
                      f'r50={s["radius_50_px"]:.2f} r90={s["radius_90_px"]:.2f}', flush=True)
    with (args.out/'psf_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    nrows, ncols = len(args.factors), len(POSITIONS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8*ncols, 3.5*nrows), squeeze=False)
    panel_map = {(factor, position): arr for factor, position, arr in panels}
    for i, factor in enumerate(args.factors):
        for j, position in enumerate(POSITIONS):
            a = panel_map[(factor, position)]
            ax = axes[i,j]
            ax.imshow(np.log10(a+1e-9), cmap='magma', vmin=-9, vmax=0)
            ax.set_title(f'{position} | x{factor:g}')
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('PSF intensity (log10), same display scale')
    fig.tight_layout()
    fig.savefig(args.out/'psf_grid_log.png', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for position in POSITIONS:
        subset = [r for r in rows if r['position']==position]
        for ax, key in zip(axs, ['center_pixel_energy', 'radius_90_px']):
            means = [np.mean([r[key] for r in subset if r['factor']==factor]) for factor in args.factors]
            ax.plot(args.factors, means, marker='o', label=position)
    axs[0].set_ylabel('Center pixel energy fraction')
    axs[1].set_ylabel('90% encircled-energy radius (pixels)')
    for ax in axs:
        ax.set_xlabel('Aberration coefficient multiplier')
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out/'psf_trends.png', dpi=180)
    plt.close(fig)
    summary = {
        'parameter_source': str(args.params),
        'parameter_indices': indices,
        'factors': args.factors,
        'positions': POSITIONS,
        'pupil_fft': ref.PUPIL_SIZE,
        'psf_crop': ref.PSF_SIZE,
        'caveat': 'FFT coordinates are simulation pixels; no sensor-pixel calibration is established. Factor 0 is the diffraction-limited reference, not an impulse kernel.'
    }
    (args.out/'audit_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nSaved:', args.out)
    print('Files: psf_metrics.csv, psf_grid_log.png, psf_trends.png, audit_manifest.json')
    print('Interpretation requires examining PSF widths and shapes, not merely PSNR.')

if __name__ == '__main__':
    main()
