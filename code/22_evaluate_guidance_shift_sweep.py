"""Frozen AAPG guidance-alignment sweep; run from the original AAPG_Project/scripts folder.

Requires aapg_4a_reference.py and 17_evaluate_generalization_v2_FIXED.py alongside.
Reuses the exact independent test images and pretrained checkpoints; no retraining.
"""
import argparse, csv, json, importlib.util
from pathlib import Path
import numpy as np
import torch
import aapg_4a_reference as ref

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('eval4c', HERE / '17_evaluate_generalization_v2_FIXED.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
CONDITIONS = ('aligned', 'shift_5pct', 'shift_10pct', 'shift_25pct', 'shift_50pct',
              'shuffle_spatial', 'wrong_image_maps')

def perturb(maps, condition, donor, index):
    if condition == 'aligned': return maps
    if condition.startswith('shift_'):
        frac = int(condition.split('_')[1].replace('pct', '')) / 100.
        return torch.roll(maps, shifts=round(maps.shape[-1] * frac), dims=-1)
    if condition == 'shuffle_spatial':
        # Identical permutation across all 6 channels; retains global value distribution.
        gen = torch.Generator(device='cpu').manual_seed(2026 + index)
        h, w = maps.shape[-2:]
        perm = torch.randperm(h*w, generator=gen).to(maps.device)
        return maps.reshape(1, maps.shape[1], -1)[:, :, perm].reshape_as(maps)
    if condition == 'wrong_image_maps': return donor
    raise ValueError(condition)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, default=ref.PROJECT_ROOT/'data/independent_test_HR')
    p.add_argument('--aapg', type=Path, default=ref.PROJECT_ROOT/'figures/fig12_final_aapg_training_v1/final_aapg_best.pth')
    p.add_argument('--image-only', type=Path, default=ref.PROJECT_ROOT/'figures/fig13_final_image_only_training_v1/final_image_only_best.pth')
    p.add_argument('--out', type=Path, default=ref.PROJECT_ROOT/'figures/fig20_guidance_shift_sweep')
    p.add_argument('--max-images', type=int, default=200)
    p.add_argument('--bootstrap', type=int, default=5000)
    a = p.parse_args()
    if not a.images.is_dir(): p.error(f'Missing image directory: {a.images}')
    paths = sorted((x for x in a.images.rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}), key=lambda x: str(x).lower())
    if a.max_images > 0: paths = paths[:a.max_images]
    if len(paths) < 2: p.error('At least 2 independent test images required')
    if not a.aapg.is_file() or not a.image_only.is_file(): p.error('Missing pretrained checkpoints')
    a.out.mkdir(parents=True, exist_ok=True)
    model = ref.FinalAAPG()
    ck = ev.load_ckpt(a.aapg, model)
    baseline = ev.ImageOnly()
    ev.load_ckpt(a.image_only, baseline)
    mean = np.asarray(ck['parameter_mean'], dtype=np.float32)
    std = np.asarray(ck['parameter_std'], dtype=np.float32)
    params = ref.load_csv(ref.ID_PARAM_CSV)
    if not params: p.error('ID test parameter CSV is empty')
    rows = []
    for i, path in enumerate(paths):
        gt = ev.patch(path, 'random', i).to(ref.DEVICE)
        par = ev.change_param(params[(i*37+17) % len(params)], index=i)
        donor_i = (i + max(1, len(paths)//2)) % len(paths)
        donor_par = ev.change_param(params[(donor_i*37+17) % len(params)], index=donor_i)
        with torch.inference_mode():
            degraded = ref.full_physics_degradation(gt, par, ref.GRID_SIZE, ref.PUPIL_SIZE, ref.PSF_SIZE).unsqueeze(0)
            target = gt.unsqueeze(0)
            maps = ref.make_physics_maps(par, mean, std, ref.PATCH_SIZE, ref.PATCH_SIZE).unsqueeze(0).to(ref.DEVICE)
            donor = ref.make_physics_maps(donor_par, mean, std, ref.PATCH_SIZE, ref.PATCH_SIZE).unsqueeze(0).to(ref.DEVICE)
            bpred = baseline(degraded).clamp(0, 1)
            bpsnr = float(ref.per_image_psnr(bpred, target).item())
            for cond in CONDITIONS:
                guide = perturb(maps, cond, donor, i)
                pred = model(degraded, guide).clamp(0, 1)
                psnr = float(ref.per_image_psnr(pred, target).item())
                rows.append(dict(index=i, image=path.name, condition=cond, psnr=psnr,
                                 ssim=float(ref.batch_ssim(pred, target).item()),
                                 baseline_psnr=bpsnr, gain_vs_baseline=psnr-bpsnr))
        if (i+1) % 10 == 0: print(f'{i+1}/{len(paths)}', flush=True)
    with (a.out/'per_image_metrics.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rng=np.random.default_rng(2026)
    aligned={r['index']:r for r in rows if r['condition']=='aligned'}
    summary=[]
    for cond in CONDITIONS:
        rr=[r for r in rows if r['condition']==cond]
        drops=np.array([aligned[r['index']]['psnr']-r['psnr'] for r in rr])
        gains=np.array([r['gain_vs_baseline'] for r in rr]); n=len(rr)
        idx=rng.integers(0,n,size=(a.bootstrap,n))
        dlo,dhi=np.quantile(drops[idx].mean(axis=1), [.025,.975])
        glo,ghi=np.quantile(gains[idx].mean(axis=1), [.025,.975])
        summary.append(dict(condition=cond,n=n,mean_psnr=np.mean([r['psnr'] for r in rr]),
                            mean_ssim=np.mean([r['ssim'] for r in rr]),gain_vs_baseline=gains.mean(),
                            gain_ci95_low=glo,gain_ci95_high=ghi,drop_from_aligned=drops.mean(),
                            drop_ci95_low=dlo,drop_ci95_high=dhi,
                            win_fraction_vs_baseline=(gains>0).mean()))
    with (a.out/'summary.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (a.out/'manifest.json').write_text(json.dumps(dict(images=len(paths),seed=2026,bootstrap=a.bootstrap,
        conditions=CONDITIONS,shift_axis='horizontal',shift_mode='circular',
        warning='Spatial shuffling and wrong-image guidance are not pure spatial-alignment controls; shifts may also be OOD.'),indent=2))
    print('Saved results to:',a.out)
    for s in summary: print(s)
if __name__=='__main__': main()
