"""Frozen AAPG control: smooth, in-distribution coefficient misassignment.

Uses the same deterministic images, crops, degradations and checkpoint loading as
22_evaluate_guidance_shift_sweep.py. No training and no metric fabrication.

For each test image, true coefficients p are fixed for degradation. Guidance uses
(1-alpha)*p + alpha*q where q is a different valid ID-test parameter row.
Convex interpolation retains the parameter range and smooth analytic maps.
This does NOT guarantee that intermediate maps were encountered during training.
"""
import argparse, csv, json, importlib.util
from pathlib import Path
import numpy as np
import torch
import aapg_4a_reference as ref

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('eval4c', HERE/'17_evaluate_generalization_v2_FIXED.py')
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
ALPHAS=(0.0,0.1,0.25,0.5,1.0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--aapg',type=Path,required=True)
    p.add_argument('--image-only',type=Path,required=True)
    p.add_argument('--parameters',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('smooth_guidance_audit'))
    p.add_argument('--max-images',type=int,default=200)
    p.add_argument('--bootstrap',type=int,default=5000)
    a=p.parse_args()
    paths=sorted((x for x in a.images.rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}),key=lambda x:str(x).lower())
    if a.max_images>0: paths=paths[:a.max_images]
    if not paths: p.error('No test images found')
    if len(paths)<2: p.error('At least 2 images needed for donor pairing')
    params=ref.load_csv(a.parameters)
    if not params: p.error('No optical parameters')
    a.out.mkdir(parents=True,exist_ok=True)
    model=ref.FinalAAPG(); ck=ev.load_ckpt(a.aapg,model)
    baseline=ev.ImageOnly(); ev.load_ckpt(a.image_only,baseline)
    mean=np.asarray(ck['parameter_mean'],dtype=np.float32)
    std=np.asarray(ck['parameter_std'],dtype=np.float32)
    rows=[]
    for i,path in enumerate(paths):
        gt=ev.patch(path,'random',i).to(ref.DEVICE)
        true=ev.change_param(params[(i*37+17)%len(params)],index=i)
        donor_i=(i+max(1,len(paths)//2))%len(paths)
        donor=ev.change_param(params[(donor_i*37+17)%len(params)],index=donor_i)
        pvec=ref.parameter_vector(true); qvec=ref.parameter_vector(donor)
        with torch.inference_mode():
            degraded=ref.full_physics_degradation(gt,true,ref.GRID_SIZE,ref.PUPIL_SIZE,ref.PSF_SIZE).unsqueeze(0)
            target=gt.unsqueeze(0)
            bpsnr=float(ref.per_image_psnr(baseline(degraded).clamp(0,1),target).item())
            for alpha in ALPHAS:
                mixed=(1-alpha)*pvec+alpha*qvec
                mixed_dict={k:float(v) for k,v in zip(ref.PHYSICS_KEYS,mixed)}
                guide=ref.make_physics_maps(mixed_dict,mean,std,ref.PATCH_SIZE,ref.PATCH_SIZE).unsqueeze(0).to(ref.DEVICE)
                pred=model(degraded,guide).clamp(0,1)
                psnr=float(ref.per_image_psnr(pred,target).item())
                rows.append(dict(index=i,image=path.name,alpha=alpha,psnr=psnr,
                                 ssim=float(ref.batch_ssim(pred,target).item()),
                                 baseline_psnr=bpsnr,gain_vs_baseline=psnr-bpsnr,
                                 relative_parameter_distance=float(np.linalg.norm(pvec-qvec)/(np.linalg.norm(pvec)+1e-8))))
        if (i+1)%10==0:print(f'{i+1}/{len(paths)}',flush=True)
    with (a.out/'per_image_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rng=np.random.default_rng(2026)
    aligned={r['index']:r for r in rows if r['alpha']==0.0}
    summary=[]
    for alpha in ALPHAS:
        rr=[r for r in rows if r['alpha']==alpha]
        gains=np.array([r['gain_vs_baseline'] for r in rr]); drops=np.array([aligned[r['index']]['psnr']-r['psnr'] for r in rr]);n=len(rr)
        indices=rng.integers(0,n,size=(a.bootstrap,n))
        glo,ghi=np.quantile(gains[indices].mean(axis=1),[.025,.975]);dlo,dhi=np.quantile(drops[indices].mean(axis=1),[.025,.975])
        summary.append(dict(alpha=alpha,n=n,mean_psnr=float(np.mean([r['psnr'] for r in rr])),
                            mean_ssim=float(np.mean([r['ssim'] for r in rr])),gain_vs_baseline=float(gains.mean()),
                            gain_ci95_low=float(glo),gain_ci95_high=float(ghi),
                            drop_from_aligned=float(drops.mean()),drop_ci95_low=float(dlo),drop_ci95_high=float(dhi),
                            win_fraction_vs_baseline=float(np.mean(gains>0))))
    with (a.out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (a.out/'manifest.json').write_text(json.dumps(dict(n_images=len(paths),bootstrap=a.bootstrap,seed=2026,alphas=ALPHAS,
        intervention='Convex interpolation of six optical coefficients between original and a valid donor row',
        caveat='Donor rows are from ID-test parameter table; not proof of training distribution membership or causal identification.'),indent=2))
    print('Saved',a.out)
    for r in summary:print(r)
if __name__=='__main__':main()
