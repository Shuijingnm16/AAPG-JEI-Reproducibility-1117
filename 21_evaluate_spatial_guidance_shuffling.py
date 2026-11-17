"""Experiment 4F: frozen-model spatial-guidance correspondence audit.

Run from AAPG_Project/scripts or provide --images. No training and no metric fabrication.
The same degraded observation is reused for every guidance intervention.
"""
import argparse, csv, json, hashlib, sys
from pathlib import Path
import numpy as np
import torch
import aapg_4a_reference as ref
import importlib.util

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('eval4c',HERE/'17_evaluate_generalization_v2_FIXED.py')
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

CONDITIONS=('aligned','roll_half','flip_horizontal','flip_vertical','wrong_image_maps','zero_maps')

def alter(maps, condition, donor=None):
    if condition=='aligned': return maps
    if condition=='roll_half': return torch.roll(maps,shifts=(maps.shape[-2]//2,maps.shape[-1]//2),dims=(-2,-1))
    if condition=='flip_horizontal': return torch.flip(maps,dims=(-1,))
    if condition=='flip_vertical': return torch.flip(maps,dims=(-2,))
    if condition=='wrong_image_maps': return donor
    if condition=='zero_maps': return torch.zeros_like(maps)
    raise ValueError(condition)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,default=ref.PROJECT_ROOT/'data/independent_test_HR')
    p.add_argument('--aapg',type=Path,default=ref.PROJECT_ROOT/'figures/fig12_final_aapg_training_v1/final_aapg_best.pth')
    p.add_argument('--image-only',type=Path,default=ref.PROJECT_ROOT/'figures/fig13_final_image_only_training_v1/final_image_only_best.pth')
    p.add_argument('--out',type=Path,default=ref.PROJECT_ROOT/'figures/fig19_spatial_guidance_shuffle')
    p.add_argument('--max-images',type=int,default=200,help='Use 20 for smoke test, 200 for final')
    p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--scenario',choices=['ID','Strong_x3.0'],default='ID')
    a=p.parse_args()
    if not a.images.is_dir(): p.error(f'Missing images: {a.images}')
    paths=sorted((x for x in a.images.rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}),key=lambda x:str(x).lower())
    if a.max_images>0: paths=paths[:a.max_images]
    if len(paths)<2: p.error('At least two images are required')
    if not a.aapg.exists() or not a.image_only.exists(): p.error('Missing checkpoints; specify --aapg and --image-only')
    a.out.mkdir(parents=True,exist_ok=True)
    model=ref.FinalAAPG()
    ck=ev.load_ckpt(a.aapg,model)
    baseline=ev.ImageOnly()
    ev.load_ckpt(a.image_only,baseline)
    mean=np.asarray(ck['parameter_mean'],dtype=np.float32)
    std=np.asarray(ck['parameter_std'],dtype=np.float32)
    testparams=ref.load_csv(ref.ID_PARAM_CSV)
    trainparams=ref.load_csv(ref.PARAM_CSV)
    bounds={k:(min(float(ref.parameter_vector(t)[j]) for t in trainparams),max(float(ref.parameter_vector(t)[j]) for t in trainparams)) for j,k in enumerate(ref.PHYSICS_KEYS)}
    if not testparams: p.error('Empty test parameters')
    rows=[]
    rng=np.random.default_rng(2026)
    for i,path in enumerate(paths):
        gt=ev.patch(path,'random',i).to(ref.DEVICE)
        par=testparams[(i*37+17)%len(testparams)]
        true_param=ev.change_param(par,factor=3.,index=i) if a.scenario=='Strong_x3.0' else ev.change_param(par,index=i)
        donor_idx=(i+max(1,len(paths)//2))%len(paths)
        donor_par=testparams[(donor_idx*37+17)%len(testparams)]
        donor_param=ev.change_param(donor_par,factor=3.,index=donor_idx) if a.scenario=='Strong_x3.0' else ev.change_param(donor_par,index=donor_idx)
        with torch.inference_mode():
            degraded=ref.full_physics_degradation(gt,true_param,ref.GRID_SIZE,ref.PUPIL_SIZE,ref.PSF_SIZE).unsqueeze(0)
            target=gt.unsqueeze(0)
            maps=ref.make_physics_maps(true_param,mean,std,ref.PATCH_SIZE,ref.PATCH_SIZE).unsqueeze(0).to(ref.DEVICE)
            donor=ref.make_physics_maps(donor_param,mean,std,ref.PATCH_SIZE,ref.PATCH_SIZE).unsqueeze(0).to(ref.DEVICE)
            bpred=baseline(degraded).clamp(0,1)
            bpsnr=float(ref.per_image_psnr(bpred,target).item())
            for condition in CONDITIONS:
                use=alter(maps,condition,donor)
                pred=model(degraded,use).clamp(0,1)
                psnr=float(ref.per_image_psnr(pred,target).item())
                ssim=float(ref.batch_ssim(pred,target).item())
                rows.append(dict(scenario=a.scenario,image=path.name,index=i,condition=condition,psnr=psnr,ssim=ssim,baseline_psnr=bpsnr,gain_vs_baseline=psnr-bpsnr))
        if (i+1)%10==0: print(f'Processed {i+1}/{len(paths)}',flush=True)
    with (a.out/'per_image_shuffle_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aligned={r['index']:r for r in rows if r['condition']=='aligned'}
    stats=[]
    for condition in CONDITIONS:
        rr=[r for r in rows if r['condition']==condition]
        psnr=np.array([r['psnr'] for r in rr]); drops=np.array([aligned[r['index']]['psnr']-r['psnr'] for r in rr]);n=len(rr)
        ids=rng.integers(0,n,size=(a.bootstrap,n))
        lo,hi=np.quantile(drops[ids].mean(axis=1),[.025,.975])
        stats.append(dict(condition=condition,n=n,mean_psnr=float(psnr.mean()),mean_ssim=float(np.mean([r['ssim'] for r in rr])),mean_drop_from_aligned=float(drops.mean()),ci95_low=float(lo),ci95_high=float(hi),fraction_aligned_better=float((drops>0).mean())))
    with (a.out/'shuffle_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    manifest=dict(scenario=a.scenario,image_count=len(paths),conditions=list(CONDITIONS),seed=2026,bootstrap=a.bootstrap,checkpoints=[str(a.aapg),str(a.image_only)],notes=['Frozen checkpoints; image degradation unchanged across conditions.','roll_half and flips preserve within-map pixel-value histograms, but not spatial alignment.','wrong_image_maps changes map parameters and spatial information; not a pure alignment control.','zero_maps is out-of-distribution guidance and is not evidence of causality.','Results are conditional on synthetic optical model and selected test images.'])
    (a.out/'shuffle_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nSUMMARY')
    for s in stats: print(s)
    print('Saved:',a.out)

if __name__=='__main__': main()
