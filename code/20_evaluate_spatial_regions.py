"""4E-2: spatial region restoration audit, frozen models, independent images.
Place beside aapg_4a_reference.py and 17_evaluate_generalization_v2_FIXED.py.
"""
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import aapg_4a_reference as ref
from importlib.util import spec_from_file_location, module_from_spec

SCRIPT = Path(__file__).resolve().parent / '17_evaluate_generalization_v2_FIXED.py'
if not SCRIPT.exists():
    raise SystemExit(f'Missing required evaluator: {SCRIPT}')
spec = spec_from_file_location('eval4c_fixed', SCRIPT)
eval4c = module_from_spec(spec)
spec.loader.exec_module(eval4c)

def masked_psnr(pred, target, mask):
    # MSE across RGB channels and selected pixels, per image.
    err = (pred.float().clamp(0, 1) - target.float()) ** 2
    mse = err[:, :, mask].mean(dim=(1, 2)).clamp_min(1e-12)
    return float((10 * torch.log10(1 / mse))[0].item())

def masks(h, w, device):
    yy, xx = torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    # Inner half in each dimension = 25% of area; border = remaining 75%.
    center = (yy >= h//4) & (yy < h-h//4) & (xx >= w//4) & (xx < w-w//4)
    return {'full':torch.ones((h,w),dtype=torch.bool,device=device),
            'center':center, 'border':~center}

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images',type=Path,default=ref.PROJECT_ROOT/'data/independent_test_HR')
    ap.add_argument('--aapg',type=Path,default=ref.PROJECT_ROOT/'figures/fig12_final_aapg_training_v1/final_aapg_best.pth')
    ap.add_argument('--image-only',type=Path,default=ref.PROJECT_ROOT/'figures/fig13_final_image_only_training_v1/final_image_only_best.pth')
    ap.add_argument('--out',type=Path,default=ref.PROJECT_ROOT/'figures/fig18_spatial_region_audit')
    ap.add_argument('--max-images',type=int,default=200)
    ap.add_argument('--bootstrap',type=int,default=5000)
    ap.add_argument('--scenarios',nargs='+',choices=['ID','OOD_x2','Strong_x3'],default=['ID'])
    ap.add_argument('--skip-hash-audit',action='store_true',help='Not recommended; use only when training files unavailable')
    args=ap.parse_args()
    if not args.images.is_dir(): ap.error(f'Image directory missing: {args.images}')
    exts={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
    images=sorted((p for p in args.images.rglob('*') if p.is_file() and p.suffix.lower() in exts),key=lambda p:str(p).lower())
    if args.max_images>0: images=images[:args.max_images]
    if len(images)<30: ap.error(f'Only {len(images)} images; require >=30')
    if not args.skip_hash_audit:
        known=set()
        for d in (ref.DIV2K_DIR,ref.VAL_DIR):
            if d.is_dir():
                for p in d.rglob('*'):
                    if p.is_file() and p.suffix.lower() in exts: known.add(sha256(p))
        if any(sha256(p) in known for p in images): ap.error('SHA256 overlap with training/validation files')
    trainparams=ref.load_csv(ref.PARAM_CSV)
    testparams=ref.load_csv(ref.ID_PARAM_CSV)
    if not trainparams or not testparams: ap.error('Empty parameter CSV')
    bounds={k:(min(float(ref.parameter_vector(t)[j]) for t in trainparams),max(float(ref.parameter_vector(t)[j]) for t in trainparams)) for j,k in enumerate(ref.PHYSICS_KEYS)}
    a=ref.FinalAAPG()
    ck=eval4c.load_ckpt(args.aapg,a)
    b=eval4c.ImageOnly()
    eval4c.load_ckpt(args.image_only,b)
    mean=np.asarray(ck['parameter_mean'],dtype=np.float32)
    std=np.asarray(ck['parameter_std'],dtype=np.float32)
    args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    settings={'ID':('ID_new_crop',1.),'OOD_x2':('OOD_x2.0',2.),'Strong_x3':('Strong_x3.0',3.)}
    for scenario in args.scenarios:
        label,factor=settings[scenario]
        print(f'[{label}] {len(images)} images',flush=True)
        for i,path in enumerate(images):
            gt=eval4c.patch(path,'random',i).to(ref.DEVICE)
            param=testparams[(i*37+17)%len(testparams)]
            true_param=(eval4c.force_ood(param,bounds,factor) if scenario=='OOD_x2'
                        else eval4c.change_param(param,factor=factor,index=i))
            with torch.no_grad():
                deg=ref.full_physics_degradation(gt,true_param,ref.GRID_SIZE,ref.PUPIL_SIZE,ref.PSF_SIZE)
                maps=ref.make_physics_maps(true_param,mean,std,ref.PATCH_SIZE,ref.PATCH_SIZE).unsqueeze(0).to(ref.DEVICE)
                x=deg.unsqueeze(0); y=gt.unsqueeze(0)
                pred_a=a(x,maps).clamp(0,1)
                pred_b=b(x).clamp(0,1)
                for region,mask in masks(gt.shape[-2],gt.shape[-1],ref.DEVICE).items():
                    ip=masked_psnr(x,y,mask)
                    bp=masked_psnr(pred_b,y,mask)
                    ap_=masked_psnr(pred_a,y,mask)
                    records.append(dict(scenario=label,image=path.name,region=region,
                                        input_psnr=ip,image_only_psnr=bp,aapg_psnr=ap_,delta_psnr=ap_-bp))
            if (i+1)%10==0: print(f'  {i+1}/{len(images)}',flush=True)
    with (args.out/'per_image_region_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    rng=np.random.default_rng(2026)
    summary=[]
    for scenario in args.scenarios:
        label,_=settings[scenario]
        for region in ('full','center','border'):
            rr=[r for r in records if r['scenario']==label and r['region']==region]
            vals=np.array([r['delta_psnr'] for r in rr]); n=len(vals)
            means=np.empty(args.bootstrap)
            for j in range(args.bootstrap): means[j]=vals[rng.integers(0,n,n)].mean()
            ci=np.quantile(means,[.025,.975])
            summary.append(dict(scenario=label,region=region,n=n,
                input_psnr=float(np.mean([r['input_psnr'] for r in rr])),
                image_only_psnr=float(np.mean([r['image_only_psnr'] for r in rr])),
                aapg_psnr=float(np.mean([r['aapg_psnr'] for r in rr])),
                mean_delta_psnr=float(vals.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                fraction_aapg_better=float((vals>0).mean())))
    with (args.out/'region_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    # Paired difference-of-gains by image (border minus center); this is the core hypothesis.
    contrast=[]
    for scenario in args.scenarios:
        label,_=settings[scenario]
        centers={r['image']:r['delta_psnr'] for r in records if r['scenario']==label and r['region']=='center'}
        borders={r['image']:r['delta_psnr'] for r in records if r['scenario']==label and r['region']=='border'}
        vals=np.array([borders[k]-centers[k] for k in sorted(centers)]); n=len(vals)
        means=np.empty(args.bootstrap)
        for j in range(args.bootstrap): means[j]=vals[rng.integers(0,n,n)].mean()
        ci=np.quantile(means,[.025,.975])
        contrast.append(dict(scenario=label,n=n,border_minus_center_gain=float(vals.mean()),
                             ci95_low=float(ci[0]),ci95_high=float(ci[1]),fraction_border_gain_larger=float((vals>0).mean())))
    with (args.out/'border_vs_center_contrast.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(contrast[0]));w.writeheader();w.writerows(contrast)
    fig,ax=plt.subplots(figsize=(8,4.5))
    labels=[r['scenario'] for r in contrast]
    x=np.arange(len(labels)); width=.35
    c=[next(r for r in summary if r['scenario']==lab and r['region']=='center')['mean_delta_psnr'] for lab in labels]
    e=[next(r for r in summary if r['scenario']==lab and r['region']=='border')['mean_delta_psnr'] for lab in labels]
    ax.bar(x-width/2,c,width,label='Center (25% area)')
    ax.bar(x+width/2,e,width,label='Border (75% area)')
    ax.set_xticks(x,labels);ax.set_ylabel('AAPG - Image-Only PSNR (dB)')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(args.out/'regional_psnr_gain.png',dpi=220);plt.close(fig)
    manifest=dict(image_count=len(images),scenario_names=args.scenarios,patch_mode='random',seed=eval4c.SEED,
        parameter_source=str(ref.ID_PARAM_CSV),region_definition='central half-width by half-height = 25% pixels; complement = 75%',
        psnr_definition='10 log10(1/mean squared error over RGB and selected pixels)',
        note='Per-region PSNR is computed from region-specific MSE. Differences are paired by image. Same ID parameter pairing/crops as 4D.')
    (args.out/'region_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('\n=== REGIONAL SUMMARY ===')
    for r in summary: print(r)
    print('\n=== BORDER MINUS CENTER GAIN ===')
    for r in contrast: print(r)
    print('Saved to:',args.out)

if __name__=='__main__': main()
