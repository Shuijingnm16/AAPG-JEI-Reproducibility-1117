"""Tune local PSF Wiener/RL hyperparameters on DIV2K validation images only.
Uses the exact same degradation, patch, PSF, and metrics code as benchmark 24.
Do not tune on independent_test_HR. No model checkpoints required.
"""
import argparse, csv, importlib.util, json
from pathlib import Path
import numpy as np
import torch
import aapg_4a_reference as ref

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('benchmark24', HERE/'24_jei_classical_noise_benchmark_FIXED.py')
bm=importlib.util.module_from_spec(spec);spec.loader.exec_module(bm)
ev=bm.ev

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--parameters',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('jei_validation_tuning'))
    ap.add_argument('--max-images',type=int,default=10)
    ap.add_argument('--tiles',type=int,default=3)
    ap.add_argument('--overlap',type=int,default=32)
    ap.add_argument('--wiener-regs',default='0.0001,0.001,0.01')
    ap.add_argument('--rl-iterations',default='5,10,20')
    args=ap.parse_args()
    regs=[float(s) for s in args.wiener_regs.split(',')]
    iters=[int(s) for s in args.rl_iterations.split(',')]
    if not regs or not iters or any(v<=0 for v in regs+iters):ap.error('Positive nonempty parameter grids required')
    if args.images.resolve()==(ref.PROJECT_ROOT/'data'/'independent_test_HR').resolve():
        ap.error('Do not tune on independent test images; use DIV2K_valid_HR')
    paths=sorted((p for p in args.images.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}),key=lambda p:str(p).lower())
    if args.max_images>0:paths=paths[:args.max_images]
    if not paths:ap.error('No validation images found')
    params=ref.load_csv(args.parameters)
    if not params:ap.error('Empty parameter CSV')
    args.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,path in enumerate(paths):
        gt=ev.patch(path,'random',i).to(ref.DEVICE)
        param=ev.change_param(params[(i*37+17)%len(params)],index=i)
        with torch.inference_mode():
            deg=ref.full_physics_degradation(gt,param,ref.GRID_SIZE,ref.PUPIL_SIZE,ref.PSF_SIZE)
        truth=gt.detach().cpu().permute(1,2,0).numpy()
        clean=deg.detach().cpu().permute(1,2,0).numpy()
        for reg in regs:
            pred=bm.local_deconvolution(clean,param,'wiener',args.tiles,args.overlap,reg,iters[0])
            psnr,ssim=bm.metrics(pred,truth)
            records.append(dict(image=path.name,index=i,method='Wiener-local',parameter=reg,psnr=psnr,ssim=ssim))
        for it in iters:
            pred=bm.local_deconvolution(clean,param,'rl',args.tiles,args.overlap,regs[0],it)
            psnr,ssim=bm.metrics(pred,truth)
            records.append(dict(image=path.name,index=i,method='RL-local',parameter=it,psnr=psnr,ssim=ssim))
        print(f'{i+1}/{len(paths)} {path.name}',flush=True)
    with (args.out/'per_image_tuning.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for method,values in [('Wiener-local',regs),('RL-local',iters)]:
        for v in values:
            rr=[r for r in records if r['method']==method and r['parameter']==v]
            summary.append(dict(method=method,parameter=v,n=len(rr),mean_psnr=float(np.mean([r['psnr'] for r in rr])),mean_ssim=float(np.mean([r['ssim'] for r in rr]))))
    with (args.out/'tuning_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    best={m:max((s for s in summary if s['method']==m),key=lambda s:s['mean_psnr']) for m in ('Wiener-local','RL-local')}
    (args.out/'selected_parameters.json').write_text(json.dumps(dict(selection_metric='validation mean PSNR',selected_wiener_reg=best['Wiener-local']['parameter'],selected_rl_iterations=int(best['RL-local']['parameter']),validation_images=str(args.images),n_images=len(paths),noise_peak=0,seed=2026,tiles=args.tiles,overlap=args.overlap,note='Select on validation data only; freeze for independent test. Noise-specific tuning not performed.'),indent=2),encoding='utf-8')
    print('\nVALIDATION SUMMARY')
    for s in summary:print(s)
    print('\nSELECTED:',(args.out/'selected_parameters.json').read_text())
    print('Saved:',args.out)
if __name__=='__main__':main()
