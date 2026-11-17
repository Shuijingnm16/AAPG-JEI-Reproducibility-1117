"""JEI supplemental benchmark (v2, corrected FFT PSF centering): frozen AAPG/image-only vs local-PSF Wiener and RL.

Local-PSF approximations are NOT exact space-variant inverses. They use 3x3
(or configured) overlapping image tiles and the known PSF at each tile center.
Use same clean images, fixed crops and PSF parameters for all methods.
No training, no test-set tuning. Report clean and noisy scenarios separately.
"""
import argparse, csv, importlib.util, json, math, random
from pathlib import Path
import numpy as np
import torch
from scipy.signal import fftconvolve
from skimage.restoration import richardson_lucy
from PIL import Image
import aapg_4a_reference as ref

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ev17', HERE/'17_evaluate_generalization_v2_FIXED.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)


def wiener2d(im, psf, reg):
    h, w = im.shape
    # Embed the PSF at the top-left, then shift its *original* center
    # to the FFT origin. ifftshift on the padded array is incorrect.
    kh, kw = psf.shape
    if kh > h or kw > w:
        raise ValueError(f"PSF {psf.shape} exceeds padded image {(h, w)}")
    ker = np.zeros((h, w), dtype=np.float64)
    ker[:kh, :kw] = psf
    ker = np.roll(ker, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))
    H = np.fft.fft2(ker)
    return np.real(np.fft.ifft2(np.conj(H)*np.fft.fft2(im)/(np.abs(H)**2+reg)))


def local_deconvolution(obs, params, method, tiles, overlap, reg, iterations):
    """Local-constant PSF approximation, overlapping crops + Hann-like blending."""
    h,w,c=obs.shape
    accum=np.zeros_like(obs,dtype=np.float64)
    weight_sum=np.zeros((h,w,1),dtype=np.float64)
    tile_h=math.ceil(h/tiles);tile_w=math.ceil(w/tiles)
    halo=max(16,overlap)
    for iy in range(tiles):
        for ix in range(tiles):
            cy0=iy*tile_h;cy1=min(h,(iy+1)*tile_h)
            cx0=ix*tile_w;cx1=min(w,(ix+1)*tile_w)
            if cy0>=cy1 or cx0>=cx1:continue
            y0=max(0,cy0-halo);y1=min(h,cy1+halo)
            x0=max(0,cx0-halo);x1=min(w,cx1+halo)
            fx=2*((cx0+cx1-1)/2)/max(w-1,1)-1
            fy=2*((cy0+cy1-1)/2)/max(h-1,1)-1
            psf=ref.generate_psf(params,fx,fy,ref.PUPIL_SIZE,ref.PSF_SIZE).detach().cpu().numpy().astype(np.float64)
            tile=obs[y0:y1,x0:x1]
            pred=np.zeros_like(tile)
            for ch in range(c):
                plane=tile[:,:,ch]
                if method=='wiener':
                    # Reflection-pad before circular Wiener to reduce edge artifacts.
                    pad=min(max(psf.shape[0]//2,8),min(plane.shape)//2-1)
                    if pad>0:
                        padded=np.pad(plane,((pad,pad),(pad,pad)),mode='reflect')
                        result=wiener2d(padded,psf,reg)[pad:-pad,pad:-pad]
                    else:result=wiener2d(plane,psf,reg)
                else:
                    result=richardson_lucy(np.maximum(plane,1e-6),psf,num_iter=iterations,clip=False)
                pred[:,:,ch]=result
            # Taper only halo margins; central region has unit weight.
            wy=np.ones(y1-y0);wx=np.ones(x1-x0)
            if y0<cy0:wy[:cy0-y0]=np.linspace(.05,1,cy0-y0,endpoint=False)
            if cy1<y1:wy[cy1-y0:]=np.linspace(1,.05,y1-cy1,endpoint=True)
            if x0<cx0:wx[:cx0-x0]=np.linspace(.05,1,cx0-x0,endpoint=False)
            if cx1<x1:wx[cx1-x0:]=np.linspace(1,.05,x1-cx1,endpoint=True)
            ww=(wy[:,None]*wx[None,:])[:,:,None]
            accum[y0:y1,x0:x1]+=pred*ww
            weight_sum[y0:y1,x0:x1]+=ww
    return np.clip(accum/np.maximum(weight_sum,1e-10),0,1).astype(np.float32)


def add_noise(obs, level, seed):
    if level==0:return obs
    rng=np.random.default_rng(seed)
    # Poisson shot noise + read noise, normalized intensity. Not measured sensor noise.
    peak=float(level)
    shot=rng.poisson(np.clip(obs,0,1)*peak).astype(np.float32)/peak
    read=rng.normal(0,0.005,size=obs.shape).astype(np.float32)
    return np.clip(shot+read,0,1)


def metrics(pred,gt):
    x=torch.from_numpy(pred.transpose(2,0,1)).unsqueeze(0)
    y=torch.from_numpy(gt.transpose(2,0,1)).unsqueeze(0)
    return float(ref.per_image_psnr(x,y).item()),float(ref.batch_ssim(x,y).item())


def main():
    p=argparse.ArgumentParser()
    for key in ['images','aapg','image-only','parameters']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('jei_benchmark_results'))
    p.add_argument('--max-images',type=int,default=10)
    p.add_argument('--tiles',type=int,default=3)
    p.add_argument('--overlap',type=int,default=32)
    p.add_argument('--wiener-reg',type=float,default=0.001)
    p.add_argument('--rl-iterations',type=int,default=10)
    p.add_argument('--noise-peaks',type=str,default='0,100,400')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    paths=sorted((x for x in a.images.rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}),key=lambda x:str(x).lower())
    if a.max_images>0:paths=paths[:a.max_images]
    if not paths: p.error('No test images')
    params=ref.load_csv(a.parameters)
    if not params:p.error('Empty parameter table')
    model=ref.FinalAAPG();ck=ev.load_ckpt(a.aapg,model)
    baseline=ev.ImageOnly();ev.load_ckpt(a.image_only,baseline)
    mean=np.asarray(ck['parameter_mean'],dtype=np.float32)
    std=np.asarray(ck['parameter_std'],dtype=np.float32)
    levels=[int(x) for x in a.noise_peaks.split(',')]
    records=[]
    for i,path in enumerate(paths):
        gt=ev.patch(path,'random',i).to(ref.DEVICE)
        param=ev.change_param(params[(i*37+17)%len(params)],index=i)
        with torch.inference_mode():
            degraded=ref.full_physics_degradation(gt,param,ref.GRID_SIZE,ref.PUPIL_SIZE,ref.PSF_SIZE)
        truth=gt.detach().cpu().permute(1,2,0).numpy()
        clean=degraded.detach().cpu().permute(1,2,0).numpy()
        guide=ref.make_physics_maps(param,mean,std,ref.PATCH_SIZE,ref.PATCH_SIZE).unsqueeze(0).to(ref.DEVICE)
        for peak in levels:
            noisy=add_noise(clean,peak,2026+100003*i+peak)
            inp=torch.from_numpy(noisy.transpose(2,0,1).copy()).unsqueeze(0).to(ref.DEVICE)
            with torch.inference_mode():
                aa=model(inp,guide).clamp(0,1).squeeze(0).cpu().permute(1,2,0).numpy()
                bb=baseline(inp).clamp(0,1).squeeze(0).cpu().permute(1,2,0).numpy()
            preds={'Degraded':noisy,'Image-only':bb,'AAPG':aa}
            for name in ['Wiener-local','RL-local']:
                preds[name]=local_deconvolution(noisy,param,'wiener' if name=='Wiener-local' else 'rl',a.tiles,a.overlap,a.wiener_reg,a.rl_iterations)
            for name,pred in preds.items():
                psnr,ssim=metrics(pred,truth)
                records.append(dict(image=path.name,index=i,noise_peak=peak,method=name,psnr=psnr,ssim=ssim))
            print(f'{i+1}/{len(paths)} image={path.name} noise_peak={peak}',flush=True)
    with (a.out/'per_image_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for peak in levels:
        for method in ['Degraded','Image-only','AAPG','Wiener-local','RL-local']:
            rr=[r for r in records if r['noise_peak']==peak and r['method']==method]
            summary.append(dict(noise_peak=peak,method=method,n=len(rr),mean_psnr=float(np.mean([r['psnr'] for r in rr])),mean_ssim=float(np.mean([r['ssim'] for r in rr]))))
    with (a.out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (a.out/'manifest.json').write_text(json.dumps(dict(images=str(a.images),image_count=len(paths),parameters=str(a.parameters),checkpoints=[str(a.aapg),str(a.image_only)],tiles=a.tiles,overlap=a.overlap,wiener_reg=a.wiener_reg,rl_iterations=a.rl_iterations,noise_peaks=levels,seed=2026,caveats=['Local-constant PSF approximation, not exact spatially variant deconvolution','Wiener regularization and RL iterations must be selected on validation data, not test data','Noise scenarios evaluate frozen clean-trained models; no retraining','Synthetic Poisson-Gaussian noise, not measured sensor noise','Verify held-out test images do not overlap training or validation']),indent=2))
    print('Saved:',a.out)
    for s in summary:print(s)
if __name__=='__main__':main()
