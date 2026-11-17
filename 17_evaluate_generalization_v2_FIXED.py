"""
Experiment 4C: frozen-model generalization and physics mismatch audit.
Place this file and aapg_4a_reference.py together in AAPG_Project/scripts.
No training. No parameter fitting on test data.
"""
import argparse, csv, json, random, hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aapg_4a_reference as ref

KEYS = ref.PHYSICS_KEYS
SEED = 2026

class ImageOnly(torch.nn.Module):
    def __init__(self, channels=56, blocks=5):
        super().__init__()
        self.head=torch.nn.Conv2d(3,channels,3,padding=1)
        self.blocks=torch.nn.ModuleList([ref.PhysicsGuidedBlock(channels) for _ in range(blocks)])
        self.tail=torch.nn.Conv2d(channels,3,3,padding=1)
    def forward(self,x):
        h=torch.nn.functional.relu(self.head(x),inplace=False)
        for b in self.blocks: h=b(h)
        return x+self.tail(h)

def load_ckpt(path, model):
    # Full checkpoint is a trusted artifact from user's own training.
    try: c=torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: c=torch.load(path,map_location="cpu")
    model.load_state_dict(c["model_state_dict"],strict=True)
    model.to(ref.DEVICE).eval()
    return c

def patch(path, mode, index):
    im=Image.open(path).convert("RGB")
    w,h=im.size; s=ref.PATCH_SIZE
    if min(w,h)<s:
        scale=max(s/w,s/h)
        im=im.resize((int(np.ceil(w*scale)),int(np.ceil(h*scale))),Image.Resampling.BICUBIC)
        w,h=im.size
    if mode=="center": x,y=(w-s)//2,(h-s)//2
    else:
        rng=random.Random(SEED+900000+index*100003)
        x=rng.randint(0,w-s); y=rng.randint(0,h-s)
    return ref.pil_to_tensor(im.crop((x,y,x+s,y+s)))

def change_param(param, factor=1., mismatch=0., index=0):
    q=dict(param)
    rng=np.random.default_rng(SEED+80000+index)
    for k in KEYS:
        v=float(q.get(k, ref.parameter_vector(q)[KEYS.index(k)]))
        if k=="field_power":
            q[k]=max(.25,v+(factor-1)*.4 + (rng.normal(0,mismatch*.4) if mismatch else 0))
        else:
            q[k]=v*factor*(1+rng.normal(0,mismatch) if mismatch else 1)
    return q

def get_metrics(x,y):
    return (float(ref.per_image_psnr(x,y).item()),
            float(ref.batch_ssim(x,y).item()))

def outside_train_range(param, bounds):
    return any(float(param[k]) < bounds[k][0] or float(param[k]) > bounds[k][1] for k in KEYS)

def force_ood(param, bounds, multiplier):
    """Guarantee all five aberration coefficients exceed training maxima.
    field_power remains unchanged to avoid mixing geometry and amplitude shifts.
    """
    q=dict(param)
    for k in KEYS:
        if k != "field_power":
            q[k]=max(float(q[k]),bounds[k][1])*multiplier
    return q

def make_error_panel(images, label, out):
    gt,deg,baseline,aapg=images
    err_b=np.abs(baseline-gt).mean(axis=2)
    err_a=np.abs(aapg-gt).mean(axis=2)
    vmax=max(float(np.quantile(np.concatenate([err_b.ravel(),err_a.ravel()]),.99)),1e-5)
    fig,axs=plt.subplots(2,4,figsize=(14,7))
    for ax,im,title in zip(axs[0],images,["GT","Degraded","Image-Only","Spatial-AAPG"]):
        ax.imshow(im);ax.set_title(title);ax.axis("off")
    for ax,err,title in [(axs[1,0],err_b,"Image-Only error"),(axs[1,1],err_a,"AAPG error")]:
        ax.imshow(err,cmap="magma",vmin=0,vmax=vmax);ax.set_title(title);ax.axis("off")
    # 2x center zooms show where details diverge without altering PSNR evaluation.
    h,w=gt.shape[:2]; x0,y0=w//4,h//4;x1,y1=3*w//4,3*h//4
    axs[1,2].imshow(baseline[y0:y1,x0:x1]);axs[1,2].set_title("Image-Only zoom");axs[1,2].axis("off")
    axs[1,3].imshow(aapg[y0:y1,x0:x1]);axs[1,3].set_title("AAPG zoom");axs[1,3].axis("off")
    fig.suptitle(label);fig.tight_layout();fig.savefig(out,dpi=200);plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--aapg",type=Path,default=ref.PROJECT_ROOT/"figures/fig12_final_aapg_training_v1/final_aapg_best.pth")
    ap.add_argument("--image-only",type=Path,default=ref.PROJECT_ROOT/"figures/fig13_final_image_only_training_v1/final_image_only_best.pth")
    ap.add_argument("--images",type=Path,default=ref.PROJECT_ROOT/"data/DIV2K_valid_HR")
    ap.add_argument("--out",type=Path,default=ref.PROJECT_ROOT/"figures/fig15_generalization_audit_v2")
    ap.add_argument("--max-images",type=int,default=30,help="0 = all images; start with 10 for smoke test")
    ap.add_argument("--mode",choices=["random","center"],default="random")
    ap.add_argument("--allow-validation-images",action="store_true",help="Explicitly permit previously used DIV2K validation images; exploratory only")
    args=ap.parse_args()
    # A true held-out-image test requires a different directory than model-selection images.
    is_validation=args.images.resolve()==ref.VAL_DIR.resolve()
    if is_validation and not args.allow_validation_images:
        ap.error("Default DIV2K_valid_HR was already used for model selection. Supply --images /path/to/disjoint_images or explicitly add --allow-validation-images for exploratory validation-pool tests.")
    args.out.mkdir(parents=True,exist_ok=True)
    if not args.images.is_dir():
        raise RuntimeError(f"Image directory not found: {args.images}")
    supported={".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images=sorted((p for p in args.images.rglob("*")
                   if p.is_file() and p.suffix.lower() in supported),
                  key=lambda p: str(p).lower())
    if not images:
        raise RuntimeError(f"No supported image files in {args.images}; expected PNG/JPG/JPEG/TIF/TIFF/BMP")
    if args.max_images>0: images=images[:args.max_images]
    trainparams=ref.load_csv(ref.PARAM_CSV)
    testparams=ref.load_csv(ref.ID_PARAM_CSV)
    if not testparams: raise RuntimeError("Empty id_test_parameters.csv")
    if not trainparams: raise RuntimeError("Empty train_parameters.csv")
    bounds={k:(min(float(ref.parameter_vector(t)[j]) for t in trainparams),max(float(ref.parameter_vector(t)[j]) for t in trainparams)) for j,k in enumerate(KEYS)}
    a=ref.FinalAAPG()
    ck=load_ckpt(args.aapg,a)
    b=ImageOnly()
    load_ckpt(args.image_only,b)
    mean=np.asarray(ck["parameter_mean"],dtype=np.float32)
    std=np.asarray(ck["parameter_std"],dtype=np.float32)
    # All conditions use same images/crops. Mismatch changes ONLY guidance maps.
    scenarios=[("ID_new_crop",1.,0.),("OOD_x1.5",1.5,0.),
               ("OOD_x2.0",2.,0.),("Strong_x3.0",3.,0.),("Strong_x5.0",5.,0.),("Strong_x8.0",8.,0.),
               ("Mismatch_0pct",1.,0.),("Mismatch_10pct",1.,.10),
               ("Mismatch_25pct",1.,.25),("Mismatch_50pct",1.,.50)]
    records=[]
    examples={}
    for label,factor,mis in scenarios:
        print(f"\n[{label}] {len(images)} images",flush=True)
        for i,path in enumerate(images):
            gt=patch(path,args.mode,i).to(ref.DEVICE)
            # Use test parameters, with a different index pairing than 4A validation.
            param=testparams[(i*37+17)%len(testparams)]
            true_param=(force_ood(param,bounds,factor) if label.startswith("OOD_") else change_param(param,factor=factor,index=i))
            is_ood=outside_train_range(true_param,bounds)
            map_param=change_param(true_param,mismatch=mis,index=i)
            with torch.no_grad():
                deg=ref.full_physics_degradation(gt,true_param,
                    ref.GRID_SIZE,ref.PUPIL_SIZE,ref.PSF_SIZE)
                maps=ref.make_physics_maps(map_param,mean,std,ref.PATCH_SIZE,ref.PATCH_SIZE).unsqueeze(0).to(ref.DEVICE)
                x=deg.unsqueeze(0); y=gt.unsqueeze(0)
                pred_a=a(x,maps).clamp(0,1)
                pred_b=b(x).clamp(0,1)
                ip,iss=get_metrics(x,y)
                bp,bss=get_metrics(pred_b,y)
                ap_,ass=get_metrics(pred_a,y)
            records.append(dict(scenario=label,image=path.name,
                input_psnr=ip,input_ssim=iss,
                image_only_psnr=bp,image_only_ssim=bss,
                aapg_psnr=ap_,aapg_ssim=ass,delta_psnr=ap_-bp,
                factor=factor,mismatch=mis,outside_train_range=is_ood))
            if i==0:
                examples[label]=[t.detach().cpu().squeeze(0).permute(1,2,0).numpy().clip(0,1)
                    for t in (y,x,pred_b,pred_a)]
            if (i+1)%10==0: print(f"  {i+1}/{len(images)}",flush=True)
    cols=list(records[0])
    with (args.out/"per_image_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(records)
    fields=["scenario","n","input_psnr","image_only_psnr","aapg_psnr",
            "delta_psnr","input_ssim","image_only_ssim","aapg_ssim"]
    summary=[]
    for label,_,_ in scenarios:
        rows=[r for r in records if r["scenario"]==label]
        summary.append(dict(scenario=label,n=len(rows),
            **{k:round(float(np.mean([r[k] for r in rows])),5) for k in fields[2:]}))
    with (args.out/"summary_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    fig,ax=plt.subplots(figsize=(9,5))
    labs=[s["scenario"] for s in summary]
    x=np.arange(len(labs)); width=.35
    ax.bar(x-width/2,[s["image_only_psnr"] for s in summary],width,label="Image-Only")
    ax.bar(x+width/2,[s["aapg_psnr"] for s in summary],width,label="Spatial-AAPG")
    ax.set_xticks(x,labs,rotation=35,ha="right")
    ax.set_ylabel("Mean per-image PSNR (dB)");ax.legend();ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(args.out/"robustness_psnr.png",dpi=220);plt.close(fig)
    for label,ims in examples.items():
        make_error_panel(ims,label,args.out/f"error_zoom_{label}.png")
        fig,axs=plt.subplots(1,4,figsize=(14,3.7))
        for ax,img,title in zip(axs,ims,["GT","Degraded","Image-Only","Spatial-AAPG"]):
            ax.imshow(img);ax.set_title(title);ax.axis("off")
        fig.suptitle(label);fig.tight_layout()
        fig.savefig(args.out/f"example_{label}.png",dpi=180);plt.close(fig)
    manifest=dict(image_dir=str(args.images),image_count=len(images),
        image_split=("reused_validation_pool" if is_validation else "user_supplied_candidate_heldout_verify_manually"),
        train_parameter_bounds={k:list(v) for k,v in bounds.items()},
        patch_mode=args.mode,seed=SEED,parameter_source=str(ref.ID_PARAM_CSV),
        train_parameter_source=str(ref.PARAM_CSV),
        aapg_checkpoint=str(args.aapg),image_only_checkpoint=str(args.image_only),
        notes=[
            "Frozen checkpoints, no training.",
            "If --allow-validation-images was used, results are exploratory only. Other image directories require manual disjointness verification.",
            "ID_new_crop uses new crop/parameter pairing; not independent-image test on default data.",
            "Strong_x3/x5/x8 scale aberration coefficients; actual input PSNR is measured, not guaranteed 25-30 dB.",
            "OOD scenarios force the five aberration amplitudes above the training maxima; field_power is held constant.",
            "Mismatch perturbations affect guidance only, not true degradation.",
            "SSIM uses original 4A implementation."
        ])
    (args.out/"evaluation_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print("\n=== RESULTS ===")
    for s in summary: print(s)
    print("Saved to:",args.out)

if __name__=="__main__":
    main()
