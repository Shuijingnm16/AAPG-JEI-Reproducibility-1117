"""Experiment 4D: independent-image audit and paired uncertainty estimates.
Runs the existing frozen 4C evaluator; refuses image duplicates by SHA256.
Requires a truly separate image directory and a separately generated test-parameter CSV
for a fully independent evaluation (parameter file can be passed with --test-params).
"""
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import aapg_4a_reference as ref

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--test-params',type=Path,default=None,help='Independently generated CSV; required for full parameter independence')
    p.add_argument('--aapg',type=Path,default=ref.PROJECT_ROOT/'figures/fig12_final_aapg_training_v1/final_aapg_best.pth')
    p.add_argument('--image-only',type=Path,default=ref.PROJECT_ROOT/'figures/fig13_final_image_only_training_v1/final_image_only_best.pth')
    p.add_argument('--out',type=Path,default=ref.PROJECT_ROOT/'figures/fig16_independent_audit_v1')
    p.add_argument('--max-images',type=int,default=100)
    p.add_argument('--bootstrap',type=int,default=5000)
    a=p.parse_args()
    if not a.images.is_dir(): p.error(f'Image directory does not exist: {a.images}')
    supported={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
    imgs=sorted((f for f in a.images.rglob('*') if f.is_file() and f.suffix.lower() in supported), key=lambda f: str(f).lower())
    if a.max_images>0: imgs=imgs[:a.max_images]
    if len(imgs)<30: p.error('At least 30 test images required; recommend >=100.')
    forbidden=[]
    for folder in (ref.DIV2K_DIR, ref.VAL_DIR):
        if folder.is_dir():
            forbidden.extend(f for f in folder.rglob('*') if f.is_file() and f.suffix.lower() in supported)
    known={sha256(f) for f in forbidden}
    overlaps=[str(f) for f in imgs if sha256(f) in known]
    if overlaps: p.error('Image SHA256 overlap with training/validation: '+str(overlaps[:5]))
    if a.images.resolve() in (ref.DIV2K_DIR.resolve(),ref.VAL_DIR.resolve()):
        p.error('Training/validation image directories are forbidden.')
    a.out.mkdir(parents=True,exist_ok=True)
    # Existing evaluator imports parameter paths from ref. Override through a tiny
    # in-process runner, so the core model/PSF/metric implementation stays unchanged.
    import importlib.util
    script=Path(__file__).resolve().parent/'17_evaluate_generalization_v2_FIXED.py'
    if not script.exists(): p.error('Place 17_evaluate_generalization_v2_FIXED.py beside this script.')
    if a.test_params:
        if not a.test_params.is_file(): p.error('Test parameter CSV missing.')
        tr={tuple(sorted(r.items())) for r in ref.load_csv(ref.PARAM_CSV)}
        val={tuple(sorted(r.items())) for r in ref.load_csv(ref.ID_PARAM_CSV)}
        test=ref.load_csv(a.test_params)
        if len(test)<len(imgs): p.error('Provide at least as many independent parameter rows as images.')
        if any(tuple(sorted(r.items())) in tr|val for r in test): p.error('Test parameters duplicate existing training/validation parameter rows.')
        ref.ID_PARAM_CSV=a.test_params
    else:
        print('WARNING: using existing id_test_parameters.csv; image-independent only, NOT parameter-independent.',flush=True)
    spec=importlib.util.spec_from_file_location('eval4c',script)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    sys.argv=[str(script),'--images',str(a.images),'--out',str(a.out),'--max-images',str(len(imgs)), '--aapg',str(a.aapg),'--image-only',str(a.image_only)]
    mod.main()
    with (a.out/'per_image_metrics.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    rng=np.random.default_rng(2026)
    result=[]
    for scene in sorted({r['scenario'] for r in rows}):
        vals=np.array([float(r['delta_psnr']) for r in rows if r['scenario']==scene]); n=len(vals)
        idx=rng.integers(0,n,size=(a.bootstrap,n))
        ci=np.quantile(vals[idx].mean(axis=1),[.025,.975])
        result.append({'scenario':scene,'n':n,'mean_delta_psnr':float(vals.mean()),'sd_delta_psnr':float(vals.std(ddof=1)), 'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'fraction_aapg_better':float((vals>0).mean())})
    with (a.out/'paired_statistics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    manifest={'image_count':len(imgs),'image_hashes':{str(f):sha256(f) for f in imgs},'parameter_independent_file':str(a.test_params) if a.test_params else None,'bootstrap_resamples':a.bootstrap,'caution':'Distinct parameter rows alone do not prove independent parameter-generation distributions; verify provenance.'}
    (a.out/'independence_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('\n=== 4D PAIRED STATISTICS ===')
    for r in result: print(r)
if __name__=='__main__':main()
