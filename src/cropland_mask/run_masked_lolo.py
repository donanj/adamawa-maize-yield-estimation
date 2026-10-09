"""Matched cropland LOLO sensitivity; reuses the accepted nested training protocol."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import sys, json, hashlib, argparse, platform, time, zipfile, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata as md
import numpy as np
import pandas as pd
import joblib
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src/stacking'))
import run_stacking as s
MODES=['cropland12','cropland12_14']
ROOT=PROJECT/'src/cropland_mask/outputs'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return pd.read_csv(p).sort_values(['lga_name','crop_year']).reset_index(drop=True)
def equal(a,b,cols,tol=0):
    assert len(a)==len(b)
    for c in cols:
        if pd.api.types.is_numeric_dtype(a[c]):
            assert np.allclose(a[c],b[c],equal_nan=True,atol=tol,rtol=0),c
        else: assert a[c].equals(b[c]),c

def worker(spec):
    original=s.tune_fit
    def progress(d,rows,model,tier,budget,folder,stage):
        print(f'{folder.name}: {stage} {model} tuning/loading',flush=True)
        value=original(d,rows,model,tier,budget,folder,stage)
        print(f'{folder.name}: {stage} {model} saved',flush=True)
        return value
    s.tune_fit=progress
    return s.task(spec)

def collect(specs,mode):
    rows=[]
    for spec in specs:
        source,out,panel,depth,tier,held,budget=spec
        d=load(Path(source)/f'{panel}_full_candidates_{depth}.csv')
        folder=Path(out)/f'{panel}_{depth}_T{tier}_{held.replace(" ","_")}'
        r=json.loads((folder/'result.json').read_text()); stack=joblib.load(folder/'stack.joblib')
        tr=np.flatnonzero(d.lga_name.to_numpy()!=held);te=np.flatnonzero(d.lga_name.to_numpy()==held)
        assert r['train_rows']==tr.tolist() and r['test_rows']==te.tolist()
        assert stack['base_order']==s.BASE
        predictions={}
        for name,bundle in zip(s.BASE,stack['base_models']):
            s.check_bundle(bundle,tr,d.lga_name.to_numpy(),budget)
            predictions[name]=bundle['model'].predict(s.tx(bundle['preprocessing'],d.iloc[te]))
        predictions['Stacking']=stack['meta'].predict(np.column_stack([predictions[n] for n in s.BASE]))
        assert np.allclose(predictions['Stacking'],[p['predicted'] for p in r['predictions']])
        predictions['mean']=np.repeat(d.iloc[tr].yield_kg_ha.mean(),len(te))
        for name,pred in predictions.items():
            for i,v in zip(te,pred):
                assert np.isfinite(v)
                rows.append(dict(mask_mode=mode,tier=tier,model=name,lga_name=held,crop_year=int(d.iloc[i].crop_year),observed=float(d.iloc[i].yield_kg_ha),predicted=float(v)))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','verify']);args=p.parse_args()
    smoke=args.stage=='smoke';budget=2 if smoke else 20;tiers=[1] if smoke else [1,2,3]
    inputs=ROOT/'feature_rebuild/model_inputs';accepted=load(PROJECT/'src/tree_models/inputs/main_min5_full_candidates_0_20.csv')
    assert len(accepted)==58 and accepted.lga_name.nunique()==17
    unmasked=load(inputs/'unmasked_main_min5_full_candidates_0_20.csv')
    equal(accepted,unmasked,list(accepted.columns),1e-8)
    held=sorted(accepted.lga_name.unique());held=held[:1] if smoke else held
    out=ROOT/'lolo'/('smoke' if smoke else 'research');out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),PROJECT/'src/stacking/run_stacking.py',PROJECT/'src/tree_models/run_models.py',PROJECT/'src/tree_models/preprocessing.py']+list(inputs.glob('*.csv'))
    manifest=dict(seed=s.SEED,trials=budget,tiers=tiers,held_lgas=held,alpha=1.0,python=platform.python_version(),packages={n:md.version(n) for n in ['numpy','pandas','scipy','scikit-learn','optuna','xgboost','catboost','joblib']},hashes={str(f.relative_to(PROJECT)):digest(f) for f in files})
    stamp=out/'environment.json'
    if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Inputs/code/environment changed: preserve this run and use a separate project copy.')
    if not stamp.exists():s.atomic(stamp,manifest)
    specs_by_mode={}
    for mode in MODES:
        masked=load(inputs/f'{mode}_main_min5_full_candidates_0_20.csv')
        equal(unmasked,masked,[c for c in unmasked.columns if c not in [f'V{i:02}' for i in range(1,13)]])
        source=out/mode/'inputs';source.mkdir(parents=True,exist_ok=True)
        target=source/'main_min5_full_candidates_0_20.csv'
        content=(inputs/f'{mode}_main_min5_full_candidates_0_20.csv').read_bytes()
        if target.exists():assert target.read_bytes()==content
        else:target.write_bytes(content)
        fits=out/mode/'fits';fits.mkdir(exist_ok=True)
        specs_by_mode[mode]=[(str(source),str(fits),'main_min5','0_20',t,h,budget) for t in tiers for h in held]
    specs=[v for values in specs_by_mode.values() for v in values]
    if args.stage!='verify':
        workers=int(os.environ.get('ADAMAWA_WORKERS','2'));assert workers>=1
        print(f'{len(specs)} outer tasks; 18 studies per task; {budget} attempted trials per study. Saved fits resume.',flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            futures=[pool.submit(worker,v) for v in specs]
            for i,f in enumerate(as_completed(futures),1):print(f'{i}/{len(specs)} {f.result()}',flush=True)
    parts=[]
    for mode,values in specs_by_mode.items():
        s.verify(values) # reconstructs nested OOF inputs, tuning isolation and stacking predictions
        parts.append(collect(values,mode))
    if not smoke:
        primary=PROJECT/'src/stacking/outputs/research/reference_99/main_min5/0_20'
        env=json.loads((primary/'environment.json').read_text())
        assert env['seed']==s.SEED and env['trials']==20 and env['alpha']==1.0
        assert env['input_hash']==digest(PROJECT/'src/tree_models/inputs/main_min5_full_candidates_0_20.csv')
        for name,version in env['packages'].items():
            canonical='scikit-learn' if name=='sklearn' else name
            assert md.version(canonical)==version, f'Original training package mismatch: {name}'
        for relative,expected in env['sources'].items():assert digest(PROJECT/relative)==expected,relative
        ref=[(str(PROJECT/'src/tree_models/inputs'),str(primary),'main_min5','0_20',t,h,20) for t in tiers for h in held]
        parts.append(collect(ref,'unmasked'))
    pred=pd.concat(parts,ignore_index=True)
    assert not pred.duplicated(['mask_mode','tier','model','lga_name','crop_year']).any()
    scores=[];folds=[]
    for key,d in pred.groupby(['mask_mode','tier','model']):
        if not smoke:assert len(d)==58 and d.lga_name.nunique()==17
        k=dict(zip(['mask_mode','tier','model'],key));scores.append(k|s.metrics(d.observed.to_numpy(),d.predicted.to_numpy()))
        for h,group in d.groupby('lga_name'):
            folds.append(k|dict(lga_name=h,n=len(group),RMSE=float(np.sqrt(np.mean((group.observed-group.predicted)**2)))))
    metrics=pd.DataFrame(scores);pred.to_csv(out/'matched_oof_predictions.csv',index=False)
    metrics.to_csv(out/'matched_pooled_metrics.csv',index=False);pd.DataFrame(folds).to_csv(out/'matched_per_LGA_metrics.csv',index=False)
    if not smoke:
        reference=metrics[metrics.mask_mode.eq('unmasked')].drop(columns='mask_mode')
        delta=metrics[~metrics.mask_mode.eq('unmasked')].merge(reference,on=['tier','model'],suffixes=('','_unmasked'))
        for metric in ['RMSE','MAE','R2','RRMSE','MAPE']:delta['delta_'+metric]=delta[metric]-delta[metric+'_unmasked']
        delta.to_csv(out/'matched_metric_differences.csv',index=False)
    (out/'Review_Report.md').write_text('# Cropland LOLO sensitivity\n\n'+('SMOKE ONLY: protocol check, not thesis results.' if smoke else 'PASS: matched 58 observations / 17 LGAs; fresh masked nested fitting; reconstructed saved predictions. Unmasked base predictions come from the original stacking final-stage fits. Delta = masked minus unmasked. Mean baseline repeated across tiers as a control. No new significance tests, uncertainty intervals or causal claims. Primary analysis remains unchanged.')+'\n',encoding='utf-8')
    archive=out/'Cropland_LOLO_Review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.glob('*'):
            if f.suffix in ['.csv','.json','.md']:z.write(f,f.name)
        for mode,values in specs_by_mode.items():
            for spec in values:
                folder=Path(spec[1])/f'main_min5_0_20_T{spec[4]}_{spec[5].replace(" ","_")}'
                z.write(folder/'result.json',str((folder/'result.json').relative_to(out)))
    print(metrics.to_string(index=False));print('PASS: review archive saved. '+('SMOKE ONLY.' if smoke else 'Research sensitivity complete.'),flush=True)
if __name__=='__main__':main()
