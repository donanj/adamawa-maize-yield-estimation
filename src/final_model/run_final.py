"""Full-development-panel stacking; fitted predictions are not validation results."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
from pathlib import Path
import sys, argparse, json, time, hashlib, platform, zipfile
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src/stacking'))
from run_stacking import tune_fit, tx, atomic, BASE, SEED, check_bundle
INPUT=PROJECT/'src/tree_models/inputs/main_min5_full_candidates_0_20.csv'
ROOT=PROJECT/'src/final_model/outputs'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(stage):
 d=pd.read_csv(INPUT).sort_values(['lga_name','crop_year']).reset_index(drop=True)
 if len(d)!=58 or d.lga_name.nunique()!=17 or d.duplicated(['lga_name','crop_year']).any():raise ValueError('Expected accepted 58-row, 17-LGA panel.')
 env=json.loads((PROJECT/'src/stacking/outputs/research/reference_99/main_min5/0_20/environment.json').read_text())
 if env['input_hash']!=digest(INPUT):raise ValueError('Input differs from primary LOLO training input.')
 for path,old in env['sources'].items():
  if digest(PROJECT/path)!=old:raise ValueError(f'Training source changed: {path}')
 for name,old in env['packages'].items():
  if version({'sklearn':'scikit-learn'}.get(name,name))!=old:raise ValueError(f'Training package changed: {name}')
 if platform.python_version()!=env['python']:raise ValueError('Training Python version changed.')
 smoke=stage=='smoke';out=ROOT/('smoke' if smoke else 'research');out.mkdir(parents=True,exist_ok=True)
 manifest=dict(input_hash=digest(INPUT),training_environment=env,seed=SEED,attempted_trials=2 if smoke else 20,meta_alpha=1.0,script_hash=digest(Path(__file__)),purpose='full-panel development model; no held-out performance claim')
 stamp=out/'environment.json'
 if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Run identity changed. Preserve outputs; use a separate project copy.')
 if not stamp.exists():atomic(stamp,manifest)
 return d,out,manifest

def verify_tier(d,folder,tier,budget):
 r=json.loads((folder/'result.json').read_text());stack=joblib.load(folder/'stack.joblib')
 rows=np.arange(len(d));groups=d.lga_name.to_numpy();y=d.yield_kg_ha.to_numpy()
 assert r['train_rows']==rows.tolist() and r['explanation_rows']==rows.tolist() and stack['base_order']==BASE
 assert np.array_equal(stack['meta_training_rows'],rows) and np.array_equal(stack['meta_training_targets'],y)
 z=np.full((len(d),3),np.nan);coverage=np.zeros(len(d),int)
 for split in r['stacking_splits']:
  a=np.array(split['train_rows']);b=np.array(split['validation_rows']);assert set(a)|set(b)==set(rows) and set(a).isdisjoint(b) and set(groups[a]).isdisjoint(groups[b]);coverage[b]+=1
  for j,name in enumerate(BASE):
   bundle=joblib.load(folder/f"{split['stage']}_{name}.joblib");check_bundle(bundle,a,groups,budget)
   z[b,j]=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[b]))
 assert np.all(coverage==1) and np.isfinite(z).all() and np.allclose(z,stack['meta_training_predictions'])
 meta=Ridge(alpha=1.0).fit(z,y);assert np.allclose(meta.coef_,stack['meta'].coef_) and np.allclose(meta.intercept_,stack['meta'].intercept_)
 final=[]
 for bundle in stack['base_models']:
  check_bundle(bundle,rows,groups,budget);final.append(bundle['model'].predict(tx(bundle['preprocessing'],d)))
 pred=stack['meta'].predict(np.column_stack(final))
 assert np.isfinite(pred).all() and np.allclose(pred,[p['predicted'] for p in r['predictions']])
 for i,p in enumerate(r['predictions']):assert p['lga_name']==d.iloc[i].lga_name and p['crop_year']==int(d.iloc[i].crop_year) and p['observed']==float(y[i])
 return pd.DataFrame(r['predictions']).assign(tier=tier,prediction_scope='in_sample_final_model')

def fit_tier(d,folder,tier,budget):
 folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return
 started=time.time();rows=np.arange(len(d));y=d.yield_kg_ha.to_numpy();g=d.lga_name.to_numpy();z=np.full((len(d),3),np.nan);coverage=np.zeros(len(d),int);audits=[]
 for k,(a,b) in enumerate(GroupKFold(5).split(rows,groups=g)):
  audits.append(dict(stage=f'oof{k}',train_rows=a.tolist(),validation_rows=b.tolist()))
  for j,name in enumerate(BASE):
   print(f'T{tier} oof{k} {name}: tuning or loading saved stage',flush=True)
   bundle=tune_fit(d,a,name,tier,budget,folder,f'oof{k}');z[b,j]=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[b]))
   print(f'T{tier} oof{k} {name}: saved',flush=True)
  coverage[b]+=1
 assert np.all(coverage==1) and np.isfinite(z).all()
 meta=Ridge(alpha=1.0).fit(z,y);final=[];x=[]
 for name in BASE:
  print(f'T{tier} final {name}: tuning or loading saved stage',flush=True)
  bundle=tune_fit(d,rows,name,tier,budget,folder,'final');final.append(bundle);x.append(bundle['model'].predict(tx(bundle['preprocessing'],d)))
  print(f'T{tier} final {name}: saved',flush=True)
 pred=meta.predict(np.column_stack(x))
 joblib.dump(dict(meta=meta,base_models=final,base_order=BASE,meta_training_predictions=z,meta_training_rows=rows,meta_training_targets=y),folder/'stack.joblib',compress=3)
 atomic(folder/'result.json',dict(tier=tier,train_rows=rows.tolist(),explanation_rows=rows.tolist(),stacking_splits=audits,meta_alpha=1.0,seconds=time.time()-started,predictions=[dict(lga_name=str(d.iloc[i].lga_name),crop_year=int(d.iloc[i].crop_year),observed=float(y[i]),predicted=float(pred[i])) for i in rows]))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','smoke','run','verify']);args=p.parse_args()
 d,out,manifest=load(args.stage)
 print('PASS: accepted panel and original training code/package/input identity.',flush=True)
 if args.stage=='check':return
 tiers=[1] if args.stage=='smoke' else [1,2,3];budget=manifest['attempted_trials'];frames=[];histories=[];summary=[]
 print(f'{len(tiers)} tiers; 18 studies per tier; {budget} attempted trials per study.',flush=True)
 for tier in tiers:
  folder=out/f'T{tier}'
  if args.stage!='verify':fit_tier(d,folder,tier,budget)
  frames.append(verify_tier(d,folder,tier,budget))
  for path in sorted(folder.glob('*_RF.joblib'))+sorted(folder.glob('*_XGBoost.joblib'))+sorted(folder.glob('*_CatBoost.joblib')):
   b=joblib.load(path);best=None
   for trial in b['trials']:
    if trial['state']=='COMPLETE':best=trial['value'] if best is None else min(best,trial['value'])
    histories.append(dict(tier=tier,study=path.stem,number=trial['number'],state=trial['state'],value=trial['value'],best_complete_so_far=best,parameters=json.dumps(trial['params'],sort_keys=True)))
   summary.append(dict(tier=tier,study=path.stem,train_rows=len(b['train_rows']),features=';'.join(b['preprocessing'].cols),best_parameters=json.dumps(b['best_params'],sort_keys=True)))
  print(f'PASS T{tier}: grouped nested OOF, Ridge reconstruction, unchanged targets and full-panel saved predictions.',flush=True)
 pd.concat(frames).to_csv(out/'fitted_predictions.csv',index=False)
 pd.DataFrame(histories).to_csv(out/'tuning_history.csv',index=False);pd.DataFrame(summary).to_csv(out/'selected_parameters.csv',index=False)
 print('SMOKE ONLY.' if args.stage=='smoke' else 'Final development models verified. Next run full-panel SHAP. Fitted predictions are not generalisation metrics.',flush=True)
if __name__=='__main__':main()
