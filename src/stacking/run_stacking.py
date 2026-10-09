"""Fully nested, geographically grouped stacking. Execute from project root."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib,argparse,time,multiprocessing,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import joblib,optuna,sklearn,xgboost,catboost
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src/tree_models'))
from run_models import params,estimator,SEED
from preprocessing import Prep,metrics
BASE=['RF','XGBoost','CatBoost']
optuna.logging.set_verbosity(optuna.logging.WARNING)
def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def tx(prep,d): return prep.transform(d)*prep.scale+prep.center
def tune_fit(d,rows,model,tier,budget,folder,stage):
 """rows excludes both outer holdout and the current stacking OOF group."""
 path=folder/f'{stage}_{model}.joblib'
 if path.exists(): return joblib.load(path)
 g=d.lga_name.to_numpy();y=d.yield_kg_ha.to_numpy();folds=[];audit=[]
 for a,b in GroupKFold(5).split(rows,groups=g[rows]):
  tr=rows[a];va=rows[b];assert set(g[tr]).isdisjoint(g[va])
  prep=Prep().fit(d.iloc[tr],tier)
  folds.append((tx(prep,d.iloc[tr]),y[tr],tx(prep,d.iloc[va]),y[va]))
  audit.append(dict(train_rows=tr.tolist(),validation_rows=va.tolist(),features=prep.cols))
 study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=10),pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=1))
 def objective(trial):
  p=params(trial,model);scores=[]
  for j,(x,yt,xv,yv) in enumerate(folds):
   pred=estimator(model,p).fit(x,yt).predict(xv)
   scores.append(float(np.sqrt(np.mean((yv-pred)**2))))
   trial.report(float(np.mean(scores)),j)
   if trial.should_prune():raise optuna.TrialPruned()
  return float(np.mean(scores))
 study.optimize(objective,n_trials=budget)
 prep=Prep().fit(d.iloc[rows],tier);est=estimator(model,study.best_params).fit(tx(prep,d.iloc[rows]),y[rows])
 bundle=dict(preprocessing=prep,model=est,train_rows=rows.tolist(),tuning_splits=audit,trials=[dict(number=t.number,state=t.state.name,value=t.value,params=t.params) for t in study.trials],best_params=study.best_params)
 tmp=path.with_suffix('.tmp');joblib.dump(bundle,tmp,compress=3);tmp.replace(path)
 joblib.dump(study,folder/f'{stage}_{model}_study.joblib',compress=3)
 return bundle

def task(spec):
 source,out,panel,depth,tier,held,budget=spec
 folder=Path(out)/f'{panel}_{depth}_T{tier}_{held.replace(" ","_")}';folder.mkdir(parents=True,exist_ok=True)
 result=folder/'result.json'
 if result.exists():return result.name+' cached '+folder.name
 start=time.time();d=pd.read_csv(Path(source)/f'{panel}_full_candidates_{depth}.csv').sort_values(['lga_name','crop_year']).reset_index(drop=True)
 g=d.lga_name.to_numpy();y=d.yield_kg_ha.to_numpy();tr=np.flatnonzero(g!=held);te=np.flatnonzero(g==held)
 z=np.full((len(tr),3),np.nan);coverage=np.zeros(len(tr),int);audits=[]
 for k,(a,b) in enumerate(GroupKFold(5).split(tr,groups=g[tr])):
  fit=tr[a];val=tr[b];assert set(g[fit]).isdisjoint(g[val])
  audits.append(dict(stage=f'oof{k}',train_rows=fit.tolist(),validation_rows=val.tolist()))
  for j,name in enumerate(BASE):
   bundle=tune_fit(d,fit,name,tier,budget,folder,f'oof{k}')
   z[b,j]=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[val]))
  coverage[b]+=1
 assert np.all(coverage==1) and np.isfinite(z).all()
 meta=Ridge(alpha=1.0).fit(z,y[tr]);test=np.empty((len(te),3));final=[]
 for j,name in enumerate(BASE):
  bundle=tune_fit(d,tr,name,tier,budget,folder,'final');final.append(bundle)
  test[:,j]=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[te]))
 pred=meta.predict(test);assert np.isfinite(pred).all()
 joblib.dump(dict(meta=meta,base_models=final,base_order=BASE,meta_training_predictions=z,meta_training_rows=tr,meta_training_targets=y[tr]),folder/'stack.joblib',compress=3)
 atomic(result,dict(panel=panel,depth=depth,tier=tier,held_lga=held,train_rows=tr.tolist(),test_rows=te.tolist(),stacking_splits=audits,meta_alpha=1.0,seconds=time.time()-start,predictions=[dict(lga_name=held,crop_year=int(d.iloc[i].crop_year),observed=float(y[i]),predicted=float(p)) for i,p in zip(te,pred)]))
 return f'{folder.name}: {time.time()-start:.1f}s'

def verify(specs):
 all_predictions=[]
 for spec in specs:
  source,out,panel,depth,tier,held,budget=spec
  folder=Path(out)/f'{panel}_{depth}_T{tier}_{held.replace(" ","_")}'
  r=json.loads((folder/'result.json').read_text());d=pd.read_csv(Path(source)/f'{panel}_full_candidates_{depth}.csv').sort_values(['lga_name','crop_year']).reset_index(drop=True)
  g=d.lga_name.to_numpy();tr=np.array(r['train_rows']);te=np.array(r['test_rows']);assert np.array_equal(tr,np.flatnonzero(g!=held)) and np.array_equal(te,np.flatnonzero(g==held))
  stack=joblib.load(folder/'stack.joblib');z=np.full((len(tr),3),np.nan);coverage=np.zeros(len(d),int)
  for split in r['stacking_splits']:
   a=np.array(split['train_rows']);b=np.array(split['validation_rows']);assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b]);coverage[b]+=1
   for j,name in enumerate(BASE):
    bundle=joblib.load(folder/f"{split['stage']}_{name}.joblib");check_bundle(bundle,a,g,budget)
    z[np.searchsorted(tr,b),j]=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[b]))
  assert np.all(coverage[tr]==1) and np.all(coverage[te]==0) and np.allclose(z,stack['meta_training_predictions'])
  assert np.array_equal(stack['meta_training_rows'],tr) and np.allclose(stack['meta_training_targets'],d.iloc[tr].yield_kg_ha)
  meta=Ridge(alpha=1.0).fit(z,d.iloc[tr].yield_kg_ha);assert np.allclose(meta.coef_,stack['meta'].coef_) and np.allclose(meta.intercept_,stack['meta'].intercept_)
  test=[]
  for bundle in stack['base_models']:
   check_bundle(bundle,tr,g,budget);test.append(bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[te])))
  predicted=stack['meta'].predict(np.column_stack(test));assert np.allclose(predicted,[p['predicted'] for p in r['predictions']])
  for i,p in zip(te,r['predictions']):
   assert p['crop_year']==int(d.iloc[i].crop_year) and p['observed']==float(d.iloc[i].yield_kg_ha)
   all_predictions.append(dict(panel=panel,depth=depth,tier=tier,model='Stacking',**p))
 pr=pd.DataFrame(all_predictions);assert not pr.duplicated(['panel','depth','tier','lga_name','crop_year']).any()
 output=Path(specs[0][1]);pr.to_csv(output/'oof_predictions.csv',index=False)
 scores=[];folds=[]
 for key,part in pr.groupby(['panel','depth','tier','model']):
  k=dict(zip(['panel','depth','tier','model'],key));scores.append(k|metrics(part.observed.to_numpy(),part.predicted.to_numpy()))
  for lga,f in part.groupby('lga_name'):folds.append(k|dict(lga_name=lga,n=len(f),RMSE=float(np.sqrt(np.mean((f.observed-f.predicted)**2)))))
 pd.DataFrame(scores).to_csv(output/'pooled_metrics.csv',index=False);pd.DataFrame(folds).to_csv(output/'per_LGA_metrics.csv',index=False)
 print('PASS: outer coverage, three-level group isolation, 20-trial budgets (2 for smoke), unchanged targets, reconstructed OOF meta inputs and saved-model predictions.')
 print(pd.DataFrame(scores).to_string(index=False))
def check_bundle(bundle,expected,g,budget):
 assert bundle['train_rows']==expected.tolist() and len(bundle['trials'])==budget
 coverage=np.zeros(len(g),int)
 for split in bundle['tuning_splits']:
  a=split['train_rows'];b=split['validation_rows'];assert set(a)|set(b)==set(expected) and set(g[a]).isdisjoint(g[b]);coverage[b]+=1
 assert np.all(coverage[expected]==1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','verify']);p.add_argument('--rule',choices=['reference_99','winsor_95','exclude_gt10000'],default='reference_99');p.add_argument('--panel',choices=['main_min5','all_cells'],default='main_min5');p.add_argument('--tier',choices=['1','2','3','all'],default='all');p.add_argument('--depth',choices=['0_20','20_50'],default='0_20');args=p.parse_args()
 source=PROJECT/'src/tree_models/inputs' if args.rule=='reference_99' else PROJECT/f'src/target_quality/outputs/runs/{args.rule}/tree_models/inputs'
 tiers=[1,2,3] if args.tier=='all' else [int(args.tier)]
 if args.depth=='20_50' and tiers!=[3]:p.error('Depth sensitivity uses --tier 3 only.')
 smoke=args.stage=='smoke';budget=2 if smoke else 20
 if smoke:tiers=[1]
 out=PROJECT/'src/stacking/outputs'/('smoke' if smoke else 'research')/args.rule/args.panel/args.depth
 out.mkdir(parents=True,exist_ok=True)
 manifest=dict(seed=SEED,trials=budget,alpha=1.0,python=platform.python_version(),packages={n:m.__version__ for n,m in [('numpy',np),('pandas',pd),('sklearn',sklearn),('optuna',optuna),('xgboost',xgboost),('catboost',catboost),('joblib',joblib)]},input_hash=hashlib.sha256((source/f'{args.panel}_full_candidates_{args.depth}.csv').read_bytes()).hexdigest(),sources={str(f.relative_to(PROJECT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),PROJECT/'src/tree_models/run_models.py',PROJECT/'src/tree_models/preprocessing.py']})
 stamp=out/'environment.json'
 if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Inputs/code/environment changed. Preserve this run; use a separate project copy.')
 if not stamp.exists():atomic(stamp,manifest)
 d=pd.read_csv(source/f'{args.panel}_full_candidates_{args.depth}.csv');held=sorted(d.lga_name.unique());held=held[:1] if smoke else held
 specs=[(str(source),str(out),args.panel,args.depth,t,h,budget) for t in tiers for h in held]
 if args.stage!='verify':
  workers=int(os.environ.get('ADAMAWA_WORKERS','2'));assert workers>=1
  print(f'{len(specs)} outer tasks; 18 tuning studies per task; {budget} attempted trials per study. Completed base-stage fits resume.',flush=True)
  with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
   futures=[pool.submit(task,s) for s in specs]
   for i,f in enumerate(as_completed(futures),1):print(f'{i}/{len(specs)} {f.result()}',flush=True)
 verify(specs)
 if smoke:print('SMOKE ONLY: installation/protocol check; not a thesis performance result.')
