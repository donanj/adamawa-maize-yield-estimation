"""Fully nested leave-one-year-out temporal sensitivity. Execute from project root."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib,argparse,time,multiprocessing,platform,zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import joblib,optuna,sklearn,xgboost,catboost
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import Ridge,LinearRegression
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
 """rows excludes both the outer year and the current stacking OOF year."""
 path=folder/f'{stage}_{model}.joblib'
 if path.exists(): return joblib.load(path)
 print(f'{folder.name}: {stage} {model} tuning',flush=True)
 g=d.crop_year.to_numpy();y=d.yield_kg_ha.to_numpy();folds=[];audit=[]
 for a,b in LeaveOneGroupOut().split(rows,groups=g[rows]):
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
 print(f'{folder.name}: {stage} {model} saved',flush=True)
 joblib.dump(study,folder/f'{stage}_{model}_study.joblib',compress=3)
 return bundle

def year_splits(d,rows):
 g=d.crop_year.to_numpy()
 for a,b in LeaveOneGroupOut().split(rows,groups=g[rows]):yield rows[a],rows[b]
def linear_fit(d,tr,tier,model):
 y=d.yield_kg_ha.to_numpy();alpha=None;tuning=[]
 if model=='Ridge':
  alphas=[.01,.1,1.,10.,100.,1000.];scores={a:[] for a in alphas}
  for a,b in year_splits(d,tr):
   prep=Prep().fit(d.iloc[a],tier)
   for value in alphas:
    est=Ridge(alpha=value).fit(prep.transform(d.iloc[a]),y[a]);p=est.predict(prep.transform(d.iloc[b]));scores[value].append(float(np.sqrt(np.mean((y[b]-p)**2))))
   tuning.append(dict(train_rows=a.tolist(),validation_rows=b.tolist()))
  alpha=min(alphas,key=lambda a:np.mean(scores[a]))
 prep=Prep().fit(d.iloc[tr],tier,vif=model=='MLR');x=prep.transform(d.iloc[tr])
 if model=='MLR':assert np.linalg.matrix_rank(np.column_stack([np.ones(len(tr)),x]))==x.shape[1]+1
 est=LinearRegression() if model=='MLR' else Ridge(alpha=alpha)
 est.fit(x,y[tr]);return dict(preprocessing=prep,model=est,train_rows=tr.tolist(),alpha=alpha,tuning_splits=tuning,alpha_scores=scores if model=='Ridge' else None)
def task(spec):
 source,out,panel,depth,tier,held,budget=spec
 folder=Path(out)/f'T{tier}_year_{held}';folder.mkdir(parents=True,exist_ok=True)
 result=folder/'result.json'
 if result.exists():return folder.name+' cached'
 start=time.time();d=pd.read_csv(source).sort_values(['lga_name','crop_year']).reset_index(drop=True)
 g=d.crop_year.to_numpy();y=d.yield_kg_ha.to_numpy();tr=np.flatnonzero(g!=held);te=np.flatnonzero(g==held)
 z=np.full((len(tr),3),np.nan);coverage=np.zeros(len(tr),int);audits=[]
 for k,(fit,val) in enumerate(year_splits(d,tr)):
  assert set(g[fit]).isdisjoint(g[val]);audits.append(dict(stage=f'oof{k}',train_rows=fit.tolist(),validation_rows=val.tolist()))
  for j,name in enumerate(BASE):
   bundle=tune_fit(d,fit,name,tier,budget,folder,f'oof{k}');z[np.searchsorted(tr,val),j]=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[val]))
  coverage[np.searchsorted(tr,val)]+=1
 assert len(audits)==4 and np.all(coverage==1) and np.isfinite(z).all()
 meta=Ridge(alpha=1.0).fit(z,y[tr]);test=np.empty((len(te),3));final=[];predictions={}
 for j,name in enumerate(BASE):
  bundle=tune_fit(d,tr,name,tier,budget,folder,'final');final.append(bundle)
  test[:,j]=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[te]));predictions[name]=test[:,j]
 predictions['Stacking']=meta.predict(test)
 joblib.dump(dict(meta=meta,base_models=final,base_order=BASE,meta_training_predictions=z,meta_training_rows=tr,meta_training_targets=y[tr]),folder/'stack.joblib',compress=3)
 for model in ['MLR']+(['Ridge'] if tier==3 else []):
  bundle=linear_fit(d,tr,tier,model);joblib.dump(bundle,folder/f'{model}.joblib',compress=3);predictions[model]=bundle['model'].predict(bundle['preprocessing'].transform(d.iloc[te]))
 if tier==1:predictions['mean']=np.repeat(y[tr].mean(),len(te))
 rows=[]
 for model,pred in predictions.items():
  assert np.isfinite(pred).all()
  for i,p in zip(te,pred):rows.append(dict(panel=panel,depth=depth,tier=0 if model=='mean' else tier,model=model,held_year=int(held),lga_name=str(d.iloc[i].lga_name),crop_year=int(d.iloc[i].crop_year),observed=float(y[i]),predicted=float(p)))
 atomic(result,dict(tier=tier,held_year=int(held),train_rows=tr.tolist(),test_rows=te.tolist(),stacking_splits=audits,meta_alpha=1.,seconds=time.time()-start,predictions=rows))
 return f'{folder.name}: {time.time()-start:.1f}s'
def check_splits(splits,expected,g):
 coverage=np.zeros(len(g),int)
 for split in splits:
  a=np.array(split['train_rows']);b=np.array(split['validation_rows'])
  assert set(a)|set(b)==set(expected) and set(a).isdisjoint(b) and set(g[a]).isdisjoint(g[b])
  assert len(set(g[b]))==1;coverage[b]+=1
 assert len(splits)==len(set(g[expected])) and np.all(coverage[expected]==1)
def check_bundle(bundle,expected,g,budget):
 assert bundle['train_rows']==expected.tolist() and len(bundle['trials'])==budget
 check_splits(bundle['tuning_splits'],expected,g)
def verify(specs,smoke=False):
 records=[]
 for source,out,panel,depth,tier,held,budget in specs:
  folder=Path(out)/f'T{tier}_year_{held}';r=json.loads((folder/'result.json').read_text());d=pd.read_csv(source).sort_values(['lga_name','crop_year']).reset_index(drop=True)
  g=d.crop_year.to_numpy();y=d.yield_kg_ha.to_numpy();tr=np.array(r['train_rows']);te=np.array(r['test_rows']);assert np.array_equal(tr,np.flatnonzero(g!=held)) and np.array_equal(te,np.flatnonzero(g==held))
  check_splits(r['stacking_splits'],tr,g);stack=joblib.load(folder/'stack.joblib');z=np.full((len(tr),3),np.nan)
  for split in r['stacking_splits']:
   a=np.array(split['train_rows']);b=np.array(split['validation_rows'])
   for j,name in enumerate(BASE):
    bundle=joblib.load(folder/f"{split['stage']}_{name}.joblib");check_bundle(bundle,a,g,budget);assert len(bundle['tuning_splits'])==3
    z[np.searchsorted(tr,b),j]=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[b]))
  assert np.isfinite(z).all() and np.allclose(z,stack['meta_training_predictions']) and np.array_equal(stack['meta_training_rows'],tr) and np.allclose(stack['meta_training_targets'],y[tr])
  refit=Ridge(alpha=1.).fit(z,y[tr]);assert np.allclose(refit.coef_,stack['meta'].coef_) and np.allclose(refit.intercept_,stack['meta'].intercept_)
  test=[];reconstructed={}
  for name,bundle in zip(BASE,stack['base_models']):
   check_bundle(bundle,tr,g,budget);assert len(bundle['tuning_splits'])==4
   pred=bundle['model'].predict(tx(bundle['preprocessing'],d.iloc[te]));test.append(pred);reconstructed[name]=pred
  reconstructed['Stacking']=stack['meta'].predict(np.column_stack(test))
  for name in ['MLR']+(['Ridge'] if tier==3 else []):
   bundle=joblib.load(folder/f'{name}.joblib');assert bundle['train_rows']==tr.tolist()
   # Independently rebuild predictor preprocessing using only the outer training rows.
   rebuilt=Prep().fit(d.iloc[tr],tier,vif=name=='MLR');prep=bundle['preprocessing']
   assert prep.cols==rebuilt.cols and prep.indices==rebuilt.indices
   for attribute in ['lo','hi','mean','center','scale']:assert np.allclose(getattr(prep,attribute),getattr(rebuilt,attribute))
   if name=='Ridge':
    check_splits(bundle['tuning_splits'],tr,g);assert bundle['alpha']==min(bundle['alpha_scores'],key=lambda a:np.mean(bundle['alpha_scores'][a]))
   reconstructed[name]=bundle['model'].predict(prep.transform(d.iloc[te]))
  if tier==1:reconstructed['mean']=np.repeat(y[tr].mean(),len(te))
  predictions=pd.DataFrame(r['predictions']);assert set(predictions.model)==set(reconstructed)
  for name,pred in reconstructed.items():
   part=predictions[predictions.model==name].reset_index(drop=True)
   assert len(part)==len(te) and np.array_equal(part.crop_year,g[te]) and np.array_equal(part.lga_name,d.iloc[te].lga_name.to_numpy())
   assert np.allclose(part.observed,y[te]) and np.allclose(part.predicted,pred)
  records.extend(r['predictions'])
 pr=pd.DataFrame(records);assert not pr.duplicated(['model','tier','lga_name','crop_year']).any()
 if not smoke:
  assert len(pr)==58*17 and pr.groupby(['model','tier']).size().eq(58).all() and pr.groupby(['model','tier']).held_year.nunique().eq(5).all()
 out=Path(specs[0][1]);pr.to_csv(out/'oof_predictions.csv',index=False);pooled=[];fold=[]
 for key,part in pr.groupby(['model','tier']):
  k=dict(model=key[0],tier=int(key[1]));pooled.append(k|metrics(part.observed.to_numpy(),part.predicted.to_numpy()))
  for year,p in part.groupby('held_year'):fold.append(k|dict(held_year=int(year),n=len(p),**metrics(p.observed.to_numpy(),p.predicted.to_numpy())))
 pd.DataFrame(pooled).sort_values('RMSE').to_csv(out/'pooled_metrics.csv',index=False);pd.DataFrame(fold).to_csv(out/'per_year_metrics.csv',index=False)
 compared=[]
 if not smoke:
  paths=[PROJECT/'src/linear_benchmarks/outputs/oof_predictions.csv',PROJECT/'src/tree_models/outputs/oof_predictions.csv',PROJECT/'src/stacking/outputs/research/reference_99/main_min5/0_20/oof_predictions.csv']
  if all(p.exists() for p in paths):
   lolo=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);lolo=lolo[(lolo.panel=='main_min5')&(lolo.depth=='0_20')]
   for row in pooled:
    part=lolo[(lolo.model==row['model'])&(lolo.tier==row['tier'])].sort_values(['lga_name','crop_year']).reset_index(drop=True)
    assert len(part)==58 and part[['lga_name','crop_year']].equals(d[['lga_name','crop_year']]) and np.allclose(part.observed,y)
    score=metrics(part.observed.to_numpy(),part.predicted.to_numpy());compared.append(dict(model=row['model'],tier=row['tier'],LOLO_RMSE=score['RMSE'],LOYO_RMSE=row['RMSE'],LOYO_minus_LOLO_RMSE=row['RMSE']-score['RMSE'],LOLO_R2=score['R2'],LOYO_R2=row['R2']))
   pd.DataFrame(compared).to_csv(out/'LOLO_vs_LOYO.csv',index=False)
 report=['# LOYO temporal sensitivity','', 'Primary q99 main panel: 58 LGA-year cells; 17 LGAs; 0–20 cm; five crop years.','', 'All years of an outer held-out crop season are excluded from model training, tuning, preprocessing fitting and meta-learner training. Inner stacking uses four leave-one-year-out folds; tuning within each OOF training subset uses three leave-one-year-out folds. Final base tuning and standalone Ridge tuning use four leave-one-year-out folds.','', 'LOYO assesses transfer across the observed years. Earlier-year folds use later years for training; this is not rolling-origin forecasting or proof of future-season generalisation. The same LGAs can occur in training and test years, so it is not joint spatial-temporal holdout. Labels remain the established wave-wide cleaned targets; no new target policy was introduced.','', 'Five held-out years provide too little information for strong year-level statistical inference. This output supplies pooled and per-year descriptive results, with no additional p-values or year-bootstrap confidence intervals. Difference from LOLO mixes a different generalisation target, training composition and inner selection scheme; it is not a direct numerical estimate of temporal dependence alone.']
 if smoke:report.append('\nSMOKE ONLY: one outer year, Tier 1, two trials. Not thesis results.')
 (out/'Review_Report.md').write_text('\n'.join(report),encoding='utf-8')
 package=out/('LOYO_Smoke_Review.zip' if smoke else 'LOYO_Main_Panel_Review.zip')
 with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as zf:
  for path in out.iterdir():
   if path.is_file() and path.suffix in ['.csv','.md','.json']:zf.write(path,path.name)
 print('PASS: outer-year exclusion, four-year stacking OOF, three/four-year tuning isolation, trial counts, unchanged targets and saved-model prediction reconstruction.',flush=True)
 print(pd.DataFrame(pooled).sort_values('RMSE').to_string(index=False),flush=True)
 if smoke:print('SMOKE ONLY: installation/protocol check; not thesis results.',flush=True)
 else:print(f'Share this review ZIP: {package}',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','verify']);args=p.parse_args();smoke=args.stage=='smoke';budget=2 if smoke else 20
 source=PROJECT/'src/tree_models/inputs/main_min5_full_candidates_0_20.csv';d=pd.read_csv(source)
 assert len(d)==58 and d.lga_name.nunique()==17 and set(d.crop_year)=={2010,2012,2015,2018,2023} and not d.duplicated(['lga_name','crop_year']).any()
 out=PROJECT/'src/temporal_validation/outputs'/('smoke' if smoke else 'research');out.mkdir(parents=True,exist_ok=True)
 manifest=dict(seed=SEED,trials=budget,meta_alpha=1.,outer='Leave-one-crop-year-out',stacking='Four inner year holdouts',tuning='Three year holdouts within OOF subsets; four for final refits',python=platform.python_version(),packages={n:m.__version__ for n,m in [('numpy',np),('pandas',pd),('sklearn',sklearn),('optuna',optuna),('xgboost',xgboost),('catboost',catboost),('joblib',joblib)]},input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sources={str(f.relative_to(PROJECT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),PROJECT/'src/tree_models/run_models.py',PROJECT/'src/tree_models/preprocessing.py']})
 stamp=out/'environment.json'
 if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Inputs/code/software changed. Preserve previous runs and use a separate project copy.')
 if not stamp.exists():atomic(stamp,manifest)
 years=[2010] if smoke else sorted(d.crop_year.unique());tiers=[1] if smoke else [1,2,3]
 specs=[(str(source),str(out),'main_min5','0_20',tier,int(year),budget) for tier in tiers for year in years]
 if args.stage!='verify':
  workers=int(os.environ.get('ADAMAWA_WORKERS','2'));assert workers>=1
  print(f'{len(specs)} outer year/tier tasks; 15 tree tuning studies per task; {budget} attempted trials per study. Completed stages resume.',flush=True)
  with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
   futures=[pool.submit(task,s) for s in specs]
   for i,f in enumerate(as_completed(futures),1):print(f'{i}/{len(specs)} {f.result()}',flush=True)
 verify(specs,smoke=smoke)
