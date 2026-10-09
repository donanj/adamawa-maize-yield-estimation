import os
for env in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[env]='1'
from pathlib import Path
import json,hashlib,platform,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import optuna,xgboost,catboost,sklearn,joblib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from preprocessing import Prep,metrics
ROOT=Path(__file__).resolve().parent
SEED=20261006;TRIALS=20
WORKERS=int(os.environ.get('ADAMAWA_WORKERS','2'))
assert WORKERS>=1
optuna.logging.set_verbosity(optuna.logging.WARNING)
def params(trial,model):
 if model=='RF':return dict(n_estimators=trial.suggest_int('n_estimators',100,800),max_depth=trial.suggest_categorical('max_depth',[None]+list(range(3,16))),min_samples_leaf=trial.suggest_int('min_samples_leaf',1,8),max_features=trial.suggest_categorical('max_features',['sqrt',.3,.5,.7]))
 if model=='XGBoost':return dict(n_estimators=trial.suggest_int('n_estimators',50,400),learning_rate=trial.suggest_float('learning_rate',.01,.3,log=True),max_depth=trial.suggest_int('max_depth',2,7),subsample=trial.suggest_float('subsample',.5,1),colsample_bytree=trial.suggest_float('colsample_bytree',.5,1),reg_alpha=trial.suggest_float('reg_alpha',0,5),reg_lambda=trial.suggest_float('reg_lambda',0,5))
 return dict(iterations=trial.suggest_int('iterations',100,400),learning_rate=trial.suggest_float('learning_rate',.01,.3,log=True),depth=trial.suggest_int('depth',3,7),l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1,10))
def estimator(model,p):
 if model=='RF':return RandomForestRegressor(**p,random_state=SEED,n_jobs=1)
 if model=='XGBoost':return XGBRegressor(**p,random_state=SEED,n_jobs=1,objective='reg:squarederror',tree_method='hist')
 return CatBoostRegressor(**p,random_seed=SEED,thread_count=1,loss_function='RMSE',boosting_type='Ordered',grow_policy='SymmetricTree',verbose=False,allow_writing_files=False)
def task(spec):
 panel,depth,tier,model,held=spec
 tag=f'{panel}_{depth}_T{tier}_{model}_{held.replace(" ","_")}'
 output=ROOT/'outputs'/f'{tag}.json'
 if output.exists():
  old=json.loads(output.read_text())
  if len(old['trials'])!=TRIALS:raise ValueError('Tuning budget changed: use a separate clean run folder.')
  return tag+' cached'
 start=time.time()
 status=ROOT/'studies'/f'{tag}.status.json'
 status.write_text(json.dumps(dict(task=tag,stage='preparing',pid=os.getpid())))
 df=pd.read_csv(ROOT/'inputs'/f'{panel}_full_candidates_{depth}.csv').sort_values(['lga_name','crop_year']).reset_index(drop=True)
 groups=df.lga_name.to_numpy();y=df.yield_kg_ha.to_numpy()
 tr=np.flatnonzero(groups!=held);te=np.flatnonzero(groups==held)
 assert set(groups[tr]).isdisjoint(groups[te])
 folds=[];split_audit=[]
 for j,(it,iv) in enumerate(GroupKFold(5).split(tr,groups=groups[tr])):
  a=tr[it];b=tr[iv];assert set(groups[a]).isdisjoint(groups[b])
  prep=Prep().fit(df.iloc[a],tier)
  # Tree inputs are winsorised/imputed; standardisation is unnecessary.
  def tree_x(part):return prep.transform(part)*prep.scale+prep.center
  folds.append((tree_x(df.iloc[a]),y[a],tree_x(df.iloc[b]),y[b]))
  split_audit.append(dict(fold=j,train_lgas=sorted(set(groups[a])),validation_lgas=sorted(set(groups[b])),features=prep.cols))
 sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=10)
 pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=1)
 study=optuna.create_study(study_name=tag,direction='minimize',sampler=sampler,pruner=pruner)
 def objective(trial):
  p=params(trial,model);scores=[]
  status.write_text(json.dumps(dict(task=tag,stage='tuning',trial=trial.number,pid=os.getpid())))
  for step,(x,yt,xv,yv) in enumerate(folds):
   pred=estimator(model,p).fit(x,yt).predict(xv)
   scores.append(float(np.sqrt(np.mean((yv-pred)**2))))
   trial.report(float(np.mean(scores)),step)
   if trial.should_prune():raise optuna.TrialPruned()
  return float(np.mean(scores))
 study.optimize(objective,n_trials=TRIALS)
 prep=Prep().fit(df.iloc[tr],tier)
 def tree_x(part):return prep.transform(part)*prep.scale+prep.center
 est=estimator(model,study.best_params).fit(tree_x(df.iloc[tr]),y[tr]);pred=est.predict(tree_x(df.iloc[te]))
 assert np.isfinite(pred).all()
 joblib.dump(dict(preprocessing=prep,model=est,features=prep.cols,tree_input_unstandardisation=True),ROOT/'models'/f'{tag}.joblib',compress=3)
 joblib.dump(study,ROOT/'studies'/f'{tag}.joblib',compress=3)
 trials=[dict(number=t.number,state=t.state.name,value=t.value,params=t.params,intermediate_values=t.intermediate_values) for t in study.trials]
 result=dict(panel=panel,depth=depth,tier=tier,model=model,held_lga=held,train_rows=len(tr),train_lgas=len(set(groups[tr])),features=prep.cols,best_params=study.best_params,best_inner_rmse=study.best_value,trials=trials,inner_split_audit=split_audit,seconds=time.time()-start,predictions=[dict(lga_name=held,crop_year=int(df.iloc[i].crop_year),observed=float(y[i]),predicted=float(p)) for i,p in zip(te,pred)])
 temp=output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(output)
 status.unlink(missing_ok=True)
 return f'{tag} {time.time()-start:.1f}s'
def aggregate():
 results=[json.loads(p.read_text()) for p in (ROOT/'outputs').glob('*.json') if p.name!='environment.json']
 preds=[];tuning=[]
 for r in results:
  key={k:r[k] for k in ['panel','depth','tier','model']}
  preds.extend(key|p for p in r['predictions'])
  tuning.append(key|dict(held_lga=r['held_lga'],best_inner_rmse=r['best_inner_rmse'],best_params=json.dumps(r['best_params']),complete_trials=sum(t['state']=='COMPLETE' for t in r['trials']),pruned_trials=sum(t['state']=='PRUNED' for t in r['trials']),feature_count=len(r['features'])))
 pr=pd.DataFrame(preds)
 assert not pr.duplicated(['panel','depth','tier','model','lga_name','crop_year']).any()
 pr.to_csv(ROOT/'outputs'/'oof_predictions.csv',index=False);pd.DataFrame(tuning).to_csv(ROOT/'outputs'/'tuning_summary.csv',index=False)
 summaries=[];folds=[]
 for key,g in pr.groupby(['panel','depth','tier','model']):
  k=dict(zip(['panel','depth','tier','model'],key));summaries.append(k|metrics(g.observed.to_numpy(),g.predicted.to_numpy()))
  for lga,f in g.groupby('lga_name'):folds.append(k|dict(lga_name=lga,n=len(f),RMSE=np.sqrt(np.mean((f.observed-f.predicted)**2))))
 pd.DataFrame(summaries).to_csv(ROOT/'outputs'/'pooled_metrics.csv',index=False);pd.DataFrame(folds).to_csv(ROOT/'outputs'/'per_LGA_metrics.csv',index=False)
 return pr
if __name__=='__main__':
 specs=[]
 for panel in ['main_min5','all_cells']:
  df=pd.read_csv(ROOT/'inputs'/f'{panel}_full_candidates_0_20.csv')
  for depth,tier in [('0_20',1),('0_20',2),('0_20',3),('20_50',3)]:
   for model in ['RF','XGBoost','CatBoost']:
    for held in sorted(df.lga_name.unique()):specs.append((panel,depth,tier,model,held))
 for directory in ['outputs','models','studies']:(ROOT/directory).mkdir(exist_ok=True)
 fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'inputs').glob('*.csv')}
 source_fingerprints={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['run_models.py','preprocessing.py']}
 existing=ROOT/'outputs'/'environment.json'
 if existing.exists():
  prev=json.loads(existing.read_text())
  if prev.get('input_sha256')!=fingerprints or prev.get('source_sha256')!=source_fingerprints or prev.get('trials_per_task')!=TRIALS or prev.get('seed')!=SEED:
   raise ValueError('Inputs/settings/code changed. Preserve the previous run and use a separate clean project copy.')
 (ROOT/'outputs'/'environment.json').write_text(json.dumps(dict(seed=SEED,trials_per_task=TRIALS,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,optuna=optuna.__version__,xgboost=xgboost.__version__,catboost=catboost.__version__,tasks=len(specs),workers=WORKERS,model_threads=1,process_start='spawn',tasks_per_worker=1,prediction_clipping=False,inner_scoring='mean five-fold RMSE',input_sha256=fingerprints,source_sha256=source_fingerprints),indent=2))
 with ProcessPoolExecutor(max_workers=WORKERS,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
  fs=[pool.submit(task,s) for s in specs]
  for n,f in enumerate(as_completed(fs),1):print(f'{n}/{len(fs)} {f.result()}',flush=True)
 pr=aggregate();assert len(pr)==(58+63)*12
 print('PASS all 420 grouped tuning tasks and 1452 unique OOF predictions',flush=True)
 print(pd.read_csv(ROOT/'outputs'/'pooled_metrics.csv').to_string(index=False),flush=True)
