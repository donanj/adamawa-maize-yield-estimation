"""Post-results tuning-seed sensitivity; no held-out performance selection."""
from pathlib import Path
import sys, json, argparse, hashlib, zipfile
import numpy as np
import pandas as pd
import joblib
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src/final_model'))
import run_final as final
import run_stacking as stacking
ROOT=PROJECT/'src/stability/outputs'
SEEDS=(41001,41002,41003)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inspect(d,folder,tier,seed,budget):
 final.verify_tier(d,folder,tier,budget)
 s=joblib.load(folder/'stack.joblib');z=s['meta_training_predictions']
 x=np.column_stack([b['model'].predict(final.tx(b['preprocessing'],d)) for b in s['base_models']])
 pred=s['meta'].predict(x); meta_oof=s['meta'].predict(z)
 # meta_oof is fitted to the SAME meta-training rows, not a validated stacking prediction.
 scale=z.std(axis=0);standard=(z-z.mean(axis=0))/np.where(scale>0,scale,1)
 singular=np.linalg.svd(standard,compute_uv=False)
 cond=float(singular[0]/singular[-1]) if singular[-1]>0 else None
 summary=dict(seed=seed,tier=tier,negative_fitted=int((pred<0).sum()),minimum_fitted=float(pred.min()),maximum_fitted=float(pred.max()),meta_intercept=float(s['meta'].intercept_),standardised_oof_condition_number=cond,meta_training_fitted_rmse=float(np.sqrt(np.mean((d.yield_kg_ha.to_numpy()-meta_oof)**2))))
 rows=[];curves=[];params=[];corr=[]
 for j,name in enumerate(final.BASE):
  summary[name+'_coefficient']=float(s['meta'].coef_[j])
  summary[name+'_oof_sd']=float(z[:,j].std());summary[name+'_full_fit_sd']=float(x[:,j].std())
  summary[name+'_oof_to_full_fit_rmse']=float(np.sqrt(np.mean((x[:,j]-z[:,j])**2)))
  for k,other in enumerate(final.BASE):
   corr.append(dict(seed=seed,tier=tier,model=name,other_model=other,oof_correlation=float(np.corrcoef(z[:,j],z[:,k])[0,1]),full_fit_correlation=float(np.corrcoef(x[:,j],x[:,k])[0,1])))
 for i in range(len(d)):
  row=dict(seed=seed,tier=tier,lga_name=d.iloc[i].lga_name,crop_year=int(d.iloc[i].crop_year),observed=float(d.iloc[i].yield_kg_ha),fitted_prediction=float(pred[i]),meta_training_fitted_prediction=float(meta_oof[i]))
  for j,name in enumerate(final.BASE):row[name+'_oof']=float(z[i,j]);row[name+'_full_fit']=float(x[i,j])
  rows.append(row)
 for stage in [f'oof{k}' for k in range(5)]+['final']:
  for name in final.BASE:
   b=joblib.load(folder/f'{stage}_{name}.joblib');best=None
   params.append(dict(seed=seed,tier=tier,stage=stage,model=name,parameters=json.dumps(b['best_params'],sort_keys=True)))
   for t in b['trials']:
    if t['state']=='COMPLETE':best=t['value'] if best is None else min(best,t['value'])
    curves.append(dict(seed=seed,tier=tier,stage=stage,model=name,trial=t['number'],state=t['state'],value=t['value'],best_complete_so_far=best))
 return summary,rows,curves,params,corr

def export(d,out,sets):
 summaries=[];rows=[];curves=[];parameters=[];correlations=[]
 for seed,tier,folder,budget in sets:
  a,b,c,e,f=inspect(d,folder,tier,seed,budget);summaries.append(a);rows+=b;curves+=c;parameters+=e;correlations+=f
 for filename,values in [('ridge_diagnostics',summaries),('prediction_diagnostics',rows),('tuning_history',curves),('selected_parameters',parameters),('base_correlations',correlations)]:pd.DataFrame(values).to_csv(out/(filename+'.csv'),index=False)
 frame=pd.DataFrame(rows);variation=[]
 for (tier,lga,year),part in frame.groupby(['tier','lga_name','crop_year']):
  variation.append(dict(tier=tier,lga_name=lga,crop_year=year,runs=len(part),minimum=float(part.fitted_prediction.min()),maximum=float(part.fitted_prediction.max()),range=float(part.fitted_prediction.max()-part.fitted_prediction.min()),sd=float(part.fitted_prediction.std(ddof=0))))
 pd.DataFrame(variation).to_csv(out/'prediction_seed_variation.csv',index=False)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figdir=out/'figures';figdir.mkdir(exist_ok=True)
 h=pd.DataFrame(curves)
 for tier in sorted(h.tier.unique()):
  fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
  for ax,stage in zip(axes.flat,[f'oof{k}' for k in range(5)]+['final']):
   for (seed,model),part in h[(h.tier==tier)&(h.stage==stage)].groupby(['seed','model']):ax.plot(part.trial+1,part.best_complete_so_far,label=f'{model}: {seed}',alpha=.7,linewidth=1)
   ax.set_title(stage);ax.set_xlabel('Attempted trial');ax.set_ylabel('Best complete inner-CV RMSE (kg/ha)')
  handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8);fig.tight_layout(rect=(0,.13,1,1))
  for ext in ['png','pdf']:fig.savefig(figdir/f'T{tier}_tuning_curves.{ext}',dpi=300,bbox_inches='tight')
  plt.close(fig)
 report=['# Tuning-seed and Ridge diagnostics','', 'This diagnostic was specified after reviewing the original results. All three tiers and all five full-panel grouped OOF partitions are retained. Only Optuna sampler seed varies; estimator seeds, folds, target policy and 20-attempt budgets remain unchanged. No seed is selected as a replacement model.','', 'Full-panel fitted predictions and meta-training fitted predictions are NOT held-out stacking performance. Correlations, condition numbers and OOF/full-fit changes are descriptive diagnostics, not proof of a cause of negative predictions.','',pd.DataFrame(summaries).to_markdown(index=False) if False else pd.DataFrame(summaries).to_string(index=False),'', 'Review negative predictions, coefficient changes, tuning trajectories and seed variation before updating Chapter Five. Complete nested LOLO validation is required for any proposed alternative meta-learner.']
 (out/'Review_Report.md').write_text('\n'.join(report),encoding='utf-8')
 with zipfile.ZipFile(out/'Tuning_Stability_Review.zip','w',zipfile.ZIP_DEFLATED) as archive:
  for p in sorted(out.glob('*.csv')):archive.write(p,p.name)
  for p in [out/'manifest.json',out/'Review_Report.md']:archive.write(p,p.name)
  for p in sorted(figdir.glob('*')):archive.write(p,'figures/'+p.name)
 print(pd.DataFrame(summaries).to_string(index=False));print('PASS: saved-model reconstruction, nested group isolation and unchanged targets. Diagnostic outputs only.',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','verify']);args=p.parse_args()
 d,primary,identity=final.load('verify');out=ROOT/('smoke' if args.stage=='smoke' else 'research');out.mkdir(parents=True,exist_ok=True)
 manifest=dict(protocol='post-results full-development-panel Optuna sampler sensitivity',seeds=list(SEEDS),primary_seed=final.SEED,attempted_trials=20,estimator_seed='unchanged original',folds='unchanged GroupKFold(5)',input_hash=identity['input_hash'],script_hash=digest(Path(__file__)),original_identity=identity,scope='No held-out stacking performance or best-seed selection')
 stamp=out/'manifest.json'
 if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Diagnostic identity changed; preserve outputs and use a separate project copy.')
 if not stamp.exists():final.atomic(stamp,manifest)
 sets=[(final.SEED,t,primary/f'T{t}',20) for t in [1,2,3]]
 if args.stage!='smoke':
  print('162 additional tuning studies: three sampler seeds x three tiers x 18 studies; 20 attempts each. Saved stages resume.',flush=True)
  original=stacking.SEED
  try:
   for seed in SEEDS:
    stacking.SEED=seed # tune_fit reads this for TPE; run_models.estimator retains its original seed.
    for tier in [1,2,3]:
     folder=out/f'seed_{seed}'/f'T{tier}'
     if args.stage=='run':final.fit_tier(d,folder,tier,20)
     sets.append((seed,tier,folder,20))
  finally:stacking.SEED=original
 export(d,out,sets)
 if args.stage=='smoke':print('SMOKE: reads the three primary saved models; no repeat fitting and no stability conclusion.')
if __name__=='__main__':main()
