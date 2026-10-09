"""Explain final full-panel models; no model training during explanation."""
from pathlib import Path
import argparse,json,platform,zipfile
from importlib.metadata import version
import numpy as np
import pandas as pd
from run_final import load,verify_tier,atomic,digest,PROJECT
import final_shap_utils as u

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','smoke','run','verify']);args=parser.parse_args()
 d,root,env=load('verify');tiers=[1] if args.stage=='smoke' else [1,2,3]
 for tier in tiers:verify_tier(d,root/f'T{tier}',tier,20)
 out=root/('shap_smoke' if args.stage=='smoke' else 'shap');out.mkdir(parents=True,exist_ok=True);u.OUT=out
 manifest=dict(input_hash=env['input_hash'],python=platform.python_version(),packages={n:version(n) for n in ['numpy','pandas','scipy','scikit-learn','xgboost','catboost','optuna','shap','joblib','matplotlib','numba']},model_hashes={f'T{t}/{n}':digest(root/f'T{t}'/n) for t in tiers for n in ['stack.joblib','result.json']},script_hashes={n:digest(Path(__file__).parent/n) for n in ['run_final.py','run_final_shap.py','final_shap_utils.py']},dictionary_hash=digest(PROJECT/'src/feature_engineering/feature_dictionary.csv'),semantics='interventional; all 58 development rows equally weighted; model-fitted values; saved Ridge weights; non-causal')
 stamp=out/'manifest.json'
 if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Explanation identity changed; preserve outputs and use a separate copy.')
 if args.stage=='check':print('PASS: three final models and grouped training/prediction reconstruction.');return
 if not stamp.exists():atomic(stamp,manifest)
 values=[]
 for tier in tiers:
  path=out/f'T{tier}_explanation.json'
  if args.stage=='verify' and not path.exists():raise FileNotFoundError(path)
  # Recompute against the actual saved models even in verification mode.
  value=u.explain(d,tier,'final model',root/f'T{tier}')
  if args.stage=='verify':
   old=json.loads(path.read_text())
   u.close(np.array([r['predicted'] for r in old['records']]),np.array([r['predicted'] for r in value['records']]),'saved explanation predictions')
   u.close(np.array([r['shap_value'] for r in old['features']]),np.array([r['shap_value'] for r in value['features']]),'saved explanation values')
  else:atomic(path,value)
  values.append(value)
 records=pd.DataFrame([x for v in values for x in v['records']]);features=pd.DataFrame([x for v in values for x in v['features']]);components=pd.DataFrame([x for v in values for x in v['base_components']])
 keys=['tier','lga_name','crop_year'];assert not records.duplicated(keys).any() and len(records)==58*len(tiers)
 u.close(records.reconstructed.to_numpy(),records.predicted.to_numpy(),'full-panel additivity')
 sums=features.groupby(keys).shap_value.sum().reset_index(name='sum_shap');joined=records.merge(sums,on=keys,validate='one_to_one');u.close((joined.baseline+joined.sum_shap).to_numpy(),joined.predicted.to_numpy(),'export sums')
 weighted=components.groupby(keys+['feature_id']).weighted_shap.sum().reset_index(name='weighted');paired=features.merge(weighted,on=keys+['feature_id'],how='left',validate='one_to_one').fillna({'weighted':0});u.close(paired.shap_value.to_numpy(),paired.weighted.to_numpy(),'weighted components')
 for tier,part in records.groupby('tier'):
  joined=part.merge(d[['lga_name','crop_year','yield_kg_ha']],on=['lga_name','crop_year'],validate='one_to_one');u.close(joined.observed.to_numpy(),joined.yield_kg_ha.to_numpy(),'unchanged explanation targets')
 if args.stage=='smoke':records.to_csv(out/'smoke_reconstruction.csv',index=False);print('PASS: full-panel Tier 1 SHAP reconstruction. Smoke only; no thesis figures.');return
 names=dict(zip(*(pd.read_csv(PROJECT/'src/feature_engineering/feature_dictionary.csv')[['feature_id','name']].to_numpy().T)))
 features['abs_shap']=features.shap_value.abs()
 importance=features.groupby(['tier','feature_id']).agg(mean_abs_shap=('abs_shap','mean'),mean_signed_shap=('shap_value','mean'),included_observations=('included','sum')).reset_index();importance['name']=importance.feature_id.map(names);importance=importance.sort_values(['tier','mean_abs_shap','feature_id'],ascending=[True,False,True])
 records['prediction_scope']='in_sample_final_model'
 for name,frame in [('fitted_reconstruction.csv',records),('feature_shap_values.csv',features),('base_shap_components.csv',components),('global_importance.csv',importance)]:frame.to_csv(out/name,index=False)
 atomic(out/'base_audit.json',[dict(tier=t,audit=v['audit']) for t,v in zip(tiers,values)])
 u.figures(records,features,importance,names)
 # Original diagnostic waterfall rule is retained, but all errors here are fitted residuals.
 (out/'Review_Report.md').write_text('''# Final full-panel model explanations
Three independently fitted tier models; each trained on all 58 observations across 17 LGAs.
Each Ridge meta-learner uses fully nested, LGA-grouped OOF base predictions; alpha is fixed at 1.0.
Each base stage uses 20 attempted Optuna trials, including pruned trials. Five group folds are used at both nesting levels.
Interventional SHAP background: all 58 development observations, equally weighted, transformed by each final saved preprocessing object.
174 explanations reconstruct saved fitted predictions. They are in-sample model descriptions, not held-out performance or causal effects.
Use the existing LOLO and LOYO results for generalisation assessment. Do not report these fitted residuals as validation metrics.
Mean absolute SHAP includes zeros for excluded candidates; excluded features are not demonstrated biologically irrelevant.
Dependence plots show final-model processed values. Waterfall examples are selected by largest, lower-middle and smallest absolute fitted residuals in Tier 3; they are diagnostic, not representative.
All figures are generated locally as PNG (400 dpi) and PDF. Tuning histories provide within-study diagnostics, not repeated-seed stability evidence.
XGBoost numeric-Booster and CatBoost scale/bias compatibility checks preserve the existing interventional explanation protocol.
''')
 archive=root/'Final_Full_Panel_Review.zip'
 files=list(out.glob('*.csv'))+list(out.glob('*.json'))+[out/'Review_Report.md']+list((out/'figures').glob('*'))
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in files:z.write(f,'shap/'+str(f.relative_to(out)))
  for name in ['environment.json','fitted_predictions.csv','tuning_history.csv','selected_parameters.csv']:z.write(root/name,name)
  for tier in tiers:z.write(root/f'T{tier}/result.json',f'model_audits/T{tier}_result.json')
 print(f'PASS: 174 final-model explanations, unchanged targets, base/stack reconstruction and figures.\nUpload: {archive}',flush=True)
if __name__=='__main__':main()
