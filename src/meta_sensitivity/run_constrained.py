"""Post-results convex meta-learner sensitivity using verified nested base fits."""
from pathlib import Path
import sys,json,argparse,hashlib,zipfile
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src/final_model'))
import run_final as final
from convex_meta import fit_convex
ROOT=PROJECT/'src/meta_sensitivity/outputs'
BASE_ROOT=PROJECT/'src/stacking/outputs/research/reference_99/main_min5/0_20'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get_inputs(d,tier,held):
 folder=BASE_ROOT/f'main_min5_0_20_T{tier}_{held.replace(" ","_")}'
 r=json.loads((folder/'result.json').read_text());s=joblib.load(folder/'stack.joblib')
 g=d.lga_name.to_numpy();y=d.yield_kg_ha.to_numpy();tr=np.flatnonzero(g!=held);te=np.flatnonzero(g==held)
 assert r['train_rows']==tr.tolist() and r['test_rows']==te.tolist()
 assert s['base_order']==final.BASE and np.array_equal(s['meta_training_rows'],tr) and np.array_equal(s['meta_training_targets'],y[tr])
 z=np.full((len(tr),3),np.nan);coverage=np.zeros(len(d),int)
 for split in r['stacking_splits']:
  a=np.array(split['train_rows']);b=np.array(split['validation_rows'])
  assert set(a)|set(b)==set(tr) and set(a).isdisjoint(b) and set(g[a]).isdisjoint(g[b]);coverage[b]+=1
  for j,name in enumerate(final.BASE):
   bundle=joblib.load(folder/f"{split['stage']}_{name}.joblib");final.check_bundle(bundle,a,g,20)
   z[np.searchsorted(tr,b),j]=bundle['model'].predict(final.tx(bundle['preprocessing'],d.iloc[b]))
 assert np.all(coverage[tr]==1) and np.all(coverage[te]==0) and np.isfinite(z).all() and np.allclose(z,s['meta_training_predictions'])
 ridge=Ridge(alpha=1.0).fit(z,y[tr]);assert np.allclose(ridge.coef_,s['meta'].coef_) and np.allclose(ridge.intercept_,s['meta'].intercept_)
 x=[]
 for bundle in s['base_models']:
  final.check_bundle(bundle,tr,g,20);x.append(bundle['model'].predict(final.tx(bundle['preprocessing'],d.iloc[te])))
 x=np.column_stack(x);original=s['meta'].predict(x)
 assert np.allclose(original,[p['predicted'] for p in r['predictions']])
 for i,p in zip(te,r['predictions']):assert p['lga_name']==held and p['crop_year']==int(d.iloc[i].crop_year) and p['observed']==float(y[i])
 return tr,te,z,x,original

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','verify']);args=p.parse_args()
 d,_,identity=final.load('verify');out=ROOT/('smoke' if args.stage=='smoke' else 'research');out.mkdir(parents=True,exist_ok=True)
 tiers=[1] if args.stage=='smoke' else [1,2,3];held=sorted(d.lga_name.unique());held=held[:1] if args.stage=='smoke' else held
 manifest=dict(protocol='post-results convex meta sensitivity; nonnegative weights sum to one, zero intercept',input_identity=identity,script_hash=digest(Path(__file__)),solver_hash=digest(Path(__file__).with_name('convex_meta.py')),base_environment_hash=digest(BASE_ROOT/'environment.json'),tiers=tiers,lgas=held,scope='same nested base fits; original Ridge preserved; no meta hyperparameters or outer-outcome selection')
 stamp=out/'manifest.json'
 if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Identity changed; preserve existing results and use a separate copy')
 if not stamp.exists():final.atomic(stamp,manifest)
 records=[];weights=[]
 for tier in tiers:
  for lga in held:
   print(f'T{tier} {lga}: verifying saved nested base fits',flush=True)
   tr,te,z,x,original=get_inputs(d,tier,lga);w=fit_convex(z,d.iloc[tr].yield_kg_ha.to_numpy());pred=x@w
   assert np.isfinite(pred).all() and np.all(pred>=x.min(axis=1)-1e-7) and np.all(pred<=x.max(axis=1)+1e-7)
   dest=out/f'T{tier}_{lga.replace(" ","_")}.json';value=dict(tier=tier,held_lga=lga,train_rows=tr.tolist(),test_rows=te.tolist(),weights=w.tolist(),predictions=pred.tolist(),reference_predictions=original.tolist())
   if args.stage=='verify':
    saved=json.loads(dest.read_text());assert saved['train_rows']==tr.tolist() and saved['test_rows']==te.tolist();assert np.allclose(saved['weights'],w) and np.allclose(saved['predictions'],pred) and np.allclose(saved['reference_predictions'],original)
   elif dest.exists():
    saved=json.loads(dest.read_text());assert saved==value
   else:final.atomic(dest,value)
   weights.append(dict(tier=tier,held_lga=lga,**dict(zip(final.BASE,w)),intercept=0))
   for j,i in enumerate(te):
    for model,predicted in [('Convex stacking',pred[j]),('Original Ridge stacking',original[j])]:records.append(dict(tier=tier,model=model,lga_name=lga,crop_year=int(d.iloc[i].crop_year),observed=float(d.iloc[i].yield_kg_ha),predicted=float(predicted)))
   print(f'T{tier} {lga}: PASS weights={np.round(w,4).tolist()}',flush=True)
 frame=pd.DataFrame(records);assert not frame.duplicated(['tier','model','lga_name','crop_year']).any()
 from preprocessing import metrics
 summaries=[];folds=[]
 for (tier,model),part in frame.groupby(['tier','model']):
  assert len(part)==(len(d) if args.stage!='smoke' else (d.lga_name==held[0]).sum())
  summaries.append(dict(tier=tier,model=model,negative_predictions=int((part.predicted<0).sum()),**metrics(part.observed.to_numpy(),part.predicted.to_numpy())))
  for lga,group in part.groupby('lga_name'):folds.append(dict(tier=tier,model=model,lga_name=lga,RMSE=float(np.sqrt(np.mean((group.observed-group.predicted)**2)))))
 frame.to_csv(out/'oof_predictions.csv',index=False);pd.DataFrame(weights).to_csv(out/'meta_weights.csv',index=False);pd.DataFrame(summaries).to_csv(out/'pooled_metrics.csv',index=False);pd.DataFrame(folds).to_csv(out/'per_LGA_metrics.csv',index=False)
 (out/'Review_Report.md').write_text('# Convex stacking sensitivity\n\nPost-results sensitivity. Base learners and their complete nested grouped tuning protocol are unchanged. New meta weights minimise squared error on the outer-training OOF matrix, constrained to be nonnegative and sum to one, without an intercept. No outer-test target enters meta fitting. These are matched LOLO predictions, not fitted development predictions. Primary Ridge results are preserved. Nonnegativity of a convex prediction requires nonnegative constituent predictions; the implementation reports negative counts rather than clipping. No new inferential tests or uncertainty intervals are claimed.\n\n'+pd.DataFrame(summaries).to_string(index=False),encoding='utf-8')
 with zipfile.ZipFile(out/'Constrained_Meta_Review.zip','w',zipfile.ZIP_DEFLATED) as archive:
  for f in sorted(out.iterdir()):
   if f.suffix in ['.csv','.json','.md']:archive.write(f,f.name)
 print(pd.DataFrame(summaries).to_string(index=False));print('PASS: nested group isolation, original Ridge reconstruction, target identity, convex optimality and held-out coverage.');
 if args.stage=='smoke':print('SMOKE ONLY: one existing outer fold; not thesis performance.')
if __name__=='__main__':main()
