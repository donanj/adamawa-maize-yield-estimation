"""Reproduce spatial linear benchmarks; run: python run_benchmarks.py."""
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
ALPHAS=[0.01,0.1,1.0,10.0,100.0,1000.0]
PH=['V01','V02','V03','V06']
TIERS={1:list(range(1,13)),2:list(range(1,23)),3:list(range(1,34))}
class Prep:
 def fit(self,df,tier,vif=False):
  ph=df[['V01','V02']].notna().all(axis=1).mean()>=.8
  self.cols=[f'V{i:02}' for i in TIERS[tier] if i not in [16,27] and (f'V{i:02}' not in PH or ph) and df[f'V{i:02}'].nunique(dropna=True)>1]
  x=df[self.cols].to_numpy(float)
  self.lo=np.nanquantile(x,.025,axis=0); self.hi=np.nanquantile(x,.975,axis=0)
  x=np.clip(x,self.lo,self.hi);self.mean=np.nanmean(x,axis=0);x=np.where(np.isnan(x),self.mean,x)
  self.center=x.mean(axis=0);self.scale=x.std(axis=0);self.scale[self.scale==0]=1
  x=(x-self.center)/self.scale
  self.indices=list(range(x.shape[1]));self.removed=[]
  if vif:
   while len(self.indices)>1:
    z=x[:,self.indices]; vals=[]
    for j in range(z.shape[1]):
     other=np.delete(z,j,axis=1);r2=LinearRegression().fit(other,z[:,j]).score(other,z[:,j]); vals.append(1/max(1-r2,1e-14))
    j=int(np.argmax(vals))
    if vals[j]<=10 and np.linalg.matrix_rank(np.column_stack([np.ones(len(z)),z]))==z.shape[1]+1:break
    self.removed.append({'feature':self.cols[self.indices[j]],'vif':vals[j]});self.indices.pop(j)
  return self
 def transform(self,df):
  x=np.clip(df[self.cols].to_numpy(float),self.lo,self.hi);x=np.where(np.isnan(x),self.mean,x)
  return ((x-self.center)/self.scale)[:,self.indices]
def metrics(y,p):
 rmse=float(np.sqrt(mean_squared_error(y,p)))
 return dict(RMSE=rmse,MAE=float(mean_absolute_error(y,p)),R2=float(r2_score(y,p)),RRMSE=100*rmse/np.mean(y),MAPE=100*np.mean(np.abs((y[y>0]-p[y>0])/y[y>0])))
preds=[];splits=[];audit=[];tuning=[]
for panel in ['main_min5','all_cells']:
 for depth in ['0_20','20_50']:
  df=pd.read_csv(ROOT/'inputs'/f'{panel}_full_candidates_{depth}.csv').sort_values(['lga_name','crop_year']).reset_index(drop=True)
  assert not df.duplicated(['lga_name','crop_year']).any()
  groups=df.lga_name.to_numpy();y=df.yield_kg_ha.to_numpy()
  for held in sorted(set(groups)):
   tr=np.flatnonzero(groups!=held);te=np.flatnonzero(groups==held)
   assert set(groups[tr]).isdisjoint(groups[te])
   if depth=='0_20':
    for ix in te:splits.append(dict(panel=panel,lga_name=held,crop_year=int(df.iloc[ix].crop_year),fold=held,train_rows=len(tr),train_lgas=len(set(groups[tr])),test_rows=len(te)))
   specs=[('mean',0)]+[('MLR',t) for t in [1,2,3]]+[('Ridge',3)]
   for model,tier in specs:
    alpha=None
    if model=='mean':p=np.repeat(y[tr].mean(),len(te));selected=[];removed=[]
    else:
     if model=='Ridge':
      errors={a:[] for a in ALPHAS}
      for inner,(it,iv) in enumerate(GroupKFold(5).split(tr,groups=groups[tr])):
       itr=tr[it];iva=tr[iv];assert set(groups[itr]).isdisjoint(groups[iva])
       prep=Prep().fit(df.iloc[itr],tier)
       for a in ALPHAS:
        ip=Ridge(alpha=a).fit(prep.transform(df.iloc[itr]),y[itr]).predict(prep.transform(df.iloc[iva]))
        errors[a].append(float(np.sqrt(mean_squared_error(y[iva],ip))))
      alpha=min(ALPHAS,key=lambda a:np.mean(errors[a]))
      for a in ALPHAS:tuning.append(dict(panel=panel,depth=depth,held_lga=held,alpha=a,mean_inner_RMSE=np.mean(errors[a]),selected=a==alpha))
     prep=Prep().fit(df.iloc[tr],tier,model=='MLR')
     est=LinearRegression() if model=='MLR' else Ridge(alpha=alpha)
     p=est.fit(prep.transform(df.iloc[tr]),y[tr]).predict(prep.transform(df.iloc[te]))
     selected=[prep.cols[i] for i in prep.indices];removed=prep.removed
     assert np.linalg.matrix_rank(np.column_stack([np.ones(len(tr)),prep.transform(df.iloc[tr])]))==len(selected)+1 or model=='Ridge'
    audit.append(dict(panel=panel,depth=depth,held_lga=held,model=model,tier=tier,alpha=alpha,feature_count=len(selected),selected_features=';'.join(selected),removed_vif=json.dumps(removed)))
    for ix,yp in zip(te,p):preds.append(dict(panel=panel,depth=depth,model=model,tier=tier,lga_name=held,crop_year=int(df.iloc[ix].crop_year),observed=y[ix],predicted=float(yp)))
pr=pd.DataFrame(preds);pr.to_csv(OUT/'oof_predictions.csv',index=False)
pd.DataFrame(splits).to_csv(OUT/'shared_LOLO_splits.csv',index=False)
pd.DataFrame(audit).to_csv(OUT/'preprocessing_audit.csv',index=False)
pd.DataFrame(tuning).to_csv(OUT/'ridge_inner_tuning.csv',index=False)
summary=[];fold=[]
for key,d in pr.groupby(['panel','depth','model','tier']):
 summary.append(dict(zip(['panel','depth','model','tier'],key))|metrics(d.observed.to_numpy(),d.predicted.to_numpy()))
 for lga,g in d.groupby('lga_name'):fold.append(dict(zip(['panel','depth','model','tier'],key))|dict(lga_name=lga,RMSE=float(np.sqrt(mean_squared_error(g.observed,g.predicted))),n=len(g)))
pd.DataFrame(summary).to_csv(OUT/'pooled_metrics.csv',index=False);pd.DataFrame(fold).to_csv(OUT/'per_LGA_metrics.csv',index=False)
assert len(pr)==(58+63)*2*5
assert pr.groupby(['panel','depth','model','tier']).apply(lambda g: not g.duplicated(['lga_name','crop_year']).any(),include_groups=False).all()
(OUT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,ridge_grid=ALPHAS,inner_scoring='unweighted mean of five validation-fold RMSEs',prediction_clipping=False),indent=2))
print(pd.DataFrame(summary).to_string(index=False))
print('PASS: every observation predicted once per configuration; grouped splits and OLS rank checks passed.')
