import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
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
