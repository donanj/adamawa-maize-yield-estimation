"""Least-squares convex combination: nonnegative weights sum to one, no intercept."""
import itertools
import numpy as np

def fit_convex(z,y):
 z=np.asarray(z,float);y=np.asarray(y,float)
 if z.ndim!=2 or y.shape!=(len(z),) or not np.isfinite(z).all() or not np.isfinite(y).all():raise ValueError('Invalid meta-training inputs')
 if z.shape[1]!=3:raise ValueError('Expected three base learners')
 scale=max(float(np.abs(z).max()),float(np.abs(y).max()),1.0);x=z/scale;t=y/scale;candidates=[]
 for n in range(1,4):
  for subset in itertools.combinations(range(3),n):
   ix=list(subset);w=np.zeros(3)
   if n==1:w[ix[0]]=1
   else:
    anchor=x[:,ix[-1]];a=x[:,ix[:-1]]-anchor[:,None]
    v=np.linalg.lstsq(a,t-anchor,rcond=None)[0];values=np.r_[v,1-v.sum()]
    if values.min() < -1e-10:continue
    values=np.maximum(values,0);values/=values.sum();w[ix]=values
   candidates.append((float(np.mean((x@w-t)**2)),w))
 loss,w=min(candidates,key=lambda a:a[0]);gradient=2*x.T@(x@w-t)/len(t)
 if np.min(gradient-gradient@w)<-1e-8:raise ValueError('Convex optimality check failed')
 if w.min()<0 or not np.isclose(w.sum(),1):raise ValueError('Invalid weights')
 return w
