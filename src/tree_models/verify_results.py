from pathlib import Path
import json
import numpy as np
import pandas as pd
from preprocessing import Prep
from predict_saved import predict_saved
R=Path(__file__).resolve().parent
# Verify partition-specific reliability gate, winsorisation and imputation.
t=pd.DataFrame({f'V{i:02}':np.arange(10,dtype=float) for i in range(1,34)})
t.loc[:2,['V01','V02']]=np.nan
t['V27']=np.nan;t['V16']=5
p=Prep().fit(t,1)
assert not any(c in p.cols for c in ['V01','V02','V03','V06'])
probe=t.iloc[[0]].copy();probe['V04']=1e12;probe['V05']=np.nan
x=p.transform(probe)*p.scale+p.center
assert np.isclose(x[0,p.cols.index('V04')],p.hi[p.cols.index('V04')])
assert np.isclose(x[0,p.cols.index('V05')],p.mean[p.cols.index('V05')])
audit=pd.read_csv(R/'shared_LOLO_splits.csv')
pr=pd.read_csv(R/'outputs/oof_predictions.csv')
assert len(pr)==1452 and not pr.duplicated(['panel','depth','tier','model','lga_name','crop_year']).any()
for key,g in pr.groupby(['panel','depth','tier','model']):
 panel,depth,tier,model=key
 expected=audit[audit.panel==panel][['lga_name','crop_year']]
 assert set(map(tuple,g[['lga_name','crop_year']].to_numpy()))==set(map(tuple,expected.to_numpy()))
 source=pd.read_csv(R/'inputs'/f'{panel}_full_candidates_{depth}.csv')
 joined=g.merge(source[['lga_name','crop_year','yield_kg_ha']],on=['lga_name','crop_year'],validate='one_to_one')
 assert np.allclose(joined.observed,joined.yield_kg_ha)
for pth in (R/'outputs').glob('*.json'):
 if pth.name=='environment.json':continue
 d=json.loads(pth.read_text());assert len(d['trials'])==20
 assert sum(t['state']=='COMPLETE' for t in d['trials'])>=1
 assert d['held_lga'] not in set(sum([s['train_lgas']+s['validation_lgas'] for s in d['inner_split_audit']],[]))
 for s in d['inner_split_audit']:assert set(s['train_lgas']).isdisjoint(s['validation_lgas'])
 assert 'V27' not in d['features'] and 'V16' not in d['features']
 source=pd.read_csv(R/'inputs'/f"{d['panel']}_full_candidates_{d['depth']}.csv").sort_values(['lga_name','crop_year'])
 held=source[source.lga_name==d['held_lga']]
 restored=predict_saved(R/'models'/f'{pth.stem}.joblib',held)
 assert np.allclose(restored,[v['predicted'] for v in d['predictions']],rtol=1e-10,atol=1e-8)
print('PASS: training-specific gate/clipping/imputation, unchanged targets, shared split coverage, all 20-trial tasks and nested group isolation.')
