from pathlib import Path
import runpy
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent
ns=runpy.run_path(str(root/'build_features.py'))
smooth,integral,crossing=ns['smooth'],ns['integral'],ns['crossing']
a,flags=smooth([np.nan,1,2,np.nan,4,5,6,np.nan]);assert np.isnan(a[[0,7]]).all() and flags[3] and flags.sum()==1
b,_=smooth([1,2,np.nan,np.nan,np.nan,6,7]);assert np.isnan(b).all()
assert np.isclose(integral(np.arange(8.),np.ones(8),1,6),5)
assert np.isnan(integral(np.arange(8.),np.array([1,1,1,np.nan,1,1,1,1]),1,6))
assert np.isnan(integral(np.arange(8.),np.ones(8),-1,6))
assert np.isclose(crossing(np.arange(3.),np.array([0.,1.,0.]),.2,True),.2)
assert np.isclose(crossing(np.arange(3.),np.array([0.,1.,0.]),.2,False),1.8)
for panel,count,lgas in [('main_min5',58,17),('all_cells',63,18),('exclude_above_10000_min5',58,17)]:
 src=pd.read_csv(root/'inputs'/(panel+'.csv'))
 for depth in ['0_20','20_50']:
  out=pd.read_csv(root/'outputs'/(panel+'_features_soil_'+depth+'.csv'))
  assert len(out)==count and out.lga_name.nunique()==lgas
  assert not out.duplicated(['lga_name','crop_year']).any()
  assert np.allclose(out.yield_kg_ha,src.yield_kg_ha)
  assert np.isfinite(out[[f'V{i:02}' for i in range(1,34)]].to_numpy()[out[[f'V{i:02}' for i in range(1,34)]].notna().to_numpy()]).all()
  assert np.allclose(out.V19,ns['features'].set_index(['lga_name','crop_year']).loc[list(zip(out.lga_name,out.crop_year)),'V19'])
  assert out.groupby('lga_name').V32.nunique().eq(1).all()
print('PASS: gap handling, no endpoint extrapolation, exact-date integration, threshold crossings, keys, counts and unchanged targets.')
