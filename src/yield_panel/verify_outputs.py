from pathlib import Path
import pandas as pd
import numpy as np
p=Path(__file__).parent/'outputs'
m=pd.read_csv(p/'main_min5.csv');a=pd.read_csv(p/'all_cells.csv');r=pd.read_csv(p/'private_plot_audit.csv')
assert not m.duplicated(['wave','ADM2_NAME']).any()
assert m.eligible_plots.ge(5).all()
assert np.isfinite(m.yield_kg_ha).all()
assert not r.harvest_complete_code_audit.eq(2).any()
for _,cell in m.iterrows():
    x=r[(r.wave==cell.wave)&(r.ADM2_NAME==cell.ADM2_NAME)]
    expected=np.average(x.yield_clean,weights=x.area_ha*x.survey_weight_audit)
    assert np.isclose(cell.yield_kg_ha,expected)
    assert len(x)==cell.eligible_plots
    other=a[(a.wave==cell.wave)&(a.ADM2_NAME==cell.ADM2_NAME)]
    assert len(other)==1 and np.isclose(other.yield_kg_ha.iloc[0],expected)
assert not any(c in m for c in ['hhid','plotid'])
print('PASS: uniqueness, minimum counts, finite targets, completion exclusion, weighted reconstruction, all-cell agreement, aggregate identifiers.')
