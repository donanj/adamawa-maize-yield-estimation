from pathlib import Path
import pandas as pd,numpy as np
BASE=Path(__file__).resolve().parent;frames=[];audit=[]
expected=set(pd.read_csv(BASE/'inputs'/'terrain.csv').lga_name)
for year in [2010,2012,2015,2018,2023]:
 f=BASE/'inputs'/f'Adamawa_MOD11A2_day_night_{year}_v3.csv';d=pd.read_csv(f)
 assert len(d)==483 and set(d.ADM2_NAME)==expected and d.crop_year.eq(year).all()
 assert not d.duplicated(['ADM2_NAME','date']).any()
 grid=pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='8D');dates=set(grid[(grid.month>=5)&(grid.month<11)].strftime('%Y-%m-%d'))
 for _,g in d.groupby('ADM2_NAME'):assert set(g.date)==dates
 assert d.export_version.eq('day_night_QCmask_repaired_v3').all() and d.scale_m.eq(1000).all()
 assert d['product'].eq('MODIS/061/MOD11A2').all()
 assert d.qa_policy.eq('mandatory_0_data_0_error_le1K_QC_zero_restored_only_with_valid_LST').all()
 for period in ['day','night']:
  value=d[f'LST_{period}_C_mean'];count=d[f'LST_{period}_C_count'];rest=d[f'{period}_qc_zero_restored_count']
  assert np.isfinite(value.dropna()).all() and value.dropna().between(-50,70).all()
  assert count.ge(0).all() and rest.ge(0).all() and rest.le(count).all()
  assert ((count==0)|value.notna()).all()
 frames.append(d)
 audit.append({'year':year,'rows':len(d),'day_valid':int(d.LST_day_C_mean.notna().sum()),'night_valid':int(d.LST_night_C_mean.notna().sum()),'day_missing':int(d.LST_day_C_mean.isna().sum()),'night_missing':int(d.LST_night_C_mean.isna().sum()),'day_rows_with_restored_QC':int(d.day_qc_zero_restored_count.gt(0).sum()),'night_rows_with_restored_QC':int(d.night_qc_zero_restored_count.gt(0).sum())})
new=pd.concat(frames,ignore_index=True);new.to_csv(BASE/'outputs/lst_day_night_v3_verified.csv',index=False)
pd.DataFrame(audit).to_csv(BASE/'outputs/lst_v3_file_audit.csv',index=False)
old=pd.read_csv(BASE/'inputs/lst_day.csv');m=old.merge(new,on=['ADM2_NAME','crop_year','date'],suffixes=('_old','_new'),validate='one_to_one')
m['day_mean_change']=m.LST_day_C_mean_new-m.LST_day_C_mean_old
m[['ADM2_NAME','crop_year','date','LST_day_C_mean_old','LST_day_C_mean_new','day_mean_change']].to_csv(BASE/'outputs/lst_day_version_comparison.csv',index=False)
print(pd.DataFrame(audit).to_string(index=False));print('Day max change',m.day_mean_change.abs().max())
