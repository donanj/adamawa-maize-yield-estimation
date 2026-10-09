"""Audit newly downloaded GEE exports before feature engineering."""
from pathlib import Path
import argparse,json,hashlib,zipfile
import pandas as pd
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
YEARS=[2010,2012,2015,2018,2023];MODES=['unmasked','cropland12','cropland12_14']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=PROJECT/'data/raw/cropland_mask');args=ap.parse_args();folder=args.input
 expected=[f'Adamawa_cropland_{kind}_{y}_v2.csv' for y in YEARS for kind in ['vegetation','thermal']]+['Adamawa_cropland_area_fractions_v1.csv']
 missing=[name for name in expected if not (folder/name).is_file()]
 if missing:raise FileNotFoundError('Missing exports: '+', '.join(missing))
 areas=pd.read_csv(folder/expected[-1]);key=['ADM2_NAME','crop_year'];assert len(areas)==105 and not areas.duplicated(key).any();assert sorted(areas.crop_year.unique())==YEARS
 names=set(areas.ADM2_NAME);assert len(names)==21 and {'Girei','Toungo'}.issubset(names) and not {'Girie','Teungo'}&names
 cols=['cropland12_fraction','mosaic14_fraction','other_classified_fraction','unknown_fraction']
 assert areas[cols].notna().all().all() and np.isfinite(areas[cols]).all().all();assert ((areas[cols]>=-1e-6)&(areas[cols]<=1+1e-6)).all().all();assert np.allclose(areas[cols].sum(axis=1),1,atol=1e-6);assert np.allclose(areas.cropland12_14_fraction,areas.cropland12_fraction+areas.mosaic14_fraction)
 summary=[];diffs=[]
 for kind in ['vegetation','thermal']:
  bands=['NDVI','EVI'] if kind=='vegetation' else ['LST_day_C','LST_night_C']
  for year in YEARS:
   df=pd.read_csv(folder/f'Adamawa_cropland_{kind}_{year}_v2.csv');k=['ADM2_NAME','date','mask_mode'];assert not df.duplicated(k).any();assert df.export_version.eq('cropland_sensitivity_v2').all();assert df.count_policy.eq('separate_centroid_counts_on_product_grid').all();assert df.crop_year.eq(year).all() and df.landcover_year.eq(year).all();assert set(df.mask_mode)==set(MODES)
   assert df.groupby(['date','mask_mode']).ADM2_NAME.apply(set).apply(lambda a:a==names).all();assert df.groupby(['ADM2_NAME','date']).size().eq(3).all()
   dates=pd.to_datetime(df.date);assert dates.dt.year.eq(year).all();assert dates.dt.month.between(4 if kind=='vegetation' else 5,11 if kind=='vegetation' else 10).all();assert (df.eligible_grid_count>=0).all()
   eligible=df.pivot(index=['ADM2_NAME','date'],columns='mask_mode',values='eligible_grid_count')
   assert (eligible.cropland12<=eligible.cropland12_14+1e-6).all() and (eligible.cropland12_14<=eligible.unmasked+1e-6).all()
   for band in bands:
    count=df[band+'_count'].fillna(0);assert (count>=0).all() and (count<=df.eligible_grid_count+1e-6).all();assert df.loc[count.gt(0),band+'_mean'].notna().all() # Weighted means can include boundary pixels even when centroid counts are zero.
    computed=count/df.eligible_grid_count.replace(0,np.nan);assert np.allclose(computed,df[band+'_valid_fraction'],equal_nan=True,atol=1e-6)
    if kind=='vegetation':assert df.loc[count.gt(0),band+'_mean'].between(-1,1).all()
    for mode,part in df.groupby('mask_mode'):summary.append(dict(product=kind,year=year,mask_mode=mode,band=band,rows=len(part),valid_composite_rows=int(part[band+'_mean'].notna().sum()),zero_eligible_rows=int(part.eligible_grid_count.eq(0).sum()),boundary_only_mean_rows=int((part[band+'_count'].fillna(0).eq(0)&part[band+'_mean'].notna()).sum()),mean_valid_grid_fraction=float(part[band+'_valid_fraction'].mean())))
    pivot=df.pivot(index=['ADM2_NAME','crop_year','date'],columns='mask_mode',values=band+'_mean')
    for mode in MODES[1:]:
     paired=pivot[['unmasked',mode]].dropna();delta=paired[mode]-paired.unmasked
     diffs.append(dict(product=kind,year=year,band=band,mask_mode=mode,matched_rows=len(paired),mean_difference=float(delta.mean()),mean_absolute_difference=float(delta.abs().mean())))
 out=PROJECT/'src/cropland_mask/outputs/export_audit';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(summary).to_csv(out/'coverage_summary.csv',index=False);pd.DataFrame(diffs).to_csv(out/'composite_differences.csv',index=False);areas.to_csv(out/'area_fractions.csv',index=False)
 (out/'input_hashes.json').write_text(json.dumps({name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in expected},indent=2))
 (out/'Review_Report.md').write_text('# Cropland extraction audit\n\nPASS: 105 area cells; matched modes, dates and 21-LGA coverage; annual alignment; valid fractions and class-area closure. Seasonal features and models have not yet been rebuilt. The new unmasked controls must also be reconciled with the original extraction before model comparison.\n\n'+pd.DataFrame(summary).to_string(index=False),encoding='utf-8')
 with zipfile.ZipFile(out/'Cropland_Extraction_Review.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in expected:z.write(folder/name,'raw/'+name)
  for p in out.iterdir():
   if p.suffix in ['.csv','.json','.md']:z.write(p,p.name)
 print(pd.DataFrame(summary).to_string(index=False));print('PASS: separate centroid-count coverage and mask audit. No new model results yet.')
 print('Upload:',out/'Cropland_Extraction_Review.zip')
if __name__=='__main__':main()
