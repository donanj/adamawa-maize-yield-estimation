"""Rebuild mask sensitivities with unmodified local temporal feature code."""
from pathlib import Path
import sys,argparse,subprocess,shutil,hashlib,json,zipfile
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[2]
YEARS=[2010,2012,2015,2018,2023];MODES=['unmasked','cropland12','cropland12_14'];KEY=['lga_name','crop_year'];SAT=[f'V{i:02}' for i in range(1,13)];NON=[f'V{i:02}' for i in range(13,34)]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=PROJECT);args=ap.parse_args();project=args.project.resolve();raw=project/'data/raw/cropland_mask';original=project/'src/feature_engineering';root=project/'src/cropland_mask/outputs/feature_rebuild';root.mkdir(parents=True,exist_ok=True)
 required=[raw/f'Adamawa_cropland_{kind}_{year}_v2.csv' for kind in ['vegetation','thermal'] for year in YEARS]+[raw/'Adamawa_cropland_area_fractions_v1.csv']
 for f in required:
  if not f.exists():raise FileNotFoundError(f)
 stamp=root/'manifest.json';identity=dict(builder_hash=digest(original/'build_features.py'),raw_hashes={p.name:digest(p) for p in required},original_inputs={p.name:digest(p) for p in sorted((original/'inputs').glob('*.csv'))},protocol='same temporal feature code copied unchanged into isolated mask workspaces; no target-based processing')
 if stamp.exists() and json.loads(stamp.read_text())!=identity:raise ValueError('Input/code identity changed; preserve outputs and use a separate project copy.')
 if not stamp.exists():stamp.write_text(json.dumps(identity,indent=2))
 controls=[]
 for kind,pattern,bands in [('vegetation','Adamawa_MOD13Q1_buffer_*.csv',['NDVI_mean','EVI_mean']),('thermal','Adamawa_MOD11A2_day_night_*_v3.csv',['LST_day_C_mean','LST_night_C_mean'])]:
  sources=sorted((original/'inputs').glob(pattern));assert len(sources)==5,'Expected five original buffered VI/repaired LST files.'
  a=pd.concat([pd.read_csv(p) for p in sources]);a.ADM2_NAME=a.ADM2_NAME.replace({'Girie':'Girei','Teungo':'Toungo'})
  b=pd.concat([pd.read_csv(raw/f'Adamawa_cropland_{kind}_{year}_v2.csv') for year in YEARS]);b=b[b.mask_mode=='unmasked'];keys=['ADM2_NAME','crop_year','date']
  merged=a.merge(b,on=keys,suffixes=('_old','_new'),validate='one_to_one');assert len(merged)==len(a)==len(b)
  for band in bands:
   x=merged[band+'_old'].to_numpy(float);y=merged[band+'_new'].to_numpy(float);assert np.allclose(x,y,atol=1e-8,rtol=1e-10,equal_nan=True),'Unmasked control changed; review before model comparison.'
   controls.append(dict(product=kind,band=band,rows=len(merged),missing_changes=int((np.isnan(x)!=np.isnan(y)).sum()),maximum_absolute_difference=float(np.nanmax(abs(x-y)))))
 pd.DataFrame(controls).to_csv(root/'unmasked_control_reconciliation.csv',index=False)
 panels={};differences=[];missing=[]
 for mode in MODES:
  work=root/mode;inputs=work/'inputs';inputs.mkdir(parents=True,exist_ok=True)
  for f in (original/'inputs').glob('*.csv'):shutil.copy2(f,inputs/f.name)
  # Overlay replacement series under the filenames consumed by the unchanged builder.
  for year in YEARS:
   for kind,name in [('vegetation',f'Adamawa_MOD13Q1_buffer_{year}.csv'),('thermal',f'Adamawa_MOD11A2_day_night_{year}_v3.csv')]:
    df=pd.read_csv(raw/f'Adamawa_cropland_{kind}_{year}_v2.csv');df[df.mask_mode==mode].to_csv(inputs/name,index=False)
  shutil.copy2(original/'build_features.py',work/'build_features.py')
  print('Building',mode,'with unchanged temporal feature code',flush=True)
  subprocess.run([sys.executable,str(work/'build_features.py')],cwd=project,check=True)
  panel=pd.read_csv(work/'outputs/main_min5_features_soil_0_20.csv').sort_values(KEY).reset_index(drop=True)
  assert len(panel)==58 and panel.lga_name.nunique()==17 and not panel.duplicated(KEY).any()
  panels[mode]=panel
  for band in SAT:missing.append(dict(mask_mode=mode,feature_id=band,valid_rows=int(panel[band].notna().sum()),missing_rows=int(panel[band].isna().sum())))
 baseline=panels['unmasked']
 # Compare against accepted primary model inputs when present.
 primary=project/'src/tree_models/inputs/main_min5_full_candidates_0_20.csv'
 if not primary.is_file():raise FileNotFoundError('Accepted primary model input is required: '+str(primary))
 reference=pd.read_csv(primary).sort_values(KEY).reset_index(drop=True)
 assert baseline[KEY].equals(reference[KEY]);assert np.allclose(baseline.yield_kg_ha,reference.yield_kg_ha,rtol=0,atol=0)
 assert np.allclose(baseline[SAT+NON],reference[SAT+NON],atol=1e-8,rtol=1e-10,equal_nan=True),'Unmasked feature rebuild differs from primary candidates.'
 destinations=root/'model_inputs';destinations.mkdir(exist_ok=True)
 for mode,panel in panels.items():
  assert panel[KEY].equals(baseline[KEY]);assert np.allclose(panel.yield_kg_ha,baseline.yield_kg_ha,rtol=0,atol=0)
  assert np.allclose(panel[NON],baseline[NON],rtol=0,atol=0,equal_nan=True),'Non-satellite features changed.'
  panel.to_csv(destinations/f'{mode}_main_min5_full_candidates_0_20.csv',index=False)
  if mode!='unmasked':
   for band in SAT:
    valid=panel[band].notna()&baseline[band].notna();delta=panel.loc[valid,band]-baseline.loc[valid,band]
    differences.append(dict(mask_mode=mode,feature_id=band,matched_rows=int(valid.sum()),mean_difference=float(delta.mean()),mean_absolute_difference=float(abs(delta).mean()),maximum_absolute_difference=float(abs(delta).max())))
 pd.DataFrame(missing).to_csv(root/'satellite_missingness.csv',index=False);pd.DataFrame(differences).to_csv(root/'seasonal_feature_differences.csv',index=False)
 (root/'Review_Report.md').write_text('# Cropland seasonal feature rebuild\n\nPASS: raw unmasked controls and rebuilt candidates reproduce the accepted primary panel. All three configurations retain the same 58 target observations across 17 LGAs. Non-satellite candidate features and yields are unchanged. Missing satellite features remain missing; fold preprocessing has not been fitted. Seasonal feature differences do not constitute model results.\n\n'+pd.DataFrame(missing).to_string(index=False),encoding='utf-8')
 with zipfile.ZipFile(root/'Cropland_Feature_Review.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in root.glob('*'):
   if p.is_file() and p.suffix in ['.csv','.json','.md']:z.write(p,p.name)
  for p in destinations.glob('*.csv'):z.write(p,'model_inputs/'+p.name)
  for mode in MODES:
   for name in ['satellite_feature_audit.csv','smoothed_vegetation_audit.csv','input_hashes.json']:
    p=root/mode/'outputs'/name
    if p.exists():z.write(p,mode+'/'+name)
 print('PASS: identical targets/non-satellite features; unmasked control matches; three separate candidate panels. No model fitting yet.')
 print('Upload:',root/'Cropland_Feature_Review.zip')
if __name__=='__main__':main()
