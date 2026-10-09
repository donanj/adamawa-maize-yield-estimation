"""Target-quality sensitivity. Run from project root; never changes source/model inputs."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'src/target_quality/outputs'
RULES={'reference_99':(.99,None),'winsor_95':(.95,None),'exclude_gt7500':(.99,7500),'exclude_gt10000':(.99,10000),'exclude_gt15000':(.99,15000)}
KEY=['lga_name','crop_year']
def aggregate(d,q,minimum):
 x=d.copy();cap=x.groupby('wave').yield_raw.transform(lambda s:s.quantile(q));x['y']=np.minimum(x.yield_raw,cap)
 x['w']=x.area_ha*x.survey_weight_audit
 rows=[]
 for (wave,year,lga),g in x.groupby(['wave','year','ADM2_NAME']):
  if len(g)<minimum:continue
  rows.append(dict(wave=wave,crop_year=int(year),lga_name=lga,yield_kg_ha=float(np.average(g.y,weights=g.w)),eligible_plots=len(g),households=g.hhid.nunique(),clipped_plots=int((g.y!=g.yield_raw).sum())))
 return pd.DataFrame(rows),x

def prepare():
 audit=ROOT/'src/yield_panel/outputs/private_plot_audit.csv'
 if not audit.exists():raise FileNotFoundError('Run pipeline.py yield first to generate the private plot audit.')
 d=pd.read_csv(audit)
 required=['wave','year','ADM2_NAME','hhid','plotid','area_ha','survey_weight_audit','yield_raw','harvest_kg_audit','eligible','yield_clean','aggregation_weight','gps_ha','reported_ha']
 if not set(required).issubset(d):raise ValueError('Audit schema differs; review before proceeding.')
 assert d.eligible.eq(True).all() and not d.duplicated(['wave','hhid','plotid']).any()
 assert d.area_ha.gt(0).all() and d.survey_weight_audit.gt(0).all() and d.yield_raw.ge(0).all()
 assert np.isfinite(d.yield_raw).all() and np.allclose(d.yield_raw,d.harvest_kg_audit/d.area_ha)
 assert np.allclose(d.aggregation_weight,d.area_ha*d.survey_weight_audit)
 cap=d.groupby('wave').yield_raw.transform(lambda s:s.quantile(.99))
 assert np.allclose(d.yield_clean,np.minimum(d.yield_raw,cap)), 'Uploaded reference is not the specified upper q99 policy.'
 OUT.mkdir(parents=True,exist_ok=True)
 fingerprints={'audit':hashlib.sha256(audit.read_bytes()).hexdigest(),'source_code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 for panel in ['main_min5','all_cells']:
  for depth in ['0_20','20_50']:
   p=ROOT/'src/feature_engineering/outputs'/f'{panel}_features_soil_{depth}.csv'
   fingerprints[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 for name in ['linear_benchmarks/run_benchmarks.py','tree_models/run_models.py','tree_models/preprocessing.py']:
  fingerprints[name]=hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
 stamp=OUT/'protocol.json'
 protocol=dict(name='Post-audit exploratory target-quality sensitivity',rules={k:dict(upper_quantile=q,raw_plot_exclusion_above=cut) for k,(q,cut) in RULES.items()},primary_policy_changed=False,scope='same measurement/weight/crop eligibility as reference; no invented quantity/area corrections',threshold_status='diagnostic grid, not validated biological maxima',selection_rule='Do not select cleaning policy by model performance',fingerprints=fingerprints)
 if stamp.exists():
  if json.loads(stamp.read_text())!=protocol:raise ValueError('Audit/code/features changed. Preserve existing outputs and use a separate project copy.')
 stamp.write_text(json.dumps(protocol,indent=2))
 refs={};summaries=[];changes=[];thresholds=[];flags=[]
 for rule,(q,cut) in RULES.items():
  eligible=d if cut is None else d[d.yield_raw<=cut]
  for wave,g in eligible.groupby('wave'):thresholds.append(dict(rule=rule,wave=wave,eligible_plots=len(g),upper_quantile=q,upper_cap_kg_ha=g.yield_raw.quantile(q)))
  folder=OUT/'runs'/rule;folder.mkdir(parents=True,exist_ok=True)
  for panel,minimum in [('main_min5',5),('all_cells',1)]:
   y,plots=aggregate(eligible,q,minimum)
   if rule=='reference_99':
    original=pd.read_csv(ROOT/'src/feature_engineering/inputs'/f'{panel}.csv').rename(columns={'ADM2_NAME':'lga_name','year':'crop_year'})
    comp=y.merge(original[KEY+['yield_kg_ha']],on=KEY,suffixes=('_new','_original'),how='outer',validate='one_to_one',indicator=True)
    assert comp['_merge'].eq('both').all() and np.allclose(comp.yield_kg_ha_new,comp.yield_kg_ha_original)
    refs[panel]=y.copy()
   y.to_csv(folder/f'{panel}_yield_panel.csv',index=False)
   summaries.append(dict(rule=rule,panel=panel,cells=len(y),lgas=y.lga_name.nunique(),eligible_plots_before_cell_filter=len(eligible),plots_in_retained_cells=int(y.eligible_plots.sum()),clipped_plots_in_retained_cells=int(y.clipped_plots.sum()),median_yield=y.yield_kg_ha.median(),max_yield=y.yield_kg_ha.max()))
   c=refs[panel][KEY+['yield_kg_ha']].merge(y[KEY+['yield_kg_ha']],on=KEY,how='outer',suffixes=('_reference','_sensitivity'),indicator=True)
   c['rule']=rule;c['panel']=panel;c['change_kg_ha']=c.yield_kg_ha_sensitivity-c.yield_kg_ha_reference
   c['change_percent']=100*c.change_kg_ha/c.yield_kg_ha_reference.replace(0,np.nan);changes.append(c)
   for depth in ['0_20','20_50']:
    f=pd.read_csv(ROOT/'src/feature_engineering/outputs'/f'{panel}_features_soil_{depth}.csv')
    f=f[KEY+[f'V{i:02}' for i in range(1,34)]]
    joined=y.merge(f,on=KEY,how='left',validate='one_to_one')
    assert len(joined)==len(y) and joined.V13.notna().all()
    for model_folder in ['linear_benchmarks','tree_models']:
     dest=folder/model_folder/'inputs';dest.mkdir(parents=True,exist_ok=True)
     joined.to_csv(dest/f'{panel}_full_candidates_{depth}.csv',index=False)
  # Copy the reviewed model scripts into isolated per-rule run folders.
  lin=folder/'linear_benchmarks';s=(ROOT/'src/linear_benchmarks/run_benchmarks.py').read_text()
  s=s.replace("assert len(pr)==(58+63)*2*5","assert len(pr)==sum(len(pd.read_csv(ROOT/'inputs'/f'{p}_full_candidates_0_20.csv')) for p in ['main_min5','all_cells'])*2*5")
  (lin/'run_benchmarks.py').write_text(s)
  tree=folder/'tree_models'
  for name in ['run_models.py','preprocessing.py','predict_saved.py','verify_results.py','summarise_results.py']:
   s=(ROOT/'src/tree_models'/name).read_text()
   expected="sum(len(pd.read_csv(ROOT/'inputs'/f'{p}_full_candidates_0_20.csv')) for p in ['main_min5','all_cells'])*12"
   if name=='run_models.py':
    s=s.replace('(58+63)*12',expected).replace("print('PASS all 420 grouped tuning tasks and 1452 unique OOF predictions',flush=True)","print(f'PASS all {len(specs)} grouped tuning tasks and {len(pr)} unique OOF predictions',flush=True)")
   elif name=='verify_results.py':
    s=s.replace('len(pr)==1452',"len(pr)==sum(len(pd.read_csv(R/'inputs'/f'{p}_full_candidates_0_20.csv')) for p in ['main_min5','all_cells'])*12")
   (tree/name).write_text(s)
  for directory in ['outputs','models','studies']:(tree/directory).mkdir(exist_ok=True)
 summary=pd.DataFrame(summaries);summary.to_csv(OUT/'panel_summary.csv',index=False)
 pd.concat(changes,ignore_index=True).to_csv(OUT/'cell_changes.csv',index=False)
 pd.DataFrame(thresholds).to_csv(OUT/'wave_caps.csv',index=False)
 private=d.copy();private['above_10000_review_only']=private.yield_raw>10000
 ratio=private.gps_ha/private.reported_ha.where(private.reported_ha>0)
 private['gps_reported_ratio']=ratio;private['area_disagreement_review_only']=(ratio<.25)|(ratio>4)
 # The factor-four area flag is an analyst review aid only, never an exclusion rule.
 private.to_csv(OUT/'private_flag_register.csv',index=False)
 print(summary.round(2).to_string(index=False))
 print('PASS: reference labels reproduced; five policies prepared in separate folders; original model inputs/results untouched.')

def execute(stage,rule):
 if not rule:raise ValueError('Specify --rule explicitly; e.g. --rule winsor_95. No automatic policy selection.')
 selected=list(RULES) if rule=='all' else [rule]
 for name in selected:
  folder=OUT/'runs'/name
  if not folder.exists():raise FileNotFoundError('Run prepare first.')
  if stage=='linear':
   subprocess.run([sys.executable,str(folder/'linear_benchmarks/run_benchmarks.py')],check=True)
   dest=folder/'tree_models/linear_reference';dest.mkdir(exist_ok=True)
   for file in ['pooled_metrics.csv','oof_predictions.csv']:shutil.copy2(folder/'linear_benchmarks/outputs'/file,dest/file)
   shutil.copy2(folder/'linear_benchmarks/outputs/shared_LOLO_splits.csv',folder/'tree_models/shared_LOLO_splits.csv')
  elif stage=='trees':
   if not (folder/'tree_models/shared_LOLO_splits.csv').exists():raise ValueError('Run this rule\'s linear stage before trees.')
   subprocess.run([sys.executable,str(folder/'tree_models/run_models.py')],check=True)
  else:
   for file in ['verify_results.py','summarise_results.py']:subprocess.run([sys.executable,str(folder/'tree_models'/file)],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','linear','trees','verify-trees']);p.add_argument('--rule',choices=list(RULES)+['all']);args=p.parse_args()
 if args.stage=='prepare':prepare()
 else:execute(args.stage,args.rule)
