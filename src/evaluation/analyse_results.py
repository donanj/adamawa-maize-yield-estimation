"""Main-panel q99 fixed-OOF analysis. No model fitting or target changes."""
from pathlib import Path
import argparse,json,hashlib,platform,zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm,rankdata,wilcoxon
ROOT=Path(__file__).resolve().parents[2]
NAMES=['RMSE','MAE','R2','RRMSE','MAPE'];KEY=['lga_name','crop_year']
def summary(s):
 n,sy,sy2,se2,sa,npos,sape=s.T
 with np.errstate(divide='ignore',invalid='ignore'):
  rmse=np.sqrt(se2/n);sst=sy2-sy*sy/n
  return np.column_stack([rmse,sa/n,1-se2/sst,100*rmse/(sy/n),100*sape/npos])
def sufficient(d,lgas):
 rows=[]
 for name in lgas:
  p=d[d.lga_name==name];y=p.observed.to_numpy();e=y-p.predicted.to_numpy();positive=y>0
  rows.append([len(p),y.sum(),np.square(y).sum(),np.square(e).sum(),np.abs(e).sum(),positive.sum(),np.abs(e[positive]/y[positive]).sum()])
 return np.array(rows,float)
def bca(point,boot,jack):
 boot=np.asarray(boot);jack=np.asarray(jack);finite=boot[np.isfinite(boot)]
 if len(finite)!=len(boot):return np.nan,np.nan,'nonfinite bootstrap replicates; interval withheld'
 if not np.isfinite(jack).all() or not np.isfinite(point):return np.nan,np.nan,'nonfinite point/jackknife; interval withheld'
 diff=jack.mean()-jack;den=6*np.sum(diff**2)**1.5
 if den<=0 or np.ptp(boot)==0:return np.nan,np.nan,'degenerate distribution; interval withheld'
 acceleration=np.sum(diff**3)/den
 prop=(np.sum(boot<point)+.5*np.sum(boot==point))/len(boot)
 z0=norm.ppf(np.clip(prop,.5/len(boot),1-.5/len(boot)))
 z=norm.ppf([.025,.975]);denom=1-acceleration*(z0+z)
 if np.any(denom<=0):return np.nan,np.nan,'unstable BCa adjustment; interval withheld'
 probs=norm.cdf(z0+(z0+z)/denom)
 if not probs[0]<probs[1]:return np.nan,np.nan,'invalid BCa probabilities; interval withheld'
 lo,hi=np.quantile(boot,probs)
 return float(lo),float(hi),'conditional LGA-cluster BCa'
def holm(p):
 p=np.array(p,float);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 return out
def rank_effect(d):
 d=np.asarray(d,float)
 ranks=rankdata(np.where(d==0,np.nan,np.abs(d)),axis=-1,nan_policy='omit')
 ranks=np.nan_to_num(ranks);total=ranks.sum(axis=-1)
 return np.divide((ranks*np.sign(d)).sum(axis=-1),total,out=np.zeros_like(total),where=total>0)
def compare(label,a,b,folds,stats,boot_weights,jack_weights,indices):
 fa=folds[a];fb=folds[b];delta=np.round(fa-fb,10)
 if np.all(delta==0):w,p=0.,1.
 else:
  test=wilcoxon(delta,zero_method='wilcox',alternative='two-sided',method='approx',correction=False);w,p=float(test.statistic),float(test.pvalue)
 point=summary(stats[a].sum(axis=0)[None,:])[0,0]-summary(stats[b].sum(axis=0)[None,:])[0,0]
 boot=summary(boot_weights@stats[a])[:,0]-summary(boot_weights@stats[b])[:,0]
 jack=summary(jack_weights@stats[a])[:,0]-summary(jack_weights@stats[b])[:,0]
 lo,hi,status=bca(point,boot,jack)
 effect=float(rank_effect(delta));rb=rank_effect(delta[indices]);rj=rank_effect(np.array([np.delete(delta,k) for k in range(len(delta))]));rl,rh,rs=bca(effect,rb,rj)
 return dict(contrast=label,A=a,B=b,n_lgas=len(delta),pooled_RMSE_A_minus_B=point,delta_RMSE_BCa_low=lo,delta_RMSE_BCa_high=hi,delta_interval_status=status,mean_fold_RMSE_A_minus_B=float(delta.mean()),median_fold_RMSE_A_minus_B=float(np.median(delta)),A_better_folds=int((delta<0).sum()),B_better_folds=int((delta>0).sum()),ties=int((delta==0).sum()),wilcoxon_W=w,p_unadjusted=p,rank_biserial_favours_B=effect,r_BCa_low=rl,r_BCa_high=rh,r_interval_status=rs)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--resamples',type=int,default=9999);args=parser.parse_args()
 if args.resamples<999:parser.error('Use at least 999 resamples; research default is 9999.')
 paths={'linear':ROOT/'src/linear_benchmarks/outputs/oof_predictions.csv','trees':ROOT/'src/tree_models/outputs/oof_predictions.csv','stacking':ROOT/'src/stacking/outputs/research/reference_99/main_min5/0_20/oof_predictions.csv'}
 target=ROOT/'src/tree_models/inputs/main_min5_full_candidates_0_20.csv'
 reference=pd.read_csv(target)[KEY+['yield_kg_ha']].sort_values(KEY).reset_index(drop=True)
 assert len(reference)==58 and reference.lga_name.nunique()==17 and not reference.duplicated(KEY).any(), 'Expected the accepted main panel: 58 rows/17 LGAs.'
 selected=[]
 for source,path in paths.items():
  d=pd.read_csv(path);d=d[(d.panel=='main_min5')&(d.depth=='0_20')].copy();d['source']=source;selected.append(d)
 data=pd.concat(selected,ignore_index=True);data['configuration']=data.model+'_T'+data.tier.astype(int).astype(str)
 expected=['mean_T0','MLR_T1','MLR_T2','MLR_T3','Ridge_T3']+[m+'_T'+str(t) for m in ['RF','XGBoost','CatBoost','Stacking'] for t in [1,2,3]]
 assert set(data.configuration)==set(expected),'Missing/unexpected configurations. Ensure stacking summary includes all three tiers.'
 configs={}
 for label,d in data.groupby('configuration'):
  d=d.sort_values(KEY).reset_index(drop=True);assert len(d)==58 and not d.duplicated(KEY).any()
  assert d[KEY].equals(reference[KEY]) and np.allclose(d.observed,reference.yield_kg_ha,rtol=1e-12,atol=1e-8)
  assert np.isfinite(d[['observed','predicted']].to_numpy()).all();configs[label]=d
 lgas=sorted(reference.lga_name.unique());n=len(lgas);rng=np.random.default_rng(20261007)
 indices=rng.integers(0,n,size=(args.resamples,n));weights=np.array([np.bincount(row,minlength=n) for row in indices]);jack=np.ones((n,n))-np.eye(n)
 stats={k:sufficient(d,lgas) for k,d in configs.items()};folds={k:np.sqrt(s[:,3]/s[:,0]) for k,s in stats.items()}
 for k,d in configs.items():
  y=d.observed.to_numpy();e=y-d.predicted.to_numpy();direct=[np.sqrt(np.mean(e**2)),np.mean(abs(e)),1-np.sum(e**2)/np.sum((y-y.mean())**2),100*np.sqrt(np.mean(e**2))/y.mean(),100*np.mean(abs(e[y>0]/y[y>0]))]
  assert np.allclose(summary(stats[k].sum(axis=0)[None,:])[0],direct)
 out=ROOT/'src/evaluation/outputs/main_min5_reference_99_0_20';out.mkdir(parents=True,exist_ok=True)
 manifest=dict(seed=20261007,resamples=args.resamples,clusters=n,rows=58,bootstrap='resample 17 LGAs with replacement; retain all years; fixed OOF predictions; no refitting',holm_family='12 planned exploratory contrasts',packages=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(paths.values())+[target,Path(__file__)]})
 stamp=out/'analysis_manifest.json'
 if stamp.exists() and json.loads(stamp.read_text())!=manifest:raise ValueError('Source/results/settings changed. Preserve prior evaluation outputs before a new run.')
 rows=[]
 for k in expected:
  s=stats[k];point=summary(s.sum(axis=0)[None,:])[0];boots=summary(weights@s);jacks=summary(jack@s);row=dict(configuration=k,model=configs[k].model.iloc[0],tier=int(configs[k].tier.iloc[0]),rows=58,LGAs=17)
  for j,metric in enumerate(NAMES):
   lo,hi,status=bca(point[j],boots[:,j],jacks[:,j]);row.update({metric:point[j],metric+'_BCa_low':lo,metric+'_BCa_high':hi,metric+'_CI_status':status})
  row['RMSE_gain_vs_mean_percent']=100*(summary(stats['mean_T0'].sum(axis=0)[None,:])[0,0]-point[0])/summary(stats['mean_T0'].sum(axis=0)[None,:])[0,0];rows.append(row)
 table=pd.DataFrame(rows).sort_values('RMSE');table.to_csv(out/'model_comparison.csv',index=False)
 contrasts=[]
 for a,b in [('RF','XGBoost'),('RF','CatBoost'),('XGBoost','CatBoost'),('RF','Stacking'),('XGBoost','Stacking'),('CatBoost','Stacking')]:contrasts.append(compare(a+' vs '+b+' (Tier 3)',a+'_T3',b+'_T3',folds,stats,weights,jack,indices))
 for model in ['RF','XGBoost','CatBoost']:
  for a,b in [(1,2),(2,3)]:contrasts.append(compare(f'{model}: Tier {a} vs Tier {b}',f'{model}_T{a}',f'{model}_T{b}',folds,stats,weights,jack,indices))
 tests=pd.DataFrame(contrasts);tests['p_Holm_12']=holm(tests.p_unadjusted);tests['exploratory_p_below_0_05']=tests.p_Holm_12<.05;tests.to_csv(out/'planned_exploratory_comparisons.csv',index=False)
 # Post-results questions kept outside the planned testing family: descriptive intervals only.
 extra=[compare('Stacking Tier 3 vs satellite-only MLR','Stacking_T3','MLR_T1',folds,stats,weights,jack,indices),compare('Stacking Tier 3 vs mean baseline','Stacking_T3','mean_T0',folds,stats,weights,jack,indices)]
 descriptive=pd.DataFrame(extra).drop(columns=['wilcoxon_W','p_unadjusted']);descriptive.to_csv(out/'post_results_descriptive_contrasts.csv',index=False)
 errors=[];foldrows=[];influence=[]
 for k,d in configs.items():
  d=d.copy();d['residual_observed_minus_predicted']=d.observed-d.predicted;d['absolute_error']=abs(d.residual_observed_minus_predicted);d['squared_error']=d.residual_observed_minus_predicted**2
  d['squared_error_share']=d.squared_error/d.squared_error.sum();d=d.sort_values('squared_error',ascending=False);d['error_rank']=np.arange(1,len(d)+1);d['cumulative_squared_error_share']=d.squared_error_share.cumsum();errors.append(d)
  for name,rmse in zip(lgas,folds[k]):foldrows.append(dict(configuration=k,lga_name=name,n=int(stats[k][lgas.index(name),0]),RMSE=float(rmse)))
  for i,name in enumerate(lgas):influence.append(dict(configuration=k,omitted_lga=name,fixed_prediction_RMSE_without_lga=float(summary((stats[k].sum(axis=0)-stats[k][i])[None,:])[0,0]),note='descriptive omission only; no model refit or change to reported evaluation'))
 pd.concat(errors,ignore_index=True).to_csv(out/'observation_error_diagnostics.csv',index=False);pd.DataFrame(foldrows).to_csv(out/'per_LGA_metrics.csv',index=False);pd.DataFrame(influence).to_csv(out/'LGA_error_influence.csv',index=False)
 top=pd.concat(errors,ignore_index=True);top=top[top.configuration=='Stacking_T3'].sort_values('error_rank').head(10);top.to_csv(out/'stacking_T3_top_errors.csv',index=False)
 report=['# Main-panel q99 evaluation','', '58 LGA-year observations; 17 LGAs; 0–20 cm soil depth. All 17 configurations share the same keys and observed targets. No models were retrained.','', '## Pooled performance','', '| Configuration | RMSE | MAE | R² | 95% conditional RMSE BCa interval |','|---|---:|---:|---:|---|']
 for _,r in table.iterrows():report.append(f"| {r.configuration} | {r.RMSE:.2f} | {r.MAE:.2f} | {r.R2:.4f} | [{r.RMSE_BCa_low:.2f}, {r.RMSE_BCa_high:.2f}] |")
 report+=['','## Interpretation limits','', 'The BCa bootstrap resamples LGAs as clusters, retaining all their years and resampling the same clusters for paired contrasts. It holds labels and fitted OOF predictions fixed. It describes sensitivity to the composition of the observed LGA clusters; it is not a prediction interval or a full sampling distribution of model training. Overlapping LOLO training sets induce dependence across folds that cluster resampling does not remove. With only 17 clusters, nominal 95% coverage is not assured. Degenerate/unstable intervals are withheld and flagged in CSVs.','', 'Wilcoxon tests use 17 paired LGA RMSEs, a two-sided normal approximation with zero differences omitted and differences rounded to 10 decimals for stable ranks. All 12 planned tests share a Holm correction. Their independence assumption remains compromised by overlapping training sets; adjusted p-values are exploratory, not confirmatory evidence. Positive rank-biserial values favour B; negative values favour A. Pooled RMSE difference A−B is negative when A performs better. Fold-rank effect intervals inherit the same dependence limitation.','', 'Stacking versus MLR and the mean baseline were raised after seeing results; their separate table contains descriptive paired intervals and no hypothesis-test p-values. No new comparisons have been silently added to the planned family.','', 'Observation error shares and omission diagnostics explain concentration of error. They are not grounds to delete observations or change target policies. MAPE uses only positive observed targets and can be sensitive to small denominators.','', 'See planned_exploratory_comparisons.csv for all test results and interval status flags. No automated claim of model superiority is made.']
 (out/'Review_Report.md').write_text('\n'.join(report),encoding='utf-8');stamp.write_text(json.dumps(manifest,indent=2))
 package=out/'Main_Panel_Evaluation_Review.zip'
 with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(out.iterdir()):
   if f.suffix in ['.csv','.json','.md']:z.write(f,f.name)
 print('PASS: 17 configurations, 58 identical target rows each, 17 aligned LGA folds; metrics reconstructed; 12-test Holm family; paired cluster resampling. No model refitting.')
 print(table[['configuration']+NAMES+['RMSE_BCa_low','RMSE_BCa_high']].to_string(index=False));print('\nExploratory comparisons:');print(tests[['contrast','pooled_RMSE_A_minus_B','p_Holm_12','rank_biserial_favours_B']].to_string(index=False));print('\nStacking Tier 3 largest errors:');print(top[KEY+['observed','predicted','squared_error_share']].to_string(index=False));print(f'\nShare this review ZIP: {package}')
if __name__=='__main__':main()
