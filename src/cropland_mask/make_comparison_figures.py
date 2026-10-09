"""Reproducible figures from completed cropland LOLO results; no model fitting."""
from pathlib import Path
import argparse,hashlib,json,platform,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
PROJECT=Path(__file__).resolve().parents[2]
MODES=['unmasked','cropland12','cropland12_14']
LABELS=['Unmasked (primary)','Cropland only','Cropland + mosaic']
COLORS=['#243746','#007C83','#B87524']
MODELS=['RF','XGBoost','CatBoost','Stacking']

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=PROJECT/'src/cropland_mask/outputs/lolo/research');p.add_argument('--output',type=Path,default=PROJECT/'src/cropland_mask/outputs/comparison_figures');a=p.parse_args()
 source=a.input;out=a.output;out.mkdir(parents=True,exist_ok=True)
 files=[source/n for n in ['matched_oof_predictions.csv','matched_pooled_metrics.csv','matched_per_LGA_metrics.csv']]
 pred,metrics,folds=[pd.read_csv(f) for f in files]
 assert len(pred)==2610 and len(metrics)==45 and len(folds)==765
 assert set(pred.mask_mode)==set(MODES) and set(pred.tier)=={1,2,3}
 assert not pred.duplicated(['mask_mode','tier','model','lga_name','crop_year']).any()
 reference=None
 for key,g in pred.groupby(['mask_mode','tier','model']):
  assert len(g)==58 and g.lga_name.nunique()==17
  target=g.sort_values(['lga_name','crop_year'])[['lga_name','crop_year','observed']].reset_index(drop=True)
  if reference is None:reference=target
  else:pd.testing.assert_frame_equal(reference,target)
  y=g.observed.to_numpy();q=g.predicted.to_numpy();e=y-q
  expected=dict(RMSE=np.sqrt(np.mean(e**2)),MAE=np.mean(abs(e)),R2=1-np.sum(e**2)/np.sum((y-y.mean())**2),RRMSE=100*np.sqrt(np.mean(e**2))/np.mean(y),MAPE=100*np.mean(abs(e/y)))
  row=metrics[(metrics.mask_mode==key[0])&(metrics.tier==key[1])&(metrics.model==key[2])];assert len(row)==1
  for name,v in expected.items():assert np.isclose(v,row.iloc[0][name],atol=1e-8)
  for lga,h in g.groupby('lga_name'):
   f=folds[(folds.mask_mode==key[0])&(folds.tier==key[1])&(folds.model==key[2])&(folds.lga_name==lga)];assert len(f)==1
   assert int(f.iloc[0]['n'])==len(h) and np.isclose(f.iloc[0].RMSE,np.sqrt(np.mean((h.observed-h.predicted)**2)))
 baseline=pred[pred.model.eq('mean')].sort_values(['mask_mode','tier','lga_name','crop_year'])
 for _,g in baseline.groupby(['mask_mode','tier']):
  for lga,h in g.groupby('lga_name'):
   expected=reference.loc[reference.lga_name.ne(lga),'observed'].mean();assert np.allclose(h.predicted,expected)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})
 def save(fig,name):
  fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 # Figure 1: small-multiple line plots; baseline is identical in all tiers/masks.
 fig,axes=plt.subplots(1,2,figsize=(9,3.9))
 for ax,measure,title in zip(axes,['RMSE','MAE'],['(a) Root mean squared error','(b) Mean absolute error']):
  for mode,label,color in zip(MODES,LABELS,COLORS):
   g=metrics[(metrics.model=='Stacking')&(metrics.mask_mode==mode)].sort_values('tier')
   ax.plot(g.tier,g[measure],marker='o',color=color,label=label,lw=1.8)
  value=metrics[(metrics.model=='mean')].iloc[0][measure]
  ax.axhline(value,color='#777777',ls='--',lw=1,label='Training-mean baseline')
  ax.set_xticks([1,2,3],['Tier 1','Tier 2','Tier 3']);ax.set_ylabel(measure+' (kg ha⁻¹)');ax.set_title(title)
  ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
 fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.03));fig.tight_layout(rect=[0,.12,1,1]);save(fig,'cropland_stacking_performance')
 # Figure 2: paired differences, identical model/tier comparator.
 delta=np.zeros((4,6));columns=[]
 for j,(mode,tier) in enumerate([(m,t) for m in MODES[1:] for t in [1,2,3]]):
  columns.append(('Crop only' if mode=='cropland12' else 'Crop + mosaic')+f'\nTier {tier}')
  for i,model in enumerate(MODELS):
   masked=metrics[(metrics.mask_mode==mode)&(metrics.tier==tier)&(metrics.model==model)].iloc[0].RMSE
   primary=metrics[(metrics.mask_mode=='unmasked')&(metrics.tier==tier)&(metrics.model==model)].iloc[0].RMSE
   delta[i,j]=masked-primary
 fig,ax=plt.subplots(figsize=(9,3.8));bound=max(abs(delta).max(),1)
 image=ax.imshow(delta,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
 ax.set_xticks(range(6),columns);ax.set_yticks(range(4),MODELS)
 for i in range(4):
  for j in range(6):ax.text(j,i,f'{delta[i,j]:+.1f}',ha='center',va='center',color='white' if abs(delta[i,j])>.65*bound else 'black',fontsize=10)
 ax.set_title('Masked minus unmasked RMSE: negative values favour masking',pad=12)
 fig.colorbar(image,ax=ax,label='ΔRMSE (kg ha⁻¹)',shrink=.9);fig.tight_layout();save(fig,'cropland_model_rmse_differences')
 pd.DataFrame(delta,index=MODELS,columns=columns).to_csv(out/'rmse_difference_plot_data.csv')
 # Figure 3: Tier 3 stacking LGA differences, not pooled significance.
 stack=folds[(folds.model=='Stacking')&(folds.tier==3)].pivot(index='lga_name',columns='mask_mode',values='RMSE').sort_index()
 lga_delta=stack[MODES[1:]].subtract(stack['unmasked'],axis=0)
 fig,ax=plt.subplots(figsize=(7.5,6.4));y=np.arange(len(stack))
 for mode,label,color,offset in zip(MODES[1:],LABELS[1:],COLORS[1:],[-.12,.12]):ax.scatter(lga_delta[mode],y+offset,color=color,label=label,s=28)
 ax.axvline(0,color='#555555',lw=1);ax.set_yticks(y,stack.index);ax.invert_yaxis();ax.set_xlabel('Masked minus unmasked fold RMSE (kg ha⁻¹)')
 ax.set_title('Tier 3 stacking: error changes by held-out LGA');ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='lower right');fig.tight_layout();save(fig,'cropland_tier3_lga_differences')
 lga_delta.to_csv(out/'tier3_lga_difference_plot_data.csv')
 captions='''# Cropland sensitivity figure captions

**cropland_stacking_performance:** Stacking ensemble pooled LOLO-CV RMSE and MAE across three predictor tiers under unmasked, cropland-only (class 12) and cropland-plus-mosaic (classes 12 and 14) extraction. All configurations use the same 58 observations from 17 LGAs. Dashed lines show the invariant training-fold mean baseline. Connecting lines guide comparison between the discrete tiers. No confidence intervals are shown.

**cropland_model_rmse_differences:** Differences in pooled LOLO-CV RMSE between each masked configuration and its unmasked counterpart for the same model and predictor tier. Negative values indicate lower error under masking. Base-learner predictions are reconstructed from the stacking final-stage fits. These descriptive differences are not significance tests.

**cropland_tier3_lga_differences:** Differences in held-out LGA RMSE for the Tier 3 stacking ensemble under each mask relative to unmasked extraction. Values below zero indicate improvement for that LGA. Each LGA has between one and five observations; pooled RMSE and the average of fold RMSE represent different summaries. No claim of independent fold-level inference is made.

Figure numbers will be assigned when inserted into Chapter 5. Annual land-cover masks identify broad cropland rather than maize specifically. Unmasked extraction remains the primary analysis.
'''
 (out/'Figure_Captions.md').write_text(captions,encoding='utf-8')
 manifest={'python':platform.python_version(),'packages':{'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},'inputs':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
 names=['cropland_stacking_performance','cropland_model_rmse_differences','cropland_tier3_lga_differences']
 with zipfile.ZipFile(out/'Cropland_Comparison_Figures.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in names:
   for ext in ['png','pdf']:z.write(out/f'{name}.{ext}',f'{name}.{ext}')
  for name in ['Figure_Captions.md','figure_manifest.json','rmse_difference_plot_data.csv','tier3_lga_difference_plot_data.csv']:z.write(out/name,name)
 print('PASS: all metrics, matched targets, LGA errors and mean baseline checked; three figures exported as PNG/PDF. No training performed.')
 print(out/'Cropland_Comparison_Figures.zip')
if __name__=='__main__':main()
