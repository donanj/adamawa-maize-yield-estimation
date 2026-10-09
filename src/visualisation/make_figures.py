"""Six thesis figures from accepted main-panel saved results; no model fitting."""
from pathlib import Path
import argparse,hashlib,json,platform,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter,MaxNLocator
ROOT=Path(__file__).resolve().parents[2]
COLORS={'mean':'#666666','MLR':'#8258A5','Ridge':'#CC79A7','RF':'#009E73','XGBoost':'#E69F00','CatBoost':'#D55E00','Stacking':'#0072B2'}
MODELS=['MLR','RF','XGBoost','CatBoost','Stacking']
MARKERS={'MLR':'s','RF':'o','XGBoost':'^','CatBoost':'D','Stacking':'P'}
TIERS={1:'Satellite',2:'Satellite + climate',3:'Satellite + climate\n+ soil/terrain'}
UNIT=r'RMSE (kg ha$^{-1}$)'
KEY=['lga_name','crop_year']
def label(model,tier):return 'Mean baseline' if model=='mean' else f'{model} · T{int(tier)}'
def style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlepad':12,'axes.axisbelow':True,'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
def polish(ax,grid='x'):
 ax.grid(axis=grid,color='#E6E6E6',linewidth=.7);ax.tick_params(length=3)
 if grid=='x':ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
 else:ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
def export(fig,name,source,out,dpi):
 fig.savefig(out/f'{name}.png',dpi=dpi,bbox_inches='tight',pad_inches=.12)
 fig.savefig(out/f'{name}.pdf',bbox_inches='tight',pad_inches=.12)
 source.to_csv(out/f'{name}_source.csv',index=False);plt.close(fig);print(f'Saved {name}: PNG, PDF and source CSV',flush=True)
def f1(m,out,dpi):
 d=m.sort_values('RMSE').reset_index(drop=True);fig,ax=plt.subplots(figsize=(7.4,8.6),layout='constrained')
 for i,r in d.iterrows():
  c=COLORS[r.model];lo=r.RMSE_BCa_low;hi=r.RMSE_BCa_high
  if np.isfinite(lo) and np.isfinite(hi):ax.hlines(i,lo,hi,color=c,linewidth=1.7,alpha=.8);ax.plot([lo,hi],[i,i],'|',color=c,markersize=7)
  ax.plot(r.RMSE,i,'o',color=c,markersize=6)
 ax.set_yticks(range(len(d)),[label(r.model,r.tier) for _,r in d.iterrows()]);ax.invert_yaxis();ax.set_xlabel(UNIT);ax.set_xlim(left=0);polish(ax)
 ax.set_title('Spatial validation: model performance\nLOLO-CV · 58 observations · 17 LGAs',loc='left')
 ax.axvline(float(d[d.model=='mean'].RMSE.iloc[0]),color=COLORS['mean'],ls='--',lw=1)
 fig.legend(handles=[Line2D([0],[0],color='#555555',marker='o',label='Point estimate + conditional 95% BCa interval'),Line2D([0],[0],color=COLORS['mean'],ls='--',label='Mean-baseline RMSE')],loc='outside lower center',fontsize=8,frameon=False)
 export(fig,'01_model_rmse_intervals',d,out,dpi)
def f2(m,out,dpi):
 d=m[m.model.isin(MODELS)].copy();baseline=float(m[m.model=='mean'].RMSE.iloc[0]);fig,ax=plt.subplots(figsize=(7.4,5.2),layout='constrained')
 for name in MODELS:
  p=d[d.model==name].sort_values('tier');ax.plot(p.tier,p.RMSE,color=COLORS[name],marker=MARKERS[name],lw=1.8,markersize=6,label=name,ls='--' if name=='MLR' else '-')
 ax.axhline(baseline,color=COLORS['mean'],ls=':',lw=1.5,label='Mean baseline');ax.set_xticks([1,2,3],[TIERS[i] for i in [1,2,3]]);ax.set_xlim(.85,3.15)
 ax.set_ylim(0,max(d.RMSE.max(),baseline)*1.12);ax.set_ylabel(UNIT);ax.set_title('Predictor enrichment under spatial validation',loc='left');polish(ax,'y');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,fontsize=9)
 d['mean_baseline_RMSE']=baseline;export(fig,'02_predictor_tiers',d,out,dpi)
def f3(e,out,dpi):
 keys=['Stacking_T3','MLR_T1','mean_T0'];d=e[e.configuration.isin(keys)].copy();years=sorted(d.crop_year.unique());yearcolors=dict(zip(years,plt.cm.viridis(np.linspace(.08,.9,len(years)))))
 fig,axes=plt.subplots(1,3,figsize=(10.6,5.0),sharex=True,sharey=True);fig.subplots_adjust(left=.08,right=.99,top=.78,bottom=.23,wspace=.16);low=min(0,float(d.predicted.min()),float(d.observed.min()));high=max(float(d.predicted.max()),float(d.observed.max()))*1.12
 for ax,key in zip(axes,keys):
  p=d[d.configuration==key];rmse=np.sqrt(np.mean((p.observed-p.predicted)**2))
  for year in years:
   s=p[p.crop_year==year];ax.scatter(s.observed,s.predicted,s=30,color=yearcolors[year],alpha=.85,edgecolors='white',linewidths=.4)
  ax.plot([low,high],[low,high],color='#555555',ls='--',lw=1);ax.set_xlim(low,high);ax.set_ylim(low,high);ax.set_aspect('equal',adjustable='box');ax.set_title(label(p.model.iloc[0],p.tier.iloc[0])+f'\nRMSE = {rmse:,.0f}',fontsize=11);ax.set_xlabel(r'Observed yield (kg ha$^{-1}$)');polish(ax);ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
  if key=='Stacking_T3':
   for _,r in p.nlargest(3,'squared_error').iterrows():ax.annotate(f'{r.lga_name}\n{int(r.crop_year)}',(r.observed,r.predicted),xytext=(0,10),textcoords='offset points',fontsize=7,ha='center')
 axes[0].set_ylabel(r'Predicted yield (kg ha$^{-1}$)');fig.suptitle('Observed and held-out predictions: LOLO-CV',fontsize=13,y=.98)
 fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color=yearcolors[y],label=str(y)) for y in years],loc='lower center',bbox_to_anchor=(.5,.015),ncol=5,title='Crop year',frameon=False,fontsize=9)
 export(fig,'03_observed_vs_predicted',d,out,dpi)
def f4(e,out,dpi):
 d=e[e.configuration=='Stacking_T3'].sort_values('squared_error',ascending=False).copy();d['rank']=np.arange(1,len(d)+1);d['share_percent']=100*d.squared_error/d.squared_error.sum();d['cumulative_percent']=d.share_percent.cumsum();top=d.head(10);total=float(d.head(3).share_percent.sum())
 fig,(a,b)=plt.subplots(1,2,figsize=(10.5,5.2),layout='constrained');a.barh(np.arange(len(top)),top.share_percent,color=[COLORS['Stacking'] if i<3 else '#8FAFC1' for i in range(len(top))]);a.set_yticks(np.arange(len(top)),[f'{r.lga_name} · {int(r.crop_year)}' for _,r in top.iterrows()]);a.invert_yaxis();a.set_xlim(0,max(45,float(top.share_percent.max())*1.2))
 for i,(_,r) in enumerate(top.iterrows()):a.text(r.share_percent+.6,i,f'{r.share_percent:.1f}%',va='center',fontsize=9)
 a.set_xlabel('Share of total squared error (%)');a.set_title('(a) Ten largest observation errors',loc='left');polish(a)
 b.plot(d['rank'],d.cumulative_percent,color=COLORS['Stacking'],lw=2);b.scatter([3],[total],color=COLORS['CatBoost'],s=45,zorder=3);b.annotate(f'First 3 observations:\n{total:.1f}% of squared error',(3,total),xytext=(17,53),arrowprops=dict(arrowstyle='->',color='#555555'),fontsize=10)
 b.set_ylim(0,103);b.set_xlim(1,len(d));b.set_xlabel('Observations ranked by squared error');b.set_ylabel('Cumulative squared error (%)');b.set_title('(b) Error concentration across all 58 rows',loc='left');polish(b,'y');fig.suptitle('Tier 3 stacking: concentration of LOLO prediction error',fontsize=13)
 export(fig,'04_error_concentration',d,out,dpi)
def f5(c,out,dpi):
 d=c.sort_values('LOLO_RMSE').reset_index(drop=True);fig,ax=plt.subplots(figsize=(7.4,8.6),layout='constrained')
 for i,r in d.iterrows():
  ax.plot([r.LOLO_RMSE,r.LOYO_RMSE],[i,i],color='#B4B4B4',lw=1.5);ax.scatter(r.LOLO_RMSE,i,color='#0072B2',marker='o',s=35,zorder=3);ax.scatter(r.LOYO_RMSE,i,color='#D55E00',marker='D',s=32,zorder=3)
 ax.set_yticks(range(len(d)),[label(r.model,r.tier) for _,r in d.iterrows()]);ax.invert_yaxis();ax.set_xlim(left=0);ax.set_xlabel(UNIT);polish(ax);ax.set_title('Spatial and temporal validation\nSame 58 observations · different holdout designs',loc='left');fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color='#0072B2',label='LOLO: held-out LGA'),Line2D([0],[0],marker='D',ls='',color='#D55E00',label='LOYO: held-out year')],loc='outside lower center',ncol=2,frameon=False,fontsize=9)
 export(fig,'05_lolo_vs_loyo',d,out,dpi)
def f6(y,out,dpi):
 specs=[('Stacking',3),('MLR',1),('RF',3),('mean',0)];parts=[];fig,ax=plt.subplots(figsize=(7.4,5.2),layout='constrained');years=sorted(y.held_year.unique());x=np.arange(len(years));width=.18
 for j,(name,tier) in enumerate(specs):
  p=y[(y.model==name)&(y.tier==tier)].sort_values('held_year').copy();assert list(p.held_year)==years;parts.append(p)
  ax.bar(x+(j-1.5)*width,p.RMSE,width=width,color=COLORS[name],label=label(name,tier),edgecolor='white',linewidth=.4,hatch='//' if name=='mean' else None)
 counts=y[(y.model=='mean')&(y.tier==0)].sort_values('held_year');ax.set_xticks(x,[f'{year}\n(n={int(n)})' for year,n in zip(years,counts.n)]);ax.set_ylabel(UNIT);ax.set_xlabel('Held-out crop year');ax.set_ylim(bottom=0);polish(ax,'y');ax.set_title('Temporal validation: performance by held-out year',loc='left');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False,fontsize=9)
 export(fig,'06_held_out_years',pd.concat(parts,ignore_index=True),out,dpi)
def load_and_check():
 evaldir=ROOT/'src/evaluation/outputs/main_min5_reference_99_0_20';temporal=ROOT/'src/temporal_validation/outputs/research'
 paths={'comparison':evaldir/'model_comparison.csv','errors':evaldir/'observation_error_diagnostics.csv','temporal_comparison':temporal/'LOLO_vs_LOYO.csv','years':temporal/'per_year_metrics.csv','temporal_predictions':temporal/'oof_predictions.csv'}
 for path in paths.values():
  if not path.exists():raise FileNotFoundError(f'Missing result file: {path}. Complete evaluation/LOYO verification first.')
 m=pd.read_csv(paths['comparison']);e=pd.read_csv(paths['errors']);c=pd.read_csv(paths['temporal_comparison']);y=pd.read_csv(paths['years']);t=pd.read_csv(paths['temporal_predictions'])
 assert len(m)==17 and not m.duplicated(['model','tier']).any() and len(c)==17 and not c.duplicated(['model','tier']).any()
 assert len(e)==986 and len(t)==986 and set(e.panel)=={'main_min5'} and set(t.panel)=={'main_min5'} and set(e.depth)=={'0_20'} and set(t.depth)=={'0_20'}
 configurations=set(zip(m.model,m.tier));assert configurations==set(zip(c.model,c.tier))==set(zip(y.model,y.tier))==set(zip(t.model,t.tier))
 reference=e[e.configuration=='Stacking_T3'].sort_values(KEY).reset_index(drop=True)
 assert len(reference)==58 and reference.lga_name.nunique()==17
 for model,tier in configurations:
  a=e[(e.model==model)&(e.tier==tier)].sort_values(KEY).reset_index(drop=True);b=t[(t.model==model)&(t.tier==tier)].sort_values(KEY).reset_index(drop=True)
  assert len(a)==len(b)==58 and not a.duplicated(KEY).any() and not b.duplicated(KEY).any()
  assert a[KEY].equals(reference[KEY]) and b[KEY].equals(reference[KEY]) and np.allclose(a.observed,reference.observed) and np.allclose(b.observed,reference.observed)
  assert np.isfinite(a[['observed','predicted']]).all().all() and np.isfinite(b[['observed','predicted']]).all().all()
  rmse=lambda d:float(np.sqrt(np.mean((d.observed-d.predicted)**2)))
  point=m[(m.model==model)&(m.tier==tier)].iloc[0];cross=c[(c.model==model)&(c.tier==tier)].iloc[0]
  assert np.isclose(point.RMSE,rmse(a)) and np.isclose(cross.LOLO_RMSE,rmse(a)) and np.isclose(cross.LOYO_RMSE,rmse(b))
  assert np.allclose(a.squared_error,(a.observed-a.predicted)**2) and np.isclose(a.squared_error_share.sum(),1)
  per=y[(y.model==model)&(y.tier==tier)];assert len(per)==5 and set(per.held_year)=={2010,2012,2015,2018,2023}
  for year,p in b.groupby('crop_year'):
   r=per[per.held_year==year].iloc[0];assert int(r.n)==len(p) and np.isclose(r.RMSE,rmse(p))
 return m,e,c,y,paths
CAPTIONS={
'01_model_rmse_intervals': 'Pooled LOLO-CV RMSE for 17 model configurations using the q99 main panel (58 LGA-year observations across 17 LGAs). Points show RMSE and horizontal lines show conditional 95% LGA-cluster BCa intervals based on 9,999 resamples of fixed out-of-fold predictions. The dashed line marks the training-set mean baseline. T1: satellite; T2: satellite plus climate; T3: satellite plus climate, soil and terrain. Ranking is descriptive. Intervals do not account for model refitting or resolve dependence from overlapping training sets; they are not individual prediction intervals.',
'02_predictor_tiers': 'Pooled LOLO-CV RMSE across progressively enriched predictor configurations for MLR, RF, XGBoost, CatBoost and stacking. The horizontal dotted line is the mean baseline. Lines connect discrete configurations and do not represent continuous predictor increments. These are point estimates; improvements are descriptive and do not establish statistically reliable tier contributions.',
'03_observed_vs_predicted': 'Observed and LOLO-held-out predicted maize yield for Tier 3 stacking, satellite-only MLR and the training-set mean baseline. Each panel contains the same 58 observations; colours identify crop years. The dashed line represents equality, and identical axes permit comparison. The three largest squared-error records are labelled in the stacking panel. All records are retained; no axis truncation conceals extreme yields.',
'04_error_concentration': 'Squared-error concentration for Tier 3 stacking under LOLO-CV. Panel (a) shows individual shares for the ten largest errors. Panel (b) shows cumulative shares over all 58 observations ordered by squared error. Shares describe contributions to the aggregate error and do not justify observation deletion or target correction.',
'05_lolo_vs_loyo': 'Pooled RMSE under LGA-held-out (LOLO) and year-held-out (LOYO) validation for matching configurations on the same 58-row q99 main panel. Connecting segments pair configurations across designs; they are not uncertainty intervals. Differences reflect changes in generalisation target, training composition and inner tuning scheme. LOYO is not a forward-only forecasting evaluation.',
'06_held_out_years': 'Year-specific LOYO-CV RMSE for Tier 3 stacking, satellite-only MLR, Tier 3 RF and the mean baseline. These configurations are highlighted for interpretation of the earlier spatial comparison; they are not asserted to be the strongest temporal models. Sample counts refer to the observations in each held-out crop year. Results are descriptive; no confidence intervals or year-level hypothesis tests are implied.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','run'],default='run',nargs='?');p.add_argument('--dpi',type=int,default=400);args=p.parse_args()
 if args.dpi<300:p.error('Use at least 300 dpi for thesis PNG exports.')
 m,e,c,y,paths=load_and_check();print(f'PASS: 17 matched configurations; 58 unchanged targets each; LOLO/LOYO metric and year-count checks. Matplotlib {matplotlib.__version__}.',flush=True)
 if args.stage=='check':return
 style();out=ROOT/'src/visualisation/outputs/main_panel_q99';out.mkdir(parents=True,exist_ok=True)
 manifest=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,dpi=args.dpi,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths.values()},scope='main_min5 q99 0–20cm; descriptive saved results; no refitting')
 for function,source in [(f1,m),(f2,m),(f3,e),(f4,e),(f5,c),(f6,y)]:function(source,out,args.dpi)
 captions=['# Figure captions and use','', 'Filenames are identifiers, not final thesis figure numbers. Assign final numbers after placing them in Chapter 5. Use selected figures by reference in Chapter 6; avoid repeating every plot in Chapters 6/7. Chapter 7 can use a synthesis table once conclusions are final.','']
 for name,caption in CAPTIONS.items():captions.extend([f'## {name}','',caption,''])
 (out/'Figure_Captions.md').write_text('\n'.join(captions),encoding='utf-8');(out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
 files=[out/(name+ext) for name in CAPTIONS for ext in ['.png','.pdf','_source.csv']]+[out/'Figure_Captions.md',out/'figure_manifest.json']
 for path in files:assert path.exists() and path.stat().st_size>0
 package=out/'Thesis_Main_Panel_Figures.zip'
 with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
  for path in files:z.write(path,path.name)
 print(f'PASS: six figures exported as {args.dpi}-dpi PNG and vector PDF, with source tables/captions.\nShare for review: {package}',flush=True)
if __name__=='__main__':main()
