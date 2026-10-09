from pathlib import Path
import pandas as pd,json
R=Path(__file__).resolve().parent
m=pd.read_csv(R/'outputs/pooled_metrics.csv');l=pd.read_csv(R/'linear_reference/pooled_metrics.csv')
combined=pd.concat([l,m],ignore_index=True)
combined.to_csv(R/'outputs/combined_metrics.csv',index=False)
lines=['# Tree-model comparison results','', 'Completed nested LOLO-CV comparison. RMSE and MAE are kg/ha; R² is pooled over OOF predictions. These are descriptive performance estimates; uncertainty intervals and paired exploratory analyses remain outstanding.','']
for panel in ['main_min5','all_cells']:
 lines.extend([f'## {panel}: primary 0–20 cm soil','', '| Model | Tier | RMSE | MAE | R² |','|---|---|---:|---:|---:|'])
 for _,row in combined[(combined.panel==panel)&(combined.depth=='0_20')].sort_values(['model','tier']).iterrows():lines.append(f'| {row.model} | {int(row.tier)} | {row.RMSE:.1f} | {row.MAE:.1f} | {row.R2:.3f} |')
 lines.extend(['','## Soil depth sensitivity: '+panel,'','| Model | Full-tier RMSE 0–20 cm | Full-tier RMSE 20–50 cm |','|---|---:|---:|'])
 for model in ['MLR','Ridge','RF','XGBoost','CatBoost']:
  d=combined[(combined.panel==panel)&(combined.model==model)&(combined.tier==3)].set_index('depth')
  lines.append(f'| {model} | {d.loc["0_20","RMSE"]:.1f} | {d.loc["20_50","RMSE"]:.1f} |')
 lines.append('')
tr=pd.read_csv(R/'outputs/tuning_summary.csv')
lines.extend(['## Checks and interpretation','',f'All {len(tr)} tuning tasks completed. {int(tr.complete_trials.sum())} completed trials and {int(tr.pruned_trials.sum())} pruned trials, from 20 attempted trials per task. Median pruning reduces unnecessary fits; a pruned trial is not a fully evaluated parameter candidate.','', 'The accepted primary soil depth and both reference yield panels were retained. This comparison must not be used to choose a yield-cleaning policy. Added sources can alter performance in either direction in this small panel; these results do not demonstrate causal predictor effects. No tree-based feature pruning, held-out-LGA tuning, prediction clipping or target transformations were applied.','', 'Next step: fully nested stacking, which needs separate tuning inside each outer-training OOF generation split. Then complete LOYO sensitivity, uncertainty characterisation, paired exploratory comparisons and final-model SHAP.',''])
(R/'RESULTS.md').write_text('\n'.join(lines))
# Dominant squared-error contributions are recorded, without modifying labels.
p=pd.read_csv(R/'outputs/oof_predictions.csv');p['squared_error']=(p.observed-p.predicted)**2
p['squared_error_share']=p.squared_error/p.groupby(['panel','depth','tier','model']).squared_error.transform('sum')
p.sort_values('squared_error_share',ascending=False).groupby(['panel','depth','tier','model']).head(5).to_csv(R/'outputs/largest_error_contributions.csv',index=False)
print(m.to_string(index=False))
