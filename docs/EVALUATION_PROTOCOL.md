# Main-panel evaluation: saved q99 predictions

Extract into the existing adamawa-maize-yield project root. Run:

```powershell
.venv\Scripts\python.exe src\evaluation\analyse_results.py
```

No additional packages are needed: NumPy, pandas and SciPy are already in the project environment. The script does not train models or change targets, predictors, preprocessing or existing results. It reads the original linear and standalone tree OOF CSVs and the completed reference stacking OOF CSV. It analyses only main_min5 and 0–20 cm: 58 LGA-year observations across 17 LGAs.

The expected 17 configurations are mean once, MLR under three tiers, Ridge under Tier 3, and RF/XGBoost/CatBoost/Stacking under three tiers each. The script rejects missing configurations, duplicate keys, unmatched observations, changed target values or nonfinite predictions. If stacking summaries were overwritten by a restricted --tier command, first run stacking verify with the default flags to reconstruct the all-tier summary.

## Outputs

src/evaluation/outputs/main_min5_reference_99_0_20/

- model_comparison.csv: pooled RMSE, MAE, R², RRMSE, MAPE and conditional 95% BCa intervals for all 17 configurations.
- planned_exploratory_comparisons.csv: six Tier-3 tree/stacking model contrasts and six adjacent tier contrasts for RF, XGBoost and CatBoost; Holm correction across the single 12-test family. Includes paired pooled RMSE differences and rank-biserial effects with conditional BCa intervals.
- post_results_descriptive_contrasts.csv: stacking Tier 3 versus satellite-only MLR and versus the mean baseline. These questions arose after inspecting results; descriptive paired intervals are supplied without hypothesis-test p-values.
- per_LGA_metrics.csv: aligned RMSE for the 17 held-out LGAs per configuration.
- observation_error_diagnostics.csv: observed/predicted yield, residuals, absolute error, squared-error share and cumulative error share for every configuration.
- stacking_T3_top_errors.csv: the ten largest squared errors of Tier-3 stacking.
- LGA_error_influence.csv: pooled RMSE after omitting each LGA's scored observations, with predictions held fixed. This is an error-concentration diagnostic, not retrained leave-one-out model performance.
- Review_Report.md: readable summary and interpretation limits; open in VS Code and use Ctrl+Shift+V for Markdown preview.
- analysis_manifest.json: sources and input hashes, software versions and resampling settings.
- Main_Panel_Evaluation_Review.zip: bundles the above outputs for review. Contains LGA/year aggregates and predictions, not household or plot identifiers.

Share Main_Panel_Evaluation_Review.zip when the command completes. This ZIP supplies the information needed to interpret uncertainty and error concentration; terminal point estimates alone cannot establish those findings.

## Statistical choices and limits

Seed 20261007; 9,999 bootstrap replicates. Sample 17 LGA clusters with replacement, retaining all years of each selected LGA. Use identical cluster draws for every model and paired contrast. Resampled pooled metrics weight each LGA-year equally; cluster sizes are preserved. BCa acceleration uses leave-one-LGA-out jackknife statistics on the fixed predictions. Intervals are calculated for pooled metrics, paired pooled RMSE differences and paired rank-biserial effects. Degenerate/nonfinite/unstable intervals are withheld and flagged, never silently replaced with percentile intervals.

This resampling conditions on existing targets and fitted OOF predictions. It describes sensitivity to observed LGA composition, not training uncertainty, individual prediction uncertainty or demonstrated future-season performance. It does not remove dependence induced by overlapping LOLO training sets. With only 17 clusters, nominal 95% coverage is not guaranteed; the intervals are conditional uncertainty summaries.

Wilcoxon comparisons use two-sided normal approximation (zero_method=wilcox, no continuity correction), omitting zero differences. Paired RMSE differences are rounded to ten decimals for stable rank ties. Rank-biserial is (positive rank sum − negative rank sum)/total rank sum, where differences are A−B. Thus positive values favour B; negative values favour A. An all-zero contrast receives p=1 and effect=0; a degenerate effect interval is withheld. Holm adjustment spans all 12 previously specified comparisons. Overlapping training sets compromise the usual independence assumption, so adjusted p-values remain exploratory. A small adjusted p-value is not confirmatory proof of superiority.

Pooled RMSE difference A−B is negative when A is better; it need not have the same sign as the rank effect or average per-LGA difference because pooled RMSE and fold ranks have different weighting. Interval overlap between individual model estimates is not a paired comparison test. Point rankings across configurations are descriptive; this script does not certify a selected winner using an independent test dataset.

Large-error records remain in the reference evaluation. Diagnostics do not authorise deleting them, modifying harvest quantities or choosing cleaning rules for better scores. MAPE excludes zero targets and may be sensitive to small positive targets. The script does not run new target sensitivity, temporal validation, SHAP, or a full training-resampling bootstrap.

## Verification completed for this patch

Synthetic tests checked pooled metric algebra against direct calculations, Holm correction, rank-biserial sign and zero handling, and BCa intervals against SciPy's implementation using matching bootstrap replicates. An end-to-end synthetic run validated all 17 configurations, the 12-test family, output packaging and rejection of deliberately changed targets. No new thesis models or thesis-result statistical analyses were run in the assistant workspace.

Outputs are fingerprinted. If you intentionally change source predictions, packages or script settings, preserve the old evaluation output folder before a new run. Rerunning identical inputs/settings is allowed. Existing model scripts and their training manifests are untouched.
