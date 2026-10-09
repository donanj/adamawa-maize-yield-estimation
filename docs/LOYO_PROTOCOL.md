# LOYO-CV: temporal sensitivity for the accepted main panel


## Local commands

```powershell
.venv\Scripts\python.exe src\temporal_validation\run_loyo.py smoke
```

Share the smoke output first. It holds out crop year 2010 at Tier 1 with two attempted trials per study and produces installation-check metrics only, in outputs/smoke. It includes mean, MLR, RF, XGBoost, CatBoost and stacking for that one year.

After the smoke check passes:

```powershell
.venv\Scripts\python.exe src\temporal_validation\run_loyo.py run
.venv\Scripts\python.exe src\temporal_validation\run_loyo.py verify
```

The research run evaluates the reference q99 main panel: 58 LGA-year observations, 17 LGAs, 0–20 cm soil depth, years 2010/2012/2015/2018/2023. Fifteen year/tier tasks cover all five years and all three tiers. Each has 15 tree tuning studies; total 225 studies and 4,500 attempted trials. The standalone Ridge additionally evaluates its six fixed alpha candidates in four inner year holdouts for each of five outer folds. Actual time depends on pruning, hardware and the fitted data. Two workers run by default; ADAMAWA_WORKERS can reduce parallelism.

Progress prints at every base tuning stage start and completion, plus each outer task completion. Base-stage models are checkpointed; the identical command reuses them if interrupted. An interrupted tuning study restarts its trial budget. Completed outer results skip all fits. Source/input/software hashes reject changed environments. Do not change code/packages/data while running; preserve old folders for intentional changes.

## Validation design

Outer LOYO holds out every observation from one crop year, training on the other four years. It permits the same LGA to occur in training and test years. This is temporal transfer across observed years, not joint unseen-LGA/unseen-year validation. Because earlier held-out years can use later years for training, it is not rolling-origin future forecasting.

Within each outer training set, stacking OOF predictions use four leave-one-year-out splits. A base learner predicting an inner held-out year is tuned exclusively on the remaining three years, using three leave-one-year-out tuning folds; each tuning training split therefore contains two years. After tuning, refit on those three years and predict the inner held-out year. Each outer-training observation receives one OOF prediction from each base learner.

Fit Ridge(alpha=1.0), with intercept and unstandardised RF/XGBoost/CatBoost OOF inputs, to outer-training targets. Final base learners are tuned using four leave-one-year-out splits on the full outer training set and refitted there. Their predictions for the outer held-out year feed the fixed meta-learner. Those same final base models supply the standalone tree results, ensuring that standalone and stacking base predictions share the final tuning/fitting procedure. No LOLO-selected hyperparameters or meta models are reused.

Linear models use the same outer year holdouts. OLS MLR uses fold-internal predictor screening, clipping, imputation, scaling and VIF/rank checks. Standalone Ridge is Tier 3 only and selects alpha from .01/.1/1/10/100/1000 by mean four-fold temporal validation RMSE, with preprocessing refitted within each split. Mean baseline is fitted to the outer-training targets and evaluated once per year. Tree features are not scaled or VIF-screened. The base tuning spaces, seed, 20-trial budget, TPE/pruning settings and mean-validation-fold RMSE objective follow the existing tree script.

The five crop years cannot support a five-fold temporal inner split after one year is held out. Thus LOYO intentionally uses four inner year holdouts (and three at the deeper stacking tuning level). Record this difference from LOLO's grouped-LGA inner five-fold scheme in Chapter 4. Every preprocessing fit excludes its current validation year and the outer held-out year. Targets remain the preconstructed wave-wide upper-q99 labels; this is not a new target-cleaning policy or fold-internal target transformation.

## Models and outputs

Research includes the same 17 primary configurations as the main LOLO comparison: mean once; MLR at three tiers; Ridge at Tier 3; RF/XGBoost/CatBoost/stacking at three tiers. Each configuration predicts all 58 observations once across five outer years.

src/temporal_validation/outputs/research/
- pooled_metrics.csv: RMSE, MAE, R², RRMSE, MAPE
- per_year_metrics.csv: held-out year metrics and sample counts
- oof_predictions.csv: keyed outer-year predictions
- LOLO_vs_LOYO.csv: descriptive comparison of matching configurations, generated when all three saved LOLO source prediction files are available
- Review_Report.md: scope and interpretation limits
- environment.json: settings, versions and input/source hashes
- T<tier>_year_<year>/ folders: saved base preprocessing/models, studies, meta model, linear models and nested split audits
- LOYO_Main_Panel_Review.zip: aggregate output files for review; fitted models and studies remain local

Share LOYO_Main_Panel_Review.zip after research verification. The output ZIP contains LGA/year aggregates, not household/plot identifiers.

Verification reconstructs stacking OOF inputs from saved models, checks every nested split for year isolation and coverage, refits the fixed Ridge meta model to verify coefficients, reproduces saved outer predictions, checks unchanged targets, and rebuilds linear outer preprocessing. It checks that all 17 configurations predict all 58 observations over five held-out years. LOLO comparison additionally requires identical keys and target values across validation schemes.

## Interpretation

LOYO is a descriptive temporal sensitivity. The difference from LOLO reflects different generalisation targets, training composition and inner selection schemes; do not interpret it as an isolated estimate of temporal dependence. With five held-out years, this patch adds no year-level hypothesis tests or bootstrap intervals. No future-season forecasting claim follows from a favourable LOYO result. Pooled metrics and year-level errors should be interpreted together, especially the influential 2012 observations already identified.

## Patch checks

No thesis models were fitted in the assistant workspace. Synthetic tests with stand-in tree estimators and tuning driver exercised all 17 configurations, all five year holdouts, three/four-year nested isolation, actual linear-model/preprocessing operations, checkpoint reuse, saved prediction reconstruction and output packaging. Deliberately adding an outer-held-out observation to an inner training audit was rejected. The local smoke command will exercise actual Optuna/tree estimators before the research run.
