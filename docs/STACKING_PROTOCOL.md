# Fully nested stacking: local execution

Extract this patch into the existing adamawa-maize-yield project root. It adds src/stacking/run_stacking.py and this document. It does not replace previous scripts or results. Uses the existing installed packages and tree-model preprocessing/estimators.

## First commands (PowerShell, project root)

```powershell
.venv\Scripts\python.exe src\stacking\run_stacking.py smoke
.venv\Scripts\python.exe src\stacking\run_stacking.py run
.venv\Scripts\python.exe src\stacking\run_stacking.py verify
```

Smoke uses one main-panel outer LGA, Tier 1, and two attempted trials per study in a separate folder. Its metrics are installation checks, not thesis results. It still exercises the full nesting. Run defaults to q99 reference, main panel, 0–20 cm, all three tiers, 20 attempted trials per study and two workers. There are 51 outer tasks, 918 tuning studies, and 18,360 attempted trials; pruned trials need not complete all folds. This is substantially more expensive than standalone models. No runtime guarantee. ADAMAWA_WORKERS can reduce parallelism if memory is constrained.

Each outer task saves every completed base-model stage (18 per outer task). Repeating the identical command skips finished outer tasks and reuses completed stages within interrupted tasks. An interrupted tuning study restarts its fixed trial budget; it does not resume trial-by-trial. Do not edit code, inputs, packages or Python version during a run: fingerprints reject changed environments. Keep prior run folders when making intentional changes.

## Three levels of geographic isolation

1. Outer LOLO: hold out every year of one LGA. Its observations are excluded from all training and tuning.
2. Stacking OOF: split the outer training LGAs into five groups. For each split and base learner, tune exclusively on the other four groups, then refit there and predict the held-out group. Every outer-training row receives one OOF prediction from each base learner.
3. Tuning: five GroupKFold splits within the current stacking training subset. Fit feature eligibility, clipping and imputation separately inside each tuning training split. Refit preprocessing on the complete current subset only after tuning.

Fit Ridge(alpha=1.0), with intercept and without meta-input scaling, to the three OOF prediction columns in RF/XGBoost/CatBoost order and the outer-training targets. Alpha is fixed, not tuned. Ridge coefficients are unconstrained; no prediction clipping is applied. Retune each base learner using five grouped folds on the full outer-training set, refit there, and feed its outer-held-out predictions into the fitted meta-learner. Pooled outer predictions are the evaluation output; meta training fit is not an evaluation result.

Base learners, tuning spaces, scoring, seed and pruning follow the existing standalone tree script. Tree preprocessing is reused unchanged: training-specific feature gate, predictor clipping and imputation; no VIF screening or scaling for trees. Targets are not transformed. Label cleaning remains the previously documented wave-wide procedure, not fold-internal target processing; interpretation is conditional on the constructed labels.

## Verification and outputs

Verification checks every tuning/stacking partition for group isolation and training coverage, attempted trial counts, outer held-out coverage and target identity. It reconstructs all stacking OOF inputs from saved base models, refits the fixed Ridge meta-learner to check coefficients, and reproduces outer predictions from saved models. The underlying preprocessing implementation remains the one checked by verify-trees; this patch does not independently reimplement its statistical operations.

Outputs: src/stacking/outputs/research/<rule>/<panel>/<depth>/
- pooled_metrics.csv, oof_predictions.csv, per_LGA_metrics.csv
- environment.json: input/source hashes and software versions
- outer-task folders: result.json with split audit, base models/preprocessors, studies, stack.joblib with meta learner and training inputs

The metrics contain only the configurations selected by the current command. Verify with the same flags used for run. Per-task JSON and models remain available if a later command writes a different selection into the summary CSVs. Local outputs are already excluded by the existing project Git ignore rules. Do not publish model/data outputs without separately reviewing data-access requirements.

## Subsequent configurations (after reviewing primary run)

All-cell q99:
```powershell
.venv\Scripts\python.exe src\stacking\run_stacking.py run --panel all_cells
.venv\Scripts\python.exe src\stacking\run_stacking.py verify --panel all_cells
```
Main-panel soil-depth sensitivity:
```powershell
.venv\Scripts\python.exe src\stacking\run_stacking.py run --tier 3 --depth 20_50
.venv\Scripts\python.exe src\stacking\run_stacking.py verify --tier 3 --depth 20_50
```
Target sensitivities can be run with --rule winsor_95 or --rule exclude_gt10000, with identical flags on verification. These read their already prepared isolated inputs. They do not promote a new target policy. Complete the reference run before deciding how much additional sensitivity computation is justified.

No new thesis models were fitted when developing this patch. A synthetic-data test with lightweight stand-in estimators and a stand-in tuning driver exercised nesting, cached resumption, prediction reconstruction and deliberate corruption rejection; real estimators are exercised by the user's local smoke command. Stacking superiority is an empirical question, not an assumption.
