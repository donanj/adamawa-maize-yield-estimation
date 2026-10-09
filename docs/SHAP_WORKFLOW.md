# Saved-model SHAP workflow

Copy src/interpretability, requirements-shap.txt and this document into the existing project. Do not move model outputs or replace earlier code.

Run in the VS Code terminal at the project root:

```powershell
.venv\Scripts\python.exe -m pip freeze > environment-before-shap.txt
.venv\Scripts\python.exe -m pip install -r requirements-shap.txt -c requirements.txt -c requirements-figures.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe src\interpretability\run_shap.py check
.venv\Scripts\python.exe src\interpretability\run_shap.py smoke
```

Use the updated requirements-figures.txt with Matplotlib 3.11.2. Constraints preserve original model/figure package versions. If installation or checks fail, share the full error; do not upgrade model libraries to work around it.

Smoke explains the first saved research outer fold (Tier 1, Demsa), using all three genuine fitted base models. It does not fit/tune models or provide research-wide SHAP results. Share its output before the full run.

After smoke passes:

```powershell
.venv\Scripts\python.exe src\interpretability\run_shap.py run
.venv\Scripts\python.exe src\interpretability\run_shap.py verify
.venv\Scripts\python.exe -m pip freeze > environment-after-shap.txt
```

Upload src/interpretability/outputs/main_min5_reference_99_0_20_v3/SHAP_Main_Panel_Review.zip.

The full run covers 51 saved outer models: 17 LGAs × three tiers, and 174 held-out observation explanations. Progress prints for each base learner and outer fold. Runtime depends on tree sizes; no tuning is performed. Completed folds resume after interruption. Code, package, input and model changes invalidate cached runs. Verify checks cached numerical decompositions and saved prediction records and regenerates figures; it does not recalculate TreeSHAP.

Interventional TreeSHAP is explicitly requested for each numeric-feature base learner. All outer-training rows form a common empirical background, processed separately with saved preprocessing. The script stops if an explainer is unsupported or additivity fails; it never substitutes path-dependent SHAP silently.

For observation i in outer fold g:

    phi_stack(i,j) = sum_b beta(g,b) * phi_base(g,b)(i,j)
    baseline_stack(g) = intercept(g) + sum_b beta(g,b) * expected_base(g,b)

No meta-input scaling factor applies: the saved Ridge model used unstandardised base predictions. Additivity is checked against both actual base-model predictions and the original saved stacking predictions. The background mean prediction is checked independently.

Global rankings are pooled descriptive model attributions, not evidence of causal drivers or accurate agronomic relationships. Limited spatial and temporal predictive performance must accompany interpretation. Correlated predictors can share or redistribute attribution; interventional masking can create implausible feature combinations. Feature-gating differences and fold-specific baselines must be reported. Absent candidates receive zero attribution in pooled summaries and have inclusion counts, not claims of irrelevance.

Outputs include global importance for all tiers, the top three dependence plots per tier, and three diagnostic Tier 3 waterfalls, with 400 dpi PNG and vector PDF figures. Source tables, selection rules, numerical audits, package versions and hashes are included. No household-level data or trained model files are included in the review ZIP. Do not publish restricted source survey data on GitHub.

Reference implementation documentation: https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html
Pinned release: https://pypi.org/project/shap/0.52.0/

Validation performed before delivery: syntax, weighted-combination algebra, synthetic protocol checks and output construction. Exact compatibility with your saved XGBoost/CatBoost versions must be established by the local smoke test; it is not claimed pre-verified here.

## Version 2 compatibility correction
SHAP 0.52.0 checks both actual categorical metadata and the sklearn wrapper enable_categorical flag. That flag can trigger rejection even for numeric trees. This adapter supplies a copied underlying XGBoost Booster to TreeExplainer only after checking the serialized feature types, split types and category partitions, and verifying Booster predictions match the saved wrapper on every background and held-out row. No retraining, parameter mutation, model-file writes or library monkey-patching occurs. Genuine categorical trees cause a clear stop. The unchanged interventional method, common background and original wrapper predictions remain the basis of all audits.

Version 2 uses a separate output folder, so the failed first smoke manifest remains untouched. Extract this patch into the project and replace src/interpretability/run_shap.py. Dependencies are unchanged; rerun check and smoke. Do not rerun model training.

## Version 3 CatBoost output correction
The SHAP 0.52.0 numeric CatBoost loader reads tree leaf values but omits the saved model scale and bias. Version 3 checks, on ALL background and held-out rows, that applying the saved scale and bias to the parsed tree predictions reproduces native CatBoost predictions. It then scales feature SHAP values and adds the saved bias only to the expected-value baseline. It also checks unscaled tree SHAP additivity, corrected native CatBoost additivity, background expectation and final stacking reconstruction. No residual-derived offset or tolerance relaxation is used. A mismatch outside this specific affine explanation stops with diagnostic errors. Models, preprocessing and interventional masking are unchanged.

Version 3 outputs use a separate folder ending _v3; keep the previous folders. No package reinstall or model fitting is required. Local saved-model smoke confirmation is still required.
