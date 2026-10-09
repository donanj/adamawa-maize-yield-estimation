# Adamawa Maize Yield Estimation

Research code supporting an MSc study of maize yield estimation at Local Government Area (LGA) level in Adamawa State, Nigeria, using satellite, climate, soil and terrain predictors.

**Author:** Donan Donald Joshua  
**Programme:** MSc Data Science and Analytics, American University of Nigeria

## Study overview

The study compared linear baselines, Random Forest, XGBoost, CatBoost and a stacking ensemble combining the three tree learners through a Ridge meta-learner. Monteith’s radiation-use efficiency framework informed the organisation of predictors into three cumulative tiers:

| Tier | Predictor configuration |
|---|---|
| 1 | Satellite vegetation, phenological and land-surface temperature predictors |
| 2 | Tier 1 plus climate predictors |
| 3 | Tier 2 plus soil and terrain predictors |

The primary panel contained **58 LGA-year observations across 17 LGAs**, covering crop seasons **2010, 2012, 2015, 2018 and 2023**. An all-cell sensitivity panel contained 63 observations across 18 LGAs. Primary soil predictors represented 0–20 cm depth; 20–50 cm was evaluated separately.

Primary evaluation used **Leave-One-LGA-Out (LOLO) cross-validation**. Hyperparameter selection, preprocessing and stacking out-of-fold prediction generation were nested within the training data. Leave-One-Year-Out (LOYO) validation assessed retrospective temporal transfer. Supporting analyses covered predictor enrichment, target treatment, soil depth, cropland masks, ensemble constraints, tuning-seed sensitivity and SHAP explanations.

## Code organisation

The implementation consists of the base project and additional analysis packages. Before using the commands below, copy the latest packages into the same project root, merging their `src/`, `docs/` and `gee/` contents while preserving their internal paths. Use the latest corrected scripts for each module, including the version 3 saved-model SHAP compatibility corrections.

| Path in the assembled project | Purpose |
|---|---|
| `pipeline.py` | Preparation and baseline/tree-model stage commands |
| `requirements.txt` | Pinned core dependencies |
| `requirements-shap.txt`, `requirements-figures.txt` | SHAP and plotting dependencies |
| `config/` | Input hashes and project settings |
| `docs/` | Setup, data access and analysis protocols |
| `gee/` | Google Earth Engine extraction scripts |
| `src/yield_panel/` | Survey source audits and yield-panel construction |
| `src/feature_engineering/` | Predictor derivation, definitions and verification |
| `src/linear_benchmarks/` | Mean, multiple linear regression and Ridge benchmarks |
| `src/tree_models/` | Random Forest, XGBoost and CatBoost evaluation |
| `src/stacking/` | Fully nested spatial stacking |
| `src/evaluation/` | Saved-prediction metrics and exploratory comparisons |
| `src/temporal_validation/` | Year-holdout evaluation |
| `src/interpretability/` | Held-out saved-model SHAP explanations |
| `src/final_model/` | Full-panel development refits and SHAP |
| `src/target_quality/` | Target-treatment sensitivity |
| `src/stability/` | Development-model tuning-seed diagnostics |
| `src/meta_sensitivity/` | Constrained stacking sensitivity |
| `src/cropland_mask/` | Masked feature rebuilding and matched spatial evaluation |
| `notebooks/` | Optional notebook setup companion |

The base project alone does not contain every additional module listed above. An upload should include the assembled source files and their protocol documents so the repository matches this inventory.

## Datasets and access

| Dataset | Role |
|---|---|
| Nigeria GHS-Panel / LSMS-ISA, National Bureau of Statistics and World Bank | Survey-derived maize yield reference |
| MOD13Q1 Collection 6.1 | NDVI/EVI time series and phenological predictors |
| MOD11A2 Collection 6.1 | Day and night land-surface temperature |
| CHIRPS Version 2 | Rainfall |
| ERA5-Land daily aggregates and hourly data | Climate predictors, including wind |
| iSDAsoil Africa v1 | Soil properties |
| SRTMGL1 Version 3 | Elevation and derived slope |
| MCD12Q1 Collection 6.1 | Annual cropland-mask sensitivity |
| FAO GAUL 2015 level 2 | LGA extraction boundaries |

See [data-access documentation](docs/DATA_ACCESS.md) for source links and local input placement. The Earth Engine scripts define the Adamawa extraction area and reductions. GAUL 2015 is the version used by the study; replacing it with a newer boundary product changes the extraction specification.

Raw household and plot microdata, survey archives, private audit identifiers and credentials should remain outside the public repository. Obtain survey microdata through the authorised distributor under the applicable access agreement. Third-party datasets retain their own licences and attribution requirements; public catalogue access does not establish a common redistribution licence for all inputs.

A code-only clone requires authorised inputs to be populated locally. The preparation stages expect extracts under `src/feature_engineering/inputs/`. Exact original extracts are checked against `config/input_hashes.json`. Newly generated exports require a separately documented input version; preserve the original manifests and outputs.

## Environment setup

Run commands from the project root. For reproduction of saved models, use the Python and package versions recorded in their run manifests. The base pipeline requires Python 3.11 or newer, but that minimum does not establish compatibility with an existing saved-model environment.

For a new Windows environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-figures.txt -r requirements-shap.txt
.venv\Scripts\python.exe -m pip check
```

The additional dependency files must be copied from the corresponding analysis packages. If the completed study environment already exists, reuse it. Keep the pinned package versions when loading saved models. The SHAP requirements record version 0.52.0. Earlier figure packages and later SHAP documentation record different Matplotlib versions; use the plotting requirement matching the relevant saved-run manifest.

## Execution workflow

### 1. Prepare and verify inputs

Populate the authorised local inputs before running:

```powershell
.venv\Scripts\python.exe pipeline.py check
.venv\Scripts\python.exe pipeline.py features
.venv\Scripts\python.exe pipeline.py prepare
```

Optional rebuilding from the five original survey archives uses `python pipeline.py yield`, with archives under `data/private/lsms/`. Consult the source-audit and data-access documentation before accepting revised labels.

### 2. Run the primary spatial comparison

```powershell
.venv\Scripts\python.exe pipeline.py linear
.venv\Scripts\python.exe pipeline.py smoke
.venv\Scripts\python.exe pipeline.py trees
.venv\Scripts\python.exe pipeline.py verify-trees
.venv\Scripts\python.exe src\stacking\run_stacking.py smoke
.venv\Scripts\python.exe src\stacking\run_stacking.py run
.venv\Scripts\python.exe src\stacking\run_stacking.py verify
.venv\Scripts\python.exe src\evaluation\analyse_results.py
```

Smoke outputs are installation checks and are stored separately from research outputs. Research tree tuning uses 20 attempted trials per study, including pruned trials. Primary stacking uses a fixed Ridge meta-learner with `alpha=1.0`, an intercept and unstandardised base predictions. Consult [the stacking protocol](docs/STACKING_PROTOCOL.md) for nesting and sensitivity options.

### 3. Run complementary analyses

For each stage-based script below, run `smoke`, then `run`, then `verify`. The interpretability and final-model scripts additionally support `check`, which should precede `smoke`.

| Script | Required preceding material |
|---|---|
| `src/temporal_validation/run_loyo.py` | Prepared primary inputs; saved spatial outputs for LOLO/LOYO comparison |
| `src/interpretability/run_shap.py` | Saved primary spatial stacking models |
| `src/final_model/run_final.py` | Prepared primary inputs and the original training helpers |
| `src/final_model/run_final_shap.py` | Verified final research models and SHAP compatibility helpers |
| `src/stability/run_stability.py` | Saved final research models |
| `src/meta_sensitivity/run_constrained.py` | Saved primary spatial stacking models |
| `src/cropland_mask/run_masked_lolo.py` | Rebuilt masked features and saved primary stacking outputs |

For example:

```powershell
.venv\Scripts\python.exe src\temporal_validation\run_loyo.py smoke
.venv\Scripts\python.exe src\temporal_validation\run_loyo.py run
.venv\Scripts\python.exe src\temporal_validation\run_loyo.py verify
```

Follow the module-specific protocol documents for target/depth configurations, masked feature preparation and output locations. Later analyses load earlier saved models; preserve the complete local output folders. Identical commands can reuse completed stages where checkpointing is supported. Keep code, inputs and dependencies fixed during a run. `ADAMAWA_WORKERS` controls concurrency for the runners that support it.

## Outputs and interpretation

Modules generate metrics, held-out predictions, split audits, configurations, environment records and, where applicable, saved models, SHAP tables and figures. Run verification with the same configuration flags used for fitting.

Use held-out LOLO and LOYO results when discussing generalisation. Full-panel fitted predictions describe the development models and are in-sample. LOYO is retrospective year-holdout validation, rather than prospective forecasting. SHAP explains model associations and does not identify causal yield effects. The evaluation bootstrap holds saved predictions fixed; tuning-seed diagnostics examine development-model sensitivity.

## Reproducibility and code availability

Archive the code commit or release identifier together with input hashes, dependency versions, seeds, run settings and verified outputs used for the thesis. Reproducing exact results requires the matching authorised data and software environment as well as the complete source modules.


When citing this work, acknowledge Donan Donald Joshua, the accompanying MSc thesis, the repository URL and the release or commit used. Dataset producers should be cited separately according to their guidance.
