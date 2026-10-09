# Thesis figures from VS Code

## Run in the VS Code PowerShell terminal

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-figures.txt
.venv\Scripts\python.exe src\visualisation\make_figures.py check
.venv\Scripts\python.exe src\visualisation\make_figures.py run
```

The dependency addition is matplotlib==3.10.8, the version used to test these plots. Keep your existing model requirements file unchanged; requirements-figures.txt records the plotting addition. No model training is performed. The check command validates the saved results and creates no plots. The run command performs the same checks and generates all six figures. This should be much faster than model fitting.

Inputs must already exist in:
- src/evaluation/outputs/main_min5_reference_99_0_20/
- src/temporal_validation/outputs/research/

Do not extract review ZIPs into those paths on your local project if the original outputs are already there; the script reads the CSVs created by your evaluation and LOYO commands.

## Figures

| File prefix | Content | Intended use |
|---|---|---|
| 01_model_rmse_intervals | All 17 spatial model configurations with conditional RMSE BCa intervals | Chapter 5 model comparison |
| 02_predictor_tiers | Point RMSE across the three predictor tiers | Chapter 5 ablation; selected interpretation in Chapter 6 |
| 03_observed_vs_predicted | Stacking T3, satellite-only MLR and mean baseline with identical axes | Chapter 5 prediction diagnostics |
| 04_error_concentration | Individual and cumulative squared-error shares for stacking T3 | Chapter 5 diagnostics; Chapter 6 discussion of influential observations |
| 05_lolo_vs_loyo | Paired spatial/temporal RMSE for all configurations | Chapter 5 temporal sensitivity; Chapter 6 generalisation discussion |
| 06_held_out_years | Year-specific temporal RMSE for highlighted spatial configurations | Chapter 5 year-level sensitivity |

Chapter assignment is a suggested use, not a new thesis structure. Final figure numbers depend on other figures already in your chapter. Refer back to selected figures in Chapter 6 rather than duplicating the entire set. Chapter 7 generally needs a concise synthesis of research questions, findings and recommendations; no new figure is forced into it. SHAP plots, sample coverage, residual diagnostics, maps and target-sensitivity figures can be added as separate later stages with their appropriate inputs.

## Outputs

src/visualisation/outputs/main_panel_q99/

Each figure has:
- PNG at 400 dpi (default), suitable for insertion into Word
- PDF with vector plot elements and embedded fonts, suitable for export/sharing
- source CSV containing the exact rows used to draw the figure

The folder also contains Figure_Captions.md, figure_manifest.json and Thesis_Main_Panel_Figures.zip. The manifest records software versions and hashes of the script and result inputs. Upload the review ZIP after your local run for final review.

Use --dpi 600 for larger raster exports:
```powershell
.venv\Scripts\python.exe src\visualisation\make_figures.py run --dpi 600
```

Repeated runs regenerate the figure files and ZIP. Model data/results remain untouched. Preserve a dated copy of figures if comparing different formatting versions. Do not manually edit source CSV values to change a plotted result.

