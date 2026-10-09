# Target-quality sensitivity protocol — 6 October 2026

## Purpose and status

Assess how label construction affects conclusions after the extreme-yield audit and reference model results. This is an exploratory, post-audit sensitivity analysis. No scenario is designated corrected ground truth. Reference labels and their completed model outputs remain unchanged. Raw harvest quantities, unit codes, GPS areas, reported areas and survey weights are not edited.

## Independent evidence and limits

World Bank LSMS methodological guidance identifies recall, non-standard harvest units, rounding and area measurement as sources of crop-productivity measurement error. It supports documented conversion/area review. It does not establish a universal maize-yield exclusion cutoff. Farmer-reported and GPS areas can disagree; neither should be substituted solely to make an extreme yield smaller.

A Nigerian rice study using the 2010/11 and 2012/13 LSMS-ISA waves reports 99th-percentile winsorisation and a 95th-percentile alternative where large values persisted (data section, PDF page 14). This supplies precedent for a stronger percentile sensitivity, not validation of a maize cutoff or endorsement of every part of that study's methodology. The present protocol uses the documented current area/survey-weight rules throughout.

IITA reports a Nigerian maize outgrower average of 6 t/ha in 2020. That is agronomic context, not a maximum, and it does not independently validate 7.5, 10 or 15 t/ha as hard ceilings. Accordingly, all fixed-cutoff scenarios remain diagnostic. They were already present in the earlier source-only extreme-yield review before model fitting; no cutoff is selected here by observed predictive performance.

## Scenarios

| Identifier | Plot-level rule | Status |
|---|---|---|
| reference_99 | Existing upper-tail wave q99 winsorisation | Reference |
| winsor_95 | Upper-tail wave q95 winsorisation, no deletion | Statistical sensitivity |
| exclude_gt7500 | Exclude raw yields >7,500 kg/ha, then recompute wave q99 | Diagnostic threshold sensitivity |
| exclude_gt10000 | Exclude raw yields >10,000 kg/ha, then recompute wave q99 | Diagnostic threshold sensitivity |
| exclude_gt15000 | Exclude raw yields >15,000 kg/ha, then recompute wave q99 | Diagnostic threshold sensitivity |

All quantiles use unweighted eligible plot yields within each wave, with pandas linear interpolation, matching the reference upper-q99 construction. For exclusions, recompute the percentile AFTER exclusions, before LGA cell filtering. No lower-tail trimming; genuine zero yields remain. Area × household survey weighting is retained. Main cells require five eligible plots, all-cell sensitivity requires one. Rules operate consistently across all five waves.

These are target definitions constructed before model CV from the full eligible survey-wave data, as in the reference pipeline. They are not training-fitted predictor transformations. Evaluation is conditional on those labels; percentile-based target construction also uses records associated with held-out LGAs. Predictor eligibility, winsorisation, imputation, scaling and tuning remain restricted to their training partitions.

GPS/reported area ratios outside [0.25,4] are recorded only as review aids. This analyst-selected factor-four flag is not a validated error criterion and causes no exclusions or substitutions.

## Checks and expected counts

Preparation validates weight/yield identities, reference q99 clipping, unique plot keys and agreement with original reference panels. Feature values remain unchanged on matching LGA-year keys.

| Rule | Main cells/LGAs | All-cell cells/LGAs | Main retained plots |
|---|---|---|---:|
| reference_99 | 58/17 | 63/18 | 662 |
| winsor_95 | 58/17 | 63/18 | 662 |
| exclude_gt7500 | 57/17 | 63/18 | 622 |
| exclude_gt10000 | 58/17 | 63/18 | 642 |
| exclude_gt15000 | 58/17 | 63/18 | 646 |

The q95 Wave 2 cap remains 27,748.19 kg/ha versus q99 121,090.38; q95 does not resolve every questionable record. Smaller final yields are not proof of accuracy.

## Evaluation and reporting

Run the same model definitions, predictor tiers, soil depths, seeds and tuning budget in isolated per-rule folders. Preserve held-out LGA grouping; the 7,500 scenario loses one main-panel cell, so cell coverage and possibly inner split assignments differ. Report those differences. Generated scripts adapt row-count assertions; do not manually force every variant to 58 rows.

Start with linear benchmarks for winsor_95. Later compare the remaining scenarios using the same protocol if needed. Do not switch primary policy because RMSE or R² improves: target scales/distributions differ across scenarios, so smaller RMSE on altered labels is not evidence that those labels are more accurate. These comparisons assess stability of conclusions under alternative label assumptions. No causal interpretation or definitive error classification is warranted. Stacking remains outstanding.

## Outputs and reproducibility

src/target_quality/outputs/protocol.json records source/data fingerprints and the complete fixed scenario set. Changed inputs, source code or base model scripts require a separate preserved run location. panel_summary.csv records counts/clipping; wave_caps.csv records thresholds; cell_changes.csv records target changes and omitted cells. runs/<rule>/ contains labelled panels, feature inputs and isolated model runners/results. private_flag_register.csv contains survey linkage keys; the project's existing **/outputs/ Git ignore rule keeps it local. No private data is included in this patch.

The previous tree results remain at src/tree_models/outputs. Sensitivity tree outputs will be at src/target_quality/outputs/runs/<rule>/tree_models/outputs. No automatic promotion or replacement of the main analysis is performed.

## Sources reviewed

- World Bank LSMS, Agriculture methodological research: https://www.worldbank.org/en/programs/lsms/priority-themes/survey-methods/agriculture
- Oseni, Durazo and McGee (2017), non-standard quantity unit guidebook: https://www.worldbank.org/en/programs/lsms/publication/the-use-of-non-standard-units-for-the-collection-of-food-quantity-a-guidebook-for-improving-the-measurement-of-Food-consumption-and-agricultural-production-in-living-standards-surveys
- Nigerian LSMS rice-study manuscript, data section: https://www.worldbank.org/content/dam/Worldbank/document/Africa/Event/Harnessing%20Africa%20s%20Growth%20for%20Faster%20Poverty%20Reduction/afr-lenis-saweda-liverpool-tasie.pdf
- IITA (6 March 2020), Nigerian maize outgrower production report: https://iita.org/news-item/iita-bip-sets-record-for-maize-production-per-hectare-in-nigeria/
