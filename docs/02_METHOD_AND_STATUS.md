# Implementation settings and status

Accepted: primary soil 0–20 cm, depth sensitivity 20–50 cm, V27 omitted. Unchanged main/all-cell labels. High-yield plot/unit review remains unresolved; do not choose a cleaning policy by model score.

RF: n_estimators 100–800; max_depth None or integers 3–15; min_samples_leaf 1–8; max_features sqrt/0.3/0.5/0.7. XGBoost: n_estimators 50–400, learning_rate 0.01–0.3 log-uniform, depth 2–7, row/column subsampling 0.5–1, L1/L2 regularisation 0–5. CatBoost: iterations 100–400, learning_rate 0.01–0.3 log-uniform, depth 3–7, L2 leaf regularisation 1–10; Ordered boosting, SymmetricTree, RMSE, no categorical predictors. RF bootstrap defaults and XGBoost histogram squared-error settings are recorded in run_models.py.

TPE: seed 20261006, 10 startup trials, 20 attempted trials per tuning task. Median pruning: five startup trials, one warmup step; running mean validation RMSE reported after each fold. Inner objective is unweighted mean of five grouped-fold RMSEs. No outer-test early stopping or prediction clipping. OLS VIF cutoff 10, mean imputation, 2.5/97.5% predictor winsorisation. Ridge grid 0.01/0.1/1/10/100/1000, chosen inside grouped tuning; future stacking meta-alpha remains fixed at 1, independent of standalone Ridge tuning.

The preliminary cloud linear benchmarks ran on both panels; tree fitting was interrupted when the user chose a local reproducibility workflow. Completed cloud checkpoints are separate from this starter project. No complete tree comparison, stacking, inferential analysis, LOYO or SHAP completion is claimed here. Your local outputs should be labelled with their input hashes, source commit and actual environment.

Feature engineering can be rebuilt from the bundled verified public-source extracts and aggregated yield labels. Rebuilding original labels from raw LSMS requires placing the five original CSV ZIP archives in data/private/lsms, running pipeline.py yield, checking the audit, and deliberately copying approved panel CSVs to feature_engineering/inputs with updated input hashes. It is not an automatic substitution because label cleaning is a research decision.
