"""Call separately on the training subset at EVERY outer/inner/stacking split."""
PHENOLOGY=['V01','V02','V03','V06']
def eligible_features(training_frame):
    pair_valid=training_frame[['V01','V02']].notna().all(axis=1)
    allow_phenology=pair_valid.mean()>=0.80
    candidates=[f'V{i:02}' for i in range(1,34)]
    kept=[]
    for col in candidates:
        if col in PHENOLOGY and not allow_phenology:continue
        if col not in training_frame or not training_frame[col].notna().any():continue
        if training_frame[col].nunique(dropna=True)<=1:continue
        kept.append(col)
    return kept
# This selects eligible columns only. Fit mean imputation, scaling, VIF screening
# and models on that same training subset, then transform validation/test rows.
# Recompute eligibility inside nested tuning and each stacking OOF training split.
