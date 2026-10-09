"""Explain saved main-panel LOLO stacking predictions; never fit a model."""
from pathlib import Path
import argparse, hashlib, json, platform, sys, zipfile
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src/tree_models'))
import preprocessing  # Required to deserialize the saved Prep objects.
BASE = ['RF', 'XGBoost', 'CatBoost']
MODELS = PROJECT / 'src/stacking/outputs/research/reference_99/main_min5/0_20'
INPUT = PROJECT / 'src/tree_models/inputs/main_min5_full_candidates_0_20.csv'
OUT = PROJECT / 'src/interpretability/outputs/main_min5_reference_99_0_20_v3'
TOL = 1e-3  # Absolute kg/ha tolerance plus 1e-6 relative tolerance.

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, atol=TOL, rtol=1e-6):
        raise ValueError(f'{label}: reconstruction failed; no interpretation should be used.')

def transform(prep, frame):
    if prep.indices != list(range(len(prep.cols))):
        raise ValueError('Unexpected feature selection in a tree preprocessing object.')
    return prep.transform(frame) * prep.scale + prep.center

def numeric_xgboost_booster(model, bg, x):
    """Explain verified numeric trees directly, avoiding the wrapper categorical flag.

    No training parameter, tree, installed SHAP code or saved model is changed.
    """
    booster = model.get_booster().copy()
    structure = json.loads(bytes(booster.save_raw(raw_format='json')).decode('utf-8'))
    learner = structure['learner']
    if any(t == 'c' for t in (learner.get('feature_types') or [])):
        raise ValueError('Actual categorical XGBoost feature metadata: numeric adapter cannot be used.')
    trees = learner['gradient_booster']['model']['trees']
    for tree in trees:
        if 'split_type' not in tree or any(int(t) != 0 for t in tree['split_type']):
            raise ValueError('Categorical or unrecognised XGBoost tree split: stop for methodological review.')
        if tree.get('categories'):
            raise ValueError('XGBoost category partitions detected: numeric adapter cannot be used.')
    missing = model.get_params().get('missing', np.nan)
    for values, label in [(bg, 'background'), (x, 'held-out')]:
        matrix = xgb.DMatrix(values, missing=missing, feature_names=booster.feature_names)
        close(np.asarray(booster.predict(matrix)).reshape(-1), np.asarray(model.predict(values)).reshape(-1),
              f'XGBoost Booster versus saved wrapper ({label})')
    return booster, dict(adapter='numeric Booster copy',
        saved_wrapper_enable_categorical=bool(model.get_params().get('enable_categorical', False)),
        numeric_split_check=True, tree_count=len(trees), wrapper_prediction_check=True)

def catboost_affine_shap(model, explainer, bg, x, values, expected):
    """Validate the parsed tree-sum function, then apply the saved affine output.

    SHAP 0.52's CatBoost numeric tree loader omits the model scale and bias.
    An offset is added only to the baseline; feature contributions are scaled.
    """
    scale, bias = model.get_scale_and_bias()
    scale = float(scale); bias = float(np.asarray(bias).reshape(-1)[0])
    for data, label in [(bg, 'background'), (x, 'held-out')]:
        parsed = np.asarray(explainer.model.predict(data), float).reshape(-1)
        actual = np.asarray(model.predict(data), float).reshape(-1)
        try:
            close(scale * parsed + bias, actual, f'CatBoost parsed trees plus saved scale/bias ({label})')
        except ValueError as err:
            details = dict(scale=scale, bias=bias,
                max_raw_error=float(np.max(np.abs(parsed-actual))),
                max_affine_error=float(np.max(np.abs(scale*parsed+bias-actual))))
            raise ValueError(f'{err} Diagnostics: {details}') from err
    close(expected + values.sum(axis=1), np.asarray(explainer.model.predict(x)).reshape(-1),
          'CatBoost unscaled tree SHAP reconstruction')
    return scale * values, scale * expected + bias, dict(
        adapter='validated CatBoost tree-sum affine output', scale=scale, bias=bias,
        parsed_prediction_check=True)

def inputs():
    d = pd.read_csv(INPUT).sort_values(['lga_name', 'crop_year']).reset_index(drop=True)
    if len(d) != 58 or d.lga_name.nunique() != 17 or d.duplicated(['lga_name', 'crop_year']).any():
        raise ValueError('Expected the accepted 58-row, 17-LGA main panel.')
    env = json.loads((MODELS / 'environment.json').read_text())
    if env['input_hash'] != digest(INPUT):
        raise ValueError('Candidate panel differs from the saved training input.')
    for relative, old_hash in env['sources'].items():
        if digest(PROJECT / relative) != old_hash:
            raise ValueError(f'Training source changed since models were saved: {relative}')
    mapping = {'sklearn': 'scikit-learn'}
    for name, old in env['packages'].items():
        if version(mapping.get(name, name)) != old:
            raise ValueError(f'{name} version changed since training: expected {old}.')
    if platform.python_version() != env['python']:
        raise ValueError('Python version differs from the saved model environment.')
    folders = [(t, h, MODELS / f'main_min5_0_20_T{t}_{h.replace(" ", "_")}')
               for t in [1, 2, 3] for h in sorted(d.lga_name.unique())]
    for _, _, folder in folders:
        for filename in ['stack.joblib', 'result.json']:
            if not (folder / filename).exists():
                raise FileNotFoundError(folder / filename)
    return d, folders

def explain(d, tier, held, folder):
    result = json.loads((folder / 'result.json').read_text())
    tr = np.flatnonzero(d.lga_name.to_numpy() != held)
    te = np.flatnonzero(d.lga_name.to_numpy() == held)
    if result['train_rows'] != tr.tolist() or result['test_rows'] != te.tolist():
        raise ValueError('Saved fold row indices do not match the candidate panel.')
    stack = joblib.load(folder / 'stack.joblib')
    if stack['base_order'] != BASE or not np.array_equal(stack['meta_training_rows'], tr):
        raise ValueError('Unexpected stacking base order or training rows.')
    close(stack['meta_training_targets'], d.iloc[tr].yield_kg_ha.to_numpy(), 'training targets')
    weights = np.asarray(stack['meta'].coef_, float)
    allcols = [f'V{i:02}' for i in range(1, {1: 12, 2: 22, 3: 33}[tier] + 1)]
    total = np.zeros((len(te), len(allcols)))
    baseline = float(stack['meta'].intercept_)
    preds, base_rows, audit = [], [], []
    common_cols, common_bg, common_test = None, None, None
    for j, (name, bundle) in enumerate(zip(BASE, stack['base_models'])):
        if bundle['train_rows'] != tr.tolist():
            raise ValueError('Base learner training set differs from the outer training fold.')
        prep, model = bundle['preprocessing'], bundle['model']
        bg, x = transform(prep, d.iloc[tr]), transform(prep, d.iloc[te])
        if j == 0:
            common_cols, common_bg, common_test = prep.cols, bg, x
        else:
            if prep.cols != common_cols:
                raise ValueError('Base learner feature spaces differ.')
            close(bg, common_bg, 'common background'); close(x, common_test, 'common held-out inputs')
        print(f'T{tier} {held}: {name} interventional TreeSHAP ({len(tr)} background rows)', flush=True)
        # Passing ALL outer-training observations explicitly avoids subsampling.
        masker = shap.maskers.Independent(bg, max_samples=len(bg))
        shap_model, adapter_audit = model, None
        if name == 'XGBoost':
            shap_model, adapter_audit = numeric_xgboost_booster(model, bg, x)
        explainer = shap.TreeExplainer(shap_model, data=masker, feature_perturbation='interventional', model_output='raw')
        if explainer.feature_perturbation != 'interventional':
            raise ValueError('SHAP changed the requested explanation semantics.')
        sv = np.asarray(explainer.shap_values(x, check_additivity=True), float)
        ev = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        pred = np.asarray(model.predict(x), float).reshape(-1)
        if sv.shape != x.shape:
            raise ValueError('Unexpected SHAP output shape.')
        catboost_audit = None
        if name == 'CatBoost':
            sv, ev, catboost_audit = catboost_affine_shap(model, explainer, bg, x, sv, ev)
        close(ev + sv.sum(axis=1), pred, f'{name} SHAP')
        close(np.array([ev]), np.array([np.mean(model.predict(bg))]), f'{name} background expectation')
        preds.append(pred); baseline += weights[j] * ev
        for k, feature in enumerate(prep.cols):
            total[:, allcols.index(feature)] += weights[j] * sv[:, k]
            for q, i in enumerate(te):
                base_rows.append(dict(tier=tier, lga_name=held, crop_year=int(d.iloc[i].crop_year),
                    model=name, feature_id=feature, shap_value=float(sv[q, k]),
                    ridge_coefficient=float(weights[j]), weighted_shap=float(weights[j] * sv[q, k])))
        audit.append(dict(model=name, ridge_coefficient=float(weights[j]), expected_value=ev,
            xgboost_adapter=adapter_audit,
            catboost_adapter=catboost_audit,
            background_rows=tr.tolist(), included_features=prep.cols,
            max_additivity_error=float(np.max(np.abs(ev + sv.sum(axis=1) - pred)))))
    predicted = np.asarray(stack['meta'].predict(np.column_stack(preds)), float)
    saved = np.array([p['predicted'] for p in result['predictions']], float)
    close(predicted, saved, 'saved outer predictions')
    for i, p in zip(te, result['predictions']):
        if p['lga_name'] != held or p['crop_year'] != int(d.iloc[i].crop_year) or p['observed'] != float(d.iloc[i].yield_kg_ha):
            raise ValueError('Saved observation keys or targets differ.')
    close(baseline + total.sum(axis=1), predicted, 'stacking SHAP')
    records, features = [], []
    for q, i in enumerate(te):
        records.append(dict(tier=tier, lga_name=held, crop_year=int(d.iloc[i].crop_year),
            observed=float(d.iloc[i].yield_kg_ha), predicted=float(predicted[q]), baseline=baseline,
            reconstructed=float(baseline + total[q].sum()), absolute_error=float(abs(predicted[q] - d.iloc[i].yield_kg_ha))))
        for k, f in enumerate(allcols):
            included = f in common_cols
            features.append(dict(tier=tier, lga_name=held, crop_year=int(d.iloc[i].crop_year), feature_id=f,
                shap_value=float(total[q, k]), included=included, raw_value=float(d.iloc[i][f]),
                processed_value=float(common_test[q, common_cols.index(f)]) if included else None))
    return dict(records=records, features=features, base_components=base_rows, audit=audit)

def savefig(path):
    plt.savefig(path.with_suffix('.png'), dpi=400, bbox_inches='tight')
    plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

def figures(records, features, importance, names):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10})
    dest = OUT / 'figures'; dest.mkdir(exist_ok=True)
    for tier in [1, 2, 3]:
        top = importance.query('tier == @tier and included_observations > 0').head(15).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh([f'{f}: {names[f]}' for f in top.feature_id], top.mean_abs_shap, color='#0072B2')
        plt.xlabel('Mean absolute held-out SHAP contribution (kg/ha)')
        plt.title(f'Stacking Tier {tier}: global model attribution')
        savefig(dest / f'T{tier}_global_importance')
        for feature in top.iloc[::-1].feature_id.head(3):
            part = features.query('tier == @tier and feature_id == @feature and included').copy()
            plt.figure(figsize=(7, 4.5))
            sc = plt.scatter(part.processed_value, part.shap_value, c=part.crop_year, cmap='viridis', s=36)
            plt.axhline(0, color='grey', linewidth=.7)
            plt.xlabel(f'{feature}: {names[feature]} (fold-processed value)')
            plt.ylabel('Held-out SHAP contribution (kg/ha)')
            plt.title(f'Tier {tier}: descriptive dependence')
            plt.colorbar(sc, label='Crop year', ticks=[2010, 2012, 2015, 2018, 2023])
            savefig(dest / f'T{tier}_dependence_{feature}')
    # Reproducible selection rule: largest, median-position and smallest absolute error.
    ordered = records.query('tier == 3').sort_values(['absolute_error', 'lga_name', 'crop_year']).reset_index(drop=True)
    selected = []
    for label, pos in [('largest_error', len(ordered)-1), ('median_error', (len(ordered)-1)//2), ('smallest_error', 0)]:
        row = ordered.iloc[pos]; selected.append(dict(selection=label, **row.to_dict()))
        part = features[(features.tier == 3) & (features.lga_name == row.lga_name) & (features.crop_year == row.crop_year) & features.included]
        exp = shap.Explanation(values=part.shap_value.to_numpy(), base_values=float(row.baseline),
            data=part.processed_value.to_numpy(), feature_names=[f'{f}: {names[f]}' for f in part.feature_id])
        shap.plots.waterfall(exp, max_display=12, show=False)
        plt.gcf().set_size_inches(10, 7)
        plt.title(f'{row.lga_name}, {int(row.crop_year)}: {label.replace("_", " ")}\nPrediction {row.predicted:.1f}; observed {row.observed:.1f} kg/ha', fontsize=11)
        savefig(dest / f'T3_waterfall_{label}')
    pd.DataFrame(selected).to_csv(OUT / 'waterfall_selection.csv', index=False)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['check', 'smoke', 'run', 'verify'])
    args = parser.parse_args(); d, folders = inputs()
    print('PASS: accepted main panel, 51 saved outer models and original training-package versions.', flush=True)
    if args.stage == 'check': return
    out = OUT / ('smoke' if args.stage == 'smoke' else 'folds'); out.mkdir(parents=True, exist_ok=True)
    manifest = dict(python=platform.python_version(), packages={n: version(n) for n in ['shap', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'catboost', 'joblib', 'numba', 'matplotlib']},
        input_hash=digest(INPUT), script_hash=digest(Path(__file__)), preprocessing_hash=digest(PROJECT / 'src/tree_models/preprocessing.py'),
        dictionary_hash=digest(PROJECT / 'src/feature_engineering/feature_dictionary.csv'),
        model_hashes={str(f.relative_to(MODELS)): digest(f) for _, _, folder in folders for f in [folder / 'stack.joblib', folder / 'result.json']},
        semantics='interventional; all outer-training rows; raw yield output; saved Ridge coefficients; no refitting')
    stamp = out / 'manifest.json'
    if stamp.exists() and json.loads(stamp.read_text()) != manifest:
        raise ValueError('Inputs, models, code or environment changed. Preserve outputs; use a separate copy.')
    stamp.write_text(json.dumps(manifest, indent=2))
    chosen = folders[:1] if args.stage == 'smoke' else folders
    results = []
    for count, (tier, held, folder) in enumerate(chosen, 1):
        path = out / f'T{tier}_{held.replace(" ", "_")}.json'
        if path.exists():
            value = json.loads(path.read_text())
        elif args.stage == 'verify':
            raise FileNotFoundError(f'Incomplete SHAP run: {path}')
        else:
            value = explain(d, tier, held, folder)
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, allow_nan=True)); tmp.replace(path)
        results.append(value); print(f'{count}/{len(chosen)} T{tier} {held}: checked', flush=True)
    records = pd.DataFrame([x for r in results for x in r['records']])
    features = pd.DataFrame([x for r in results for x in r['features']])
    components = pd.DataFrame([x for r in results for x in r['base_components']])
    close(records.reconstructed.to_numpy(), records.predicted.to_numpy(), 'cached stack additivity')
    keys = ['tier', 'lga_name', 'crop_year']
    if records.duplicated(keys).any(): raise ValueError('Duplicate held-out explanations.')
    sums = features.groupby(keys).shap_value.sum().reset_index(name='sum_shap')
    merged = records.merge(sums, on=keys, validate='one_to_one')
    close((merged.baseline + merged.sum_shap).to_numpy(), merged.predicted.to_numpy(), 'export reconstruction')
    weighted = components.groupby(keys + ['feature_id']).weighted_shap.sum().reset_index(name='weighted')
    paired = features.merge(weighted, on=keys + ['feature_id'], how='left', validate='one_to_one').fillna({'weighted': 0})
    close(paired.shap_value.to_numpy(), paired.weighted.to_numpy(), 'weighted base decomposition')
    if args.stage == 'smoke':
        records.to_csv(out / 'smoke_reconstruction.csv', index=False)
        print('PASS: three real base-model explainers and saved stacking prediction reconstruction. SMOKE ONLY.'); return
    if len(records) != 174 or not (records.groupby('tier').size() == 58).all():
        raise ValueError('Expected 58 held-out explanations per tier.')
    for tier, part in records.groupby('tier'):
        matched = part.merge(d[['lga_name', 'crop_year', 'yield_kg_ha']], on=['lga_name', 'crop_year'], validate='one_to_one')
        close(matched.observed.to_numpy(), matched.yield_kg_ha.to_numpy(), 'export targets')
        for held, fold in part.groupby('lga_name'):
            saved = json.loads((MODELS / f'main_min5_0_20_T{tier}_{held.replace(" ", "_")}' / 'result.json').read_text())
            match = fold.merge(pd.DataFrame(saved['predictions']), on=['lga_name', 'crop_year'], suffixes=('', '_saved'), validate='one_to_one')
            close(match.predicted.to_numpy(), match.predicted_saved.to_numpy(), 'export saved predictions')
    dictionary = pd.read_csv(PROJECT / 'src/feature_engineering/feature_dictionary.csv')
    names = dict(zip(dictionary.feature_id, dictionary.name))
    features['abs_shap'] = features.shap_value.abs()
    importance = features.groupby(['tier', 'feature_id']).agg(mean_abs_shap=('abs_shap', 'mean'),
        mean_signed_shap=('shap_value', 'mean'), included_observations=('included', 'sum')).reset_index()
    importance['name'] = importance.feature_id.map(names)
    importance = importance.sort_values(['tier', 'mean_abs_shap', 'feature_id'], ascending=[True, False, True])
    for filename, frame in [('heldout_reconstruction.csv', records), ('feature_shap_values.csv', features),
                            ('base_shap_components.csv', components), ('global_importance.csv', importance)]:
        frame.to_csv(OUT / filename, index=False)
    (OUT / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'fold_audit.json').write_text(json.dumps([dict(tier=t, held_lga=h, audit=r['audit']) for (t,h,_),r in zip(chosen,results)], indent=2))
    figures(records, features, importance, names)
    (OUT / 'Review_Report.md').write_text('''# Held-out stacking SHAP review
58 observations per tier; 17 outer LGA folds; three tiers. All models were loaded, never refitted.
SHAP explains predictions, not yield causation or predictive accuracy. Limited LOLO/LOYO performance constrains substantive interpretation.
Background: every outer-training observation, equally weighted, transformed with saved fold preprocessing; no held-out rows.
Stack contribution = sum of saved Ridge coefficient times base-model contribution. Fold baseline = Ridge intercept plus the weighted base expectations.
Values are in kg/ha. Fold baselines and fitted functions differ; pooled global importance is descriptive. No SHAP significance tests or causal claims.
Global means include zero contributions for candidates omitted in a fold. Included-observation counts identify availability; omitted features are not demonstrated irrelevant.
Dependence plots use fold-clipped/imputed values, omit excluded feature-fold pairs, and show the three highest mean-absolute attributions within each tier.
Waterfalls use Tier 3 records with largest, lower-median-position and smallest absolute prediction errors; selection is diagnostic, not representative.
Each figure has PNG (400 dpi) and PDF versions. Source values are in feature_shap_values.csv; waterfall selections are recorded separately.
Numerical checks use absolute tolerance 0.001 kg/ha plus relative tolerance 0.000001. Package versions and file hashes are archived.
''')
    files = list(OUT.glob('*.csv')) + list(OUT.glob('*.json')) + [OUT / 'Review_Report.md'] + list((OUT / 'figures').glob('*'))
    archive = OUT / 'SHAP_Main_Panel_Review.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, str(f.relative_to(OUT)))
    print(f'PASS: 174 held-out explanations, coverage, unchanged targets, base/stack additivity and weighted decomposition.\nUpload: {archive}', flush=True)

if __name__ == '__main__': main()
