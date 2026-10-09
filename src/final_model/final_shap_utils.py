"""Verified interventional SHAP utilities for saved final full-panel models."""
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
INPUT = PROJECT / 'src/tree_models/inputs/main_min5_full_candidates_0_20.csv'
OUT = PROJECT / 'src/final_model/outputs/research/shap'
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
    for values, label in [(bg, 'background'), (x, 'full-panel fitted')]:
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
    for data, label in [(bg, 'background'), (x, 'full-panel fitted')]:
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

def explain(d, tier, held, folder):
    result = json.loads((folder / 'result.json').read_text())
    tr = np.arange(len(d))
    te = np.arange(len(d))
    if result['train_rows'] != tr.tolist() or result['explanation_rows'] != te.tolist():
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
            close(bg, common_bg, 'common background'); close(x, common_test, 'common full-panel fitted inputs')
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
                base_rows.append(dict(tier=tier, lga_name=str(d.iloc[i].lga_name), crop_year=int(d.iloc[i].crop_year),
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
        if p['lga_name'] != str(d.iloc[i].lga_name) or p['crop_year'] != int(d.iloc[i].crop_year) or p['observed'] != float(d.iloc[i].yield_kg_ha):
            raise ValueError('Saved observation keys or targets differ.')
    close(baseline + total.sum(axis=1), predicted, 'stacking SHAP')
    records, features = [], []
    for q, i in enumerate(te):
        records.append(dict(tier=tier, lga_name=str(d.iloc[i].lga_name), crop_year=int(d.iloc[i].crop_year),
            observed=float(d.iloc[i].yield_kg_ha), predicted=float(predicted[q]), baseline=baseline,
            reconstructed=float(baseline + total[q].sum()), absolute_error=float(abs(predicted[q] - d.iloc[i].yield_kg_ha))))
        for k, f in enumerate(allcols):
            included = f in common_cols
            features.append(dict(tier=tier, lga_name=str(d.iloc[i].lga_name), crop_year=int(d.iloc[i].crop_year), feature_id=f,
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
        plt.xlabel('Mean absolute full-panel fitted SHAP contribution (kg/ha)')
        plt.title(f'Final full-panel stacking Tier {tier}: model attribution')
        savefig(dest / f'T{tier}_global_importance')
        for feature in top.iloc[::-1].feature_id.head(3):
            part = features.query('tier == @tier and feature_id == @feature and included').copy()
            plt.figure(figsize=(7, 4.5))
            sc = plt.scatter(part.processed_value, part.shap_value, c=part.crop_year, cmap='viridis', s=36)
            plt.axhline(0, color='grey', linewidth=.7)
            plt.xlabel(f'{feature}: {names[feature]} (model-processed value)')
            plt.ylabel('Full-panel fitted SHAP contribution (kg/ha)')
            plt.title(f'Final full-panel Tier {tier}: descriptive dependence')
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
        plt.title(f'Final fitted model — {row.lga_name}, {int(row.crop_year)}: {label.replace("_", " ")}\nPrediction {row.predicted:.1f}; observed {row.observed:.1f} kg/ha', fontsize=11)
        savefig(dest / f'T3_waterfall_{label}')
    pd.DataFrame(selected).to_csv(OUT / 'waterfall_selection.csv', index=False)

