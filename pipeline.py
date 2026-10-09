"""Stage-by-stage entry point. No expensive model job starts without its stage command."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,shutil,importlib.metadata as md
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'src';FE=SRC/'feature_engineering'
def run(script,*args):subprocess.run([sys.executable,str(script),*map(str,args)],cwd=ROOT,check=True)
def check():
 if sys.version_info<(3,11):raise RuntimeError('Use Python 3.11 or newer.')
 print('Python:',sys.version.split()[0]);print('Interpreter:',sys.executable)
 for name in ['numpy','pandas','scipy','scikit-learn','optuna','xgboost','catboost','joblib']:print(name,md.version(name))
 expected=json.loads((ROOT/'config/input_hashes.json').read_text())
 for name,h in expected.items():
  file=FE/'inputs'/name
  if not file.exists():raise FileNotFoundError(f'Missing {file}; use the complete project ZIP, not just the code-only GitHub clone.')
  if hashlib.sha256(file.read_bytes()).hexdigest()!=h:raise ValueError(f'Input changed: {name}. Record deliberate data revisions before running.')
 print('PASS: packages available and all',len(expected),'source-input hashes match.')
def prepare():
 for panel in ['main_min5','all_cells']:
  for depth in ['0_20','20_50']:
   source=FE/'outputs'/f'{panel}_features_soil_{depth}.csv'
   if not source.exists():raise FileNotFoundError('Run python pipeline.py features first.')
   for model_dir in ['linear_benchmarks','tree_models']:
    dest=SRC/model_dir/'inputs'/f'{panel}_full_candidates_{depth}.csv';dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest)
 print('PASS: rebuilt full candidate panels copied to model stages; training-fold preprocessing remains unfitted.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['check','features','prepare','linear','smoke','trees','verify-trees','yield']);a=p.parse_args()
 if a.stage=='check':check()
 elif a.stage=='features':
  (FE/'outputs').mkdir(exist_ok=True);run(FE/'audit_lst_v3.py');run(FE/'verify_features.py')
 elif a.stage=='prepare':prepare()
 elif a.stage=='linear':
  run(SRC/'linear_benchmarks/run_benchmarks.py')
  dest=SRC/'tree_models/linear_reference';dest.mkdir(exist_ok=True)
  for name in ['pooled_metrics.csv','oof_predictions.csv']:shutil.copy2(SRC/'linear_benchmarks/outputs'/name,dest/name)
  shutil.copy2(SRC/'linear_benchmarks/outputs/shared_LOLO_splits.csv',SRC/'tree_models/shared_LOLO_splits.csv')
 elif a.stage=='smoke':run(SRC/'tree_models/smoke_test.py')
 elif a.stage=='trees':run(SRC/'tree_models/run_models.py')
 elif a.stage=='verify-trees':run(SRC/'tree_models/verify_results.py');run(SRC/'tree_models/summarise_results.py')
 elif a.stage=='yield':run(SRC/'yield_panel/build_yield_panel.py','--input-dir',ROOT/'data/private/lsms','--output-dir',SRC/'yield_panel/outputs')
