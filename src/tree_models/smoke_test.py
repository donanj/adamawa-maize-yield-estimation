"""Tiny installation check; outputs are isolated from research evaluation."""
from pathlib import Path
import shutil
import pandas as pd
import run_models as models
source=Path(__file__).resolve().parent
models.ROOT=source/'smoke';models.TRIALS=2
for folder in ['inputs','outputs','models','studies']:(models.ROOT/folder).mkdir(parents=True,exist_ok=True)
file='main_min5_full_candidates_0_20.csv'
shutil.copy2(source/'inputs'/file,models.ROOT/'inputs'/file)
held=sorted(pd.read_csv(models.ROOT/'inputs'/file).lga_name.unique())[0]
print(models.task(('main_min5','0_20',1,'RF',held)))
print('PASS: one RF outer fold, two tuning trials. Installation check only; not a thesis result.')
