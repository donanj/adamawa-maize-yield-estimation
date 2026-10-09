"""Predict with a saved OUTER-FOLD evaluation model (not a full-data model)."""
import joblib
import pandas as pd
def predict_saved(model_path,frame):
 saved=joblib.load(model_path)
 prep=saved['preprocessing']
 x=prep.transform(frame)*prep.scale+prep.center
 return saved['model'].predict(x)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('csv');p.add_argument('output');a=p.parse_args()
 frame=pd.read_csv(a.csv);frame['predicted_kg_ha']=predict_saved(a.model,frame);frame.to_csv(a.output,index=False)
