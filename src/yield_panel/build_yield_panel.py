"""Build Adamawa yield panels; input archives remain unchanged."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from source_audit import process,read,code,positive,assert_unique

FACTORS={1:.00016,2:.004,3:.00016,4:.0667,5:.4,6:1.,7:.0001}

def areas(path,w):
    with zipfile.ZipFile(path) as z:
        a=read(z,f'sect11a1_plantingw{w}.csv');a=a[code(a.state)==2].copy()
        ph=read(z,f'secta1_harvestw{w}.csv');ph=ph[code(ph.state)==2].copy()
        assert_unique(a,['hhid','plotid'],'planting');assert_unique(ph,['hhid','plotid'],'postharvest')
        gf={1:'s11aq4d',2:'s11aq4c',3:'s11aq4c',4:'s11aq4c',5:'s11mq3'}[w]
        a['plant_gps']=positive(a[gf])/10000
        if w<5:
            q='s11aq4aa' if w==4 else 's11aq4a'
            a['reported_ha']=positive(a[q])*code(a.s11aq4b).map(FACTORS)
        else:a['reported_ha']=positive(a.SR_hect)
        pf={1:'sa1q9d',2:'sa1q9c',3:'sa1q9c',4:'prefilled_gps_area',5:'sa1q1c'}[w]
        ph['ph_gps']=positive(ph[pf])/10000
        x=a[['hhid','plotid','plant_gps','reported_ha']].merge(ph[['hhid','plotid','ph_gps']],on=['hhid','plotid'],how='outer',validate='one_to_one')
        # W4/W5 postharvest fields are carried-forward GPS, not verified fresh measurements.
        if w<=3:
            x['gps_ha']=x.ph_gps.fillna(x.plant_gps)
            x['gps_source']=np.where(x.ph_gps.notna(),'postharvest_GPS','planting_GPS')
        else:
            x['gps_ha']=x.plant_gps.fillna(x.ph_gps)
            x['gps_source']=np.where(x.plant_gps.notna(),'planting_GPS','postharvest_carried_GPS')
        return x

def aggregate(d,variant,minimum,bilateral=False):
    x=d[d.eligible].copy()
    if bilateral:
        low=x.groupby('wave').yield_raw.transform(lambda v:v.quantile(.01));x['yield_clean']=x.yield_raw.clip(lower=low)
    else:x['yield_clean']=x.yield_raw
    high=x.groupby('wave').yield_raw.transform(lambda v:v.quantile(.99))
    x['yield_clean']=x.yield_clean.clip(upper=high)
    x['aggregation_weight']=x.area_ha*x.survey_weight_audit
    rows=[]
    for keys,g in x.groupby(['wave','year','ADM2_NAME']):
        n=len(g)
        if n<minimum:continue
        weights=g.aggregation_weight.to_numpy();y=g.yield_clean.to_numpy()
        estimate=float(np.dot(weights,y)/weights.sum())
        assert np.isfinite(estimate) and y.min()-1e-8<=estimate<=y.max()+1e-8
        rows.append(dict(zip(['wave','year','ADM2_NAME'],keys),variant=variant,yield_kg_ha=estimate,eligible_plots=n,households=g.hhid.nunique(),total_plot_area_ha=g.area_ha.sum(),reported_area_plots=int((g.area_source=='reported_area').sum()),incomplete_harvest_plots=int((g.harvest_complete_code_audit==2).sum()),extreme_raw_yield_plots=int((g.yield_raw>10000).sum()),winsorised_plots=int((g.yield_clean!=g.yield_raw).sum()),effective_plot_weight_count=float(weights.sum()**2/(weights**2).sum())))
    return pd.DataFrame(rows),x

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=Path('outputs'));args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    frames=[];prov=[]
    for w in range(1,6):
        files=list(args.input_dir.glob(f'NGA_*-W{w}_*_CSV.zip'));assert len(files)==1
        h,_,_,s=process(files[0],w);a=areas(files[0],w)
        h=h.merge(a,on=['hhid','plotid'],how='left',validate='one_to_one')
        h['area_ha']=h.gps_ha.fillna(h.reported_ha)
        h['area_source']=np.where(h.gps_ha.notna(),h.gps_source,np.where(h.reported_ha.notna(),'reported_area','missing'))
        h['yield_raw']=h.harvest_kg_audit/h.area_ha
        h['valid_measurements']=h.area_ha.gt(0)&h.harvest_kg_audit.ge(0)&h.survey_weight_audit.gt(0)&np.isfinite(h.yield_raw)&h.ADM2_NAME.notna()
        h['eligible']=h.valid_measurements&h.harvest_complete_code_audit.ne(2)
        frames.append(h);prov.append(s)
    d=pd.concat(frames,ignore_index=True)
    definitions=[('main_min5',5,False,False,False),('all_cells',1,False,False,False),('gps_only_min5',5,True,False,False),('including_incomplete_min5',5,False,True,False),('bilateral_winsor_min5',5,False,False,True),('exclude_above_10000_min5',5,False,False,False)]
    panels=[]
    for name,m,gps,inc,bilateral in definitions:
        x=d.copy()
        if gps:x['eligible']=x.eligible&x.gps_ha.notna()
        if inc:x['eligible']=x.valid_measurements
        if name.startswith('exclude_above'):x['eligible']=x.eligible&x.yield_raw.le(10000)
        panel,records=aggregate(x,name,m,bilateral);panels.append(panel)
        panel.to_csv(args.output_dir/f'{name}.csv',index=False)
        if name=='main_min5':
            # Local audit contains survey linkage IDs, not for modelling or public dissemination.
            records.to_csv(args.output_dir/'private_plot_audit.csv',index=False)
    summary=pd.DataFrame([{'variant':p.variant.iloc[0] if len(p) else 'empty','cells':len(p),'LGAs':p.ADM2_NAME.nunique(),'plots':int(p.eligible_plots.sum())} for p in panels]);summary.to_csv(args.output_dir/'panel_summary.csv',index=False)
    cut=d[d.eligible].groupby('wave').yield_raw.agg(eligible_plots='size',q99=lambda x:x.quantile(.99),maximum='max');cut.to_csv(args.output_dir/'winsorisation_thresholds.csv')
    audit=d.groupby('wave').agg(maize_records=('wave','size'),valid_measurements=('valid_measurements','sum'),eligible_complete_or_unknown=('eligible','sum'),explicit_incomplete=('harvest_complete_code_audit',lambda x:int((x==2).sum())),reported_fallback=('area_source',lambda x:int((x=='reported_area').sum())))
    audit.to_csv(args.output_dir/'wave_cleaning_summary.csv')
    (args.output_dir/'provenance.json').write_text(json.dumps(prov,indent=2))
    print(summary.to_string(index=False));print(cut.to_string());print(audit.to_string())
if __name__=='__main__':main()
