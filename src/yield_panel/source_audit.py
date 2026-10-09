"""Reproducible, provisional Adamawa maize source audit; NOT a final model dataset.
Run: python audit_lsms_archives.py --input-dir PATH_TO_ZIPS --output-dir audit_outputs
Requires pandas and numpy. Reads archives without extracting or modifying them.
"""
import argparse, hashlib, json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

LGAS = dict(zip(range(201,222), ['Demsa','Fufore','Ganye','Girei','Gombi','Guyuk','Hong','Jada','Lamurde','Madagali','Maiha','Mayo-Belwa','Michika','Mubi North','Mubi South','Numan','Shelleng','Song','Toungo','Yola North','Yola South']))
YEARS = {1:2010,2:2012,3:2015,4:2018,5:2023}

def code(s):
    return pd.to_numeric(s.astype(str).str.extract(r'^\s*(\d+)')[0], errors='coerce')

def numeric(s):
    v = pd.to_numeric(s, errors='coerce')
    return v.where(np.isfinite(v) & (v.abs()<1e100))

def positive(s):
    v=numeric(s)
    return v.where(v>0)

def member(z, filename):
    names=[n for n in z.namelist() if n.rsplit('/',1)[-1].lower()==filename.lower()]
    if len(names)!=1:
        raise ValueError(f'Expected exactly one {filename}, found {len(names)}')
    return names[0]

def read(z, filename):
    return pd.read_csv(z.open(member(z,filename)),low_memory=False)

def assert_unique(d, keys, name):
    if d[keys].isna().any().any():
        raise ValueError(f'{name}: missing key')
    if d.duplicated(keys).any():
        raise ValueError(f'{name}: duplicate {keys}; resolve before continuing')

def process(path, wave):
    provenance={'wave':wave,'archive':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with zipfile.ZipFile(path) as z:
        h=read(z,f'secta3{ "" if wave<=2 else "i" }_harvestw{wave}.csv')
        crop_field='sa3q2' if wave==1 else 'cropcode'
        h=h[(code(h.state)==2)&code(h[crop_field]).isin([1080,1081,1082])].copy()
        assert_unique(h,['hhid','plotid'],'Adamawa maize harvest')
        h['crop_code_audit']=code(h[crop_field])
        h['lga_code_audit']=code(h.lga)
        a=read(z,f'sect11a1_plantingw{wave}.csv')
        a=a[code(a.state)==2].copy()
        assert_unique(a,['hhid','plotid'],'planting areas')
        if wave<=3:
            area_field='s11aq4d' if wave==1 else 's11aq4c'
            a['gps_area_sqm']=positive(a[area_field])
        elif wave==4:
            a['gps_area_sqm']=positive(a['s11aq4c'])
        else:
            # Wave 5 prior-visit GPS areas are in the post-harvest plot roster.
            ph=read(z,'secta1_harvestw5.csv');ph=ph[code(ph.state)==2].copy()
            assert_unique(ph,['hhid','plotid'],'post-harvest areas')
            a=ph
            a['gps_area_sqm']=positive(a['sa1q1c'])
        keep=['hhid','plotid','gps_area_sqm']
        h=h.merge(a[keep],on=['hhid','plotid'],how='left',validate='one_to_one',indicator='area_join')
        h['area_ha_audit']=h['gps_area_sqm']/10000
        h['area_source_audit']='planting_GPS' if wave<=4 else 'postharvest_prior_visit_GPS'
        if wave==4:
            ph=read(z,'secta1_harvestw4.csv');ph=ph[code(ph.state)==2].copy()
            assert_unique(ph,['hhid','plotid'],'post-harvest areas')
            ph['prefilled_area']=positive(ph.prefilled_gps_area)
            h=h.merge(ph[['hhid','plotid','prefilled_area']],on=['hhid','plotid'],how='left',validate='one_to_one')
            h['area_ha_audit']=h['area_ha_audit'].fillna(h.prefilled_area/10000)
        if wave==3:
            # Official W3 dictionary identifies sa1q9c as GPS square metres.
            ph=read(z,'secta1_harvestw3.csv');ph=ph[code(ph.state)==2].copy()
            assert_unique(ph,['hhid','plotid'],'post-harvest GPS areas')
            ph['new_gps_ha']=positive(ph.sa1q9c)/10000
            h=h.merge(ph[['hhid','plotid','new_gps_ha']],on=['hhid','plotid'],how='left',validate='one_to_one')
            use_new=h['new_gps_ha'].notna()
            h.loc[use_new,'area_ha_audit']=h.loc[use_new,'new_gps_ha']
            h.loc[use_new,'area_source_audit']='postharvest_GPS'
        # Deliberately do not assume units for other unverified area fields.
        # Farmer-area fallbacks and primary-crop eligibility require a separate approved protocol.
        if wave<=2:
            qty_field='sa3q6a' if wave==1 else 'sa3q6a1'
            unit_field='sa3q6b' if wave==1 else 'sa3q6a2'
            conv=read(z,f'w{wave}agnsconversion.csv')
            cc='agcropid' if wave==1 else 'cropcode'
            conv=conv[code(conv[cc])==1080][['nscode','conversion']].drop_duplicates()
            assert_unique(conv,['nscode'],'maize conversion factors')
            lookup=dict(zip(code(conv.nscode),positive(conv.conversion)))
            # Questionnaire standard codes: kg=1, grams=2; never treat litre=3 as tonnes.
            lookup.update({1:1.0,2:0.001})
            h['conversion_audit']=code(h[unit_field]).map(lookup)
            # Specific grain/cob codes are not silently mapped to generic nonstandard factors.
            generic_nonstd=(h.crop_code_audit!=1080)&~code(h[unit_field]).isin([1,2])
            h.loc[generic_nonstd,'conversion_audit']=np.nan
            h['harvest_kg_audit']=numeric(h[qty_field])*h.conversion_audit
        elif wave==3:
            conv=read(z,'ag_conv_w3.csv')
            conv=conv[code(conv.crop_cd).isin([1080,1081,1082])][['crop_cd','unit_cd','conv_NE_2']].drop_duplicates()
            assert_unique(conv,['crop_cd','unit_cd'],'NE harvest conversion factors')
            conv=conv.rename(columns={'crop_cd':'crop_code_audit','unit_cd':'unit_code_audit'})
            h['unit_code_audit']=code(h.sa3iq6ii)
            h=h.merge(conv,on=['crop_code_audit','unit_code_audit'],how='left',validate='many_to_one')
            h['conversion_audit']=positive(h.conv_NE_2)
            h['harvest_kg_audit']=numeric(h.sa3iq6i)*h.conversion_audit
        else:
            quantity='sa3iq6i' if wave==4 else 'sa3iq9a'
            conversion='sa3iq6_conv' if wave==4 else 'sa3iq9_conv'
            h['conversion_audit']=positive(h[conversion])
            h['harvest_kg_audit']=numeric(h[quantity])*h.conversion_audit
        weights=read(z,f'secta_harvestw{wave}.csv')
        weights=weights[code(weights.state)==2].copy()
        assert_unique(weights,['hhid'],'household weights')
        weight_field=f'wt_wave{wave}' if wave<5 else 'wt_cross_wave5'
        weights['survey_weight_audit']=positive(weights[weight_field])
        h=h.merge(weights[['hhid','survey_weight_audit']],on='hhid',how='left',validate='many_to_one')
        h['weight_field_audit']=weight_field
        h['ADM2_NAME']=h.lga_code_audit.map(LGAS)
        h['wave']=f'W{wave}';h['year']=YEARS[wave]
        completed_field='sa3iq11' if wave==5 else ('sa3iq6b' if wave>=3 else None)
        h['harvest_complete_code_audit']=code(h[completed_field]) if completed_field else np.nan
        h['missing_or_invalid_area']=h.area_ha_audit.isna()
        h['missing_or_invalid_harvest']=h.harvest_kg_audit.isna()|(h.harvest_kg_audit<0)
        h['missing_or_invalid_weight']=h.survey_weight_audit.isna()
        valid=~(h.missing_or_invalid_area|h.missing_or_invalid_harvest)
        h['yield_raw_kg_ha_audit']=(h.harvest_kg_audit/h.area_ha_audit).where(valid)
        h['technical_candidate']=valid&np.isfinite(h.yield_raw_kg_ha_audit)&h.ADM2_NAME.notna()
        h['weighted_technical_candidate']=h.technical_candidate&~h.missing_or_invalid_weight
        h['extreme_yield_review']=h.yield_raw_kg_ha_audit>10000
        h['zero_yield_review']=h.yield_raw_kg_ha_audit==0
        # No outliers dropped, no winsorisation applied, no missing harvest turned into zero.
        provenance.update({'crop_field':crop_field,'weight_field':weight_field,'files_used':[n for n in z.namelist() if n.endswith('.csv') and any(k in n for k in ['secta3','sect11a1','secta_harvest','conversion','ag_conv','secta1_harvest'])]})
        rows=[]
        for lga,name in LGAS.items():
            x=h[h.lga_code_audit==lga];v=x[x.technical_candidate];wv=x[x.weighted_technical_candidate]
            rows.append({'wave':f'W{wave}','year':YEARS[wave],'ADM2_NAME':name,'harvest_records':len(x),'technical_candidate_plots':len(v),'weighted_candidate_plots':len(wv),'candidate_households':v.hhid.nunique(),'extreme_yield_records':int(x.extreme_yield_review.sum()),'has_any_candidate':len(v)>0,'meets_five_plot_count_only':len(v)>=5})
        summary={'wave':f'W{wave}','year':YEARS[wave],'harvest_maize_records':len(h),'code_1080_records':int((h.crop_code_audit==1080).sum()),'code_1082_records':int((h.crop_code_audit==1082).sum()),'technical_candidate_plots':int(h.technical_candidate.sum()),'weighted_candidate_plots':int(h.weighted_technical_candidate.sum()),'missing_invalid_area':int(h.missing_or_invalid_area.sum()),'missing_invalid_harvest':int(h.missing_or_invalid_harvest.sum()),'missing_invalid_weight':int(h.missing_or_invalid_weight.sum()),'extreme_yield_records':int(h.extreme_yield_review.sum()),'LGAs_any_candidate':sum(r['has_any_candidate'] for r in rows),'LGAs_five_plot_count_only':sum(r['meets_five_plot_count_only'] for r in rows)}
        outcols=['wave','year','ADM2_NAME','lga_code_audit','hhid','plotid','crop_code_audit','harvest_complete_code_audit','area_ha_audit','area_source_audit','harvest_kg_audit','conversion_audit','survey_weight_audit','weight_field_audit','yield_raw_kg_ha_audit','missing_or_invalid_area','missing_or_invalid_harvest','missing_or_invalid_weight','technical_candidate','weighted_technical_candidate','extreme_yield_review','zero_yield_review']
        return h[outcols],summary,rows,provenance

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,default=Path('audit_outputs'))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    frames=[];summaries=[];counts=[];provenance=[]
    for wave in range(1,6):
        files=sorted(args.input_dir.glob(f'NGA_*-W{wave}_*_CSV.zip'))
        if len(files)!=1:raise ValueError(f'Wave {wave}: expected one archive, got {len(files)}')
        records,s,c,p=process(files[0],wave);frames.append(records);summaries.append(s);counts.extend(c);provenance.append(p)
    pd.concat(frames,ignore_index=True).to_csv(args.output_dir/'candidate_record_audit.csv',index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir/'wave_audit_summary.csv',index=False)
    pd.DataFrame(counts).to_csv(args.output_dir/'lga_wave_candidate_counts.csv',index=False)
    (args.output_dir/'source_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    print('PROVISIONAL AUDIT ONLY: crop eligibility, harvest completeness, area fallback, and outlier review are unresolved.')
if __name__=='__main__':main()
