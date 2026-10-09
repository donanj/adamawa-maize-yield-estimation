"""Run from this directory; Python 3 with numpy, pandas, scipy. No cross-cell imputation."""
from pathlib import Path
import calendar, hashlib, json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
BASE=Path(__file__).resolve().parent; IN=BASE/'inputs'; OUT=BASE/'outputs';OUT.mkdir(exist_ok=True)
KEY=['lga_name','crop_year']; IDS=[f'V{i:02}' for i in range(1,34)]
def read(name): return pd.read_csv(IN/name)
def rename(d):return d.rename(columns={'ADM2_NAME':'lga_name'}).replace({'lga_name':{'Girie':'Girei','Teungo':'Toungo'}})
def smooth(a):
 a=np.asarray(a,float).copy();filled=np.zeros(len(a),bool)
 # Only internal runs <=2; no boundary extrapolation.
 i=0
 while i<len(a):
  if np.isfinite(a[i]): i+=1;continue
  j=i
  while j<len(a) and not np.isfinite(a[j]):j+=1
  if i>0 and j<len(a) and j-i<=2:
   a[i:j]=np.linspace(a[i-1],a[j],j-i+2)[1:-1];filled[i:j]=True
  i=j
 out=np.full(len(a),np.nan);i=0
 while i<len(a):
  if not np.isfinite(a[i]):i+=1;continue
  j=i
  while j<len(a) and np.isfinite(a[j]):j+=1
  if j-i>=5:out[i:j]=savgol_filter(a[i:j],5,2,mode='interp')
  i=j
 return out,filled

def crossing(t,y,threshold,rising):
 pairs=[]
 for i in range(len(y)-1):
  if not np.isfinite(y[i:i+2]).all():continue
  ok=(y[i]<threshold<=y[i+1]) if rising else (y[i]>=threshold>y[i+1])
  if ok:pairs.append(t[i]+(threshold-y[i])/(y[i+1]-y[i])*(t[i+1]-t[i]))
 return (pairs[0] if rising else pairs[-1]) if pairs else np.nan

def integral(t,y,start,end):
 # Require a supported segment spanning BOTH exact fixed-window boundaries.
 good=np.isfinite(y)
 if not good.any():return np.nan
 idx=np.flatnonzero(good)
 cuts=np.split(idx,np.where(np.diff(idx)>1)[0]+1)
 for ix in cuts:
  if len(ix)>1 and t[ix[0]]<=start and t[ix[-1]]>=end:
   tx=np.r_[start,t[ix][(t[ix]>start)&(t[ix]<end)],end]
   yy=np.interp(tx,t[ix],y[ix]);return np.trapz(yy,tx)
 return np.nan

veg=rename(read('vegetation.csv'));lst=rename(read('lst_day.csv'))
v3files=sorted(IN.glob('Adamawa_MOD11A2_day_night_*_v3.csv'))
if v3files:
 assert len(v3files)==5,'Upload all five repaired day/night years together.'
 lst=rename(pd.concat([pd.read_csv(f) for f in v3files],ignore_index=True))
# Buffered exports replace (not append duplicate) May–October inputs when supplied.
bufferfiles=sorted(IN.glob('Adamawa_MOD13Q1_buffer_*.csv'))
if bufferfiles:
 assert len(bufferfiles)==5,'Upload all five buffered years together.'
 veg=rename(pd.concat([pd.read_csv(f) for f in bufferfiles],ignore_index=True))
veg['date']=pd.to_datetime(veg.date);assert not veg.duplicated(KEY+['date']).any()
rows=[];qa=[];curves=[]
for (name,year),g in veg.groupby(KEY):
 g=g.sort_values('date');t=(g.date-pd.Timestamp(year=int(year),month=1,day=1)).dt.days.to_numpy(float)+1
 assert np.allclose(np.diff(t),16),'Expected native 16-day MOD13Q1 timestamp spacing.'
 nd,ni=smooth(g.NDVI_mean);ev,ei=smooth(g.EVI_mean)
 start=pd.Timestamp(year=int(year),month=5,day=1).dayofyear
 end=pd.Timestamp(year=int(year),month=11,day=1).dayofyear
 window=(t>=start)&(t<end);n=nd.copy();n[~window]=np.nan
 r={'lga_name':name,'crop_year':year};r.update({i:np.nan for i in IDS[:9]})
 if np.isfinite(n).any():
  peak=int(np.nanargmax(n));r['V04']=n[peak];r['V09']=ev[peak] if np.isfinite(ev[peak]) else np.nan
  # Left/right baselines are minimum supported within-window values; earliest tied peak.
  left=np.flatnonzero(window & np.isfinite(nd) & (np.arange(len(nd))<peak))
  right=np.flatnonzero(window & np.isfinite(nd) & (np.arange(len(nd))>peak))
  sos=eos=np.nan
  if len(left) and len(right):
   al=nd[peak]-np.min(nd[left]);ar=nd[peak]-np.min(nd[right])
   if al>=.05 and ar>=.05:
    # Do not bridge unresolved segments on either flank.
    li=np.arange(left[0],peak+1);ri=np.arange(peak,right[-1]+1)
    sos=crossing(t[li],nd[li],np.min(nd[left])+.2*al,True)
    eos=crossing(t[ri],nd[ri],np.min(nd[right])+.2*ar,False)
  if np.isfinite(sos) and np.isfinite(eos) and sos<t[peak]<eos:
   r.update(V01=sos,V02=eos,V03=eos-sos)
   if t[peak]-sos>=2:r['V06']=(eos-t[peak])/(t[peak]-sos)
 if np.isfinite(ev[window]).any():r['V07']=np.nanmax(ev[window])
 r['V05']=integral(t,nd,start,end);r['V08']=integral(t,ev,start,end)
 rows.append(r);qa.append({'lga_name':name,'crop_year':year,'ndvi_raw_missing':int(g.NDVI_mean.isna().sum()),'evi_raw_missing':int(g.EVI_mean.isna().sum()),'ndvi_interpolated':int(ni.sum()),'evi_interpolated':int(ei.sum()),'phenology_pair_valid':bool(np.isfinite(r['V01']) and np.isfinite(r['V02'])),'fixed_ndvi_valid':bool(np.isfinite(r['V05'])),'fixed_evi_valid':bool(np.isfinite(r['V08']))})
 for j in range(len(g)):curves.append({'lga_name':name,'crop_year':year,'date':g.date.iloc[j],'ndvi_raw':g.NDVI_mean.iloc[j],'evi_raw':g.EVI_mean.iloc[j],'ndvi_smoothed':nd[j],'evi_smoothed':ev[j],'ndvi_interpolated':ni[j],'evi_interpolated':ei[j]})
features=pd.DataFrame(rows)
lst['date']=pd.to_datetime(lst.date)
assert not lst.duplicated(KEY+['date']).any()
ls=lst.groupby(KEY).LST_day_C_mean.agg(V10='mean',V12='max').reset_index()
# Means are of available native 8-day composites; no daily-extreme interpretation.
features=features.merge(ls,on=KEY,validate='one_to_one');features['V11']=np.nan
nightfiles=v3files if v3files else sorted(IN.glob('Adamawa_MOD11A2_night_*_v2.csv'))
if not nightfiles:nightfiles=sorted(IN.glob('Adamawa_MOD11A2_night_*.csv'))
if nightfiles:
 assert len(nightfiles)==5,'Upload all five nighttime years together.'
 night=rename(pd.concat([pd.read_csv(f) for f in nightfiles],ignore_index=True))
 assert not night.duplicated(KEY+['date']).any()
 ns=night.groupby(KEY).LST_night_C_mean.mean().rename('night').reset_index()
 features=features.merge(ns,on=KEY,validate='one_to_one');features['V11']=features.pop('night')
rain=rename(read('rainfall.csv'));assert not rain.duplicated(KEY+['month']).any()
r=rain.groupby(KEY).rainfall_mm.agg(V13=lambda x:x.sum(min_count=6),V14='max',V15=lambda x:int((x<50).sum())).reset_index()
assert rain.groupby(KEY).size().eq(6).all()
features=features.merge(r,on=KEY,validate='one_to_one');features['V16']=np.nan
auxfile=IN/'Adamawa_feature_auxiliary_v1.csv';aux=None
if auxfile.exists():
 aux=rename(pd.read_csv(auxfile));assert len(aux)==210 and not aux.duplicated(KEY+['month']).any()
 assert aux.month.isin([3,4]).all() and aux.rainfall_mm.notna().all()
 assert aux.source_images.eq(aux.expected_days).all() and aux.valid_area_fraction.between(0,1+1e-9).all()
 assert aux.latitude_deg.between(7,12).all() and aux.groupby('lga_name').latitude_deg.nunique().eq(1).all()
 rainall=pd.concat([rain[KEY+['month','rainfall_mm']],aux[KEY+['month','rainfall_mm']]])
 onset=[]
 for (name,year),g in rainall.groupby(KEY):
  s=g.set_index('month').rainfall_mm;assert set(s.index)==set(range(3,11))
  hits=[m for m in range(5,11) if s.loc[[m-2,m-1,m]].sum()>25]
  onset.append({'lga_name':name,'crop_year':year,'onset':hits[0] if hits else np.nan})
 features=features.merge(pd.DataFrame(onset),on=KEY,validate='one_to_one');features['V16']=features.pop('onset')
era=rename(read('era5.csv'));assert not era.duplicated(KEY+['month']).any()
c=[];eto_audit=[]
terrain=read('terrain.csv')
def es(t):return .6108*np.exp(17.27*t/(t+237.3))
for (name,year),g in era.groupby(KEY):
 g=g.sort_values('month');assert set(g.month)==set(range(4,11))
 season=g[g.month>=5];days=season.expected_days.to_numpy();total=days.sum()
 row={'lga_name':name,'crop_year':year,'V17':np.average(season.Tmax_daily_mean_C,weights=days),'V18':np.average(season.Tmin_daily_mean_C,weights=days),'V19':np.sum(days*np.maximum(season.Tmean_C.to_numpy()-10,0)),'V22':season.solar_MJ_m2_month.sum()*1e6/(total*86400),'V20':np.nan}
 if aux is not None:
  lat=aux.loc[(aux.lga_name==name)&(aux.crop_year==year),'latitude_deg'].iloc[0]*np.pi/180
  z=terrain.loc[terrain.lga_name==name,'V32_elevation_m'].iloc[0];etolist=[]
  prev=float(g.loc[g.month==4,'Tmean_C'].iloc[0])
  for _,s in season.iterrows():
   month=int(s.month);num=int(s.expected_days);dates=pd.date_range(f'{int(year)}-{month:02d}-01',periods=num)
   j=dates.dayofyear.to_numpy();dr=1+.033*np.cos(2*np.pi*j/365);decl=.409*np.sin(2*np.pi*j/365-1.39)
   ws=np.arccos(np.clip(-np.tan(lat)*np.tan(decl),-1,1))
   ra=np.mean((24*60/np.pi)*.0820*dr*(ws*np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl)*np.sin(ws)))
   rso=(.75+2e-5*z)*ra;ratio=s.solar_MJ_m2_day/rso;bounded=np.clip(ratio,.3,1)
   tx=s.Tmax_daily_mean_C;tn=s.Tmin_daily_mean_C;tm=(tx+tn)/2
   ea=es(s.dewpoint_C);sat=(es(tx)+es(tn))/2;vpd=max(sat-ea,0)
   delta=4098*es(tm)/(tm+237.3)**2;gamma=.000665*s.pressure_kPa
   u2=s.wind10_ms*4.87/np.log(67.8*10-5.42)
   rns=.77*s.solar_MJ_m2_day
   rnl=4.903e-9*((tx+273.16)**4+(tn+273.16)**4)/2*(.34-.14*np.sqrt(ea))*(1.35*bounded-.35)
   rn=rns-rnl;heat=.14*(s.Tmean_C-prev);prev=s.Tmean_C
   eto=(.408*delta*(rn-heat)+gamma*900/(tm+273)*u2*vpd)/(delta+gamma*(1+.34*u2))
   etolist.append(eto*num)
   eto_audit.append({'lga_name':name,'crop_year':year,'month':month,'Ra_MJ_m2_day':ra,'Rs_Rso_raw':ratio,'Rs_Rso_used':bounded,'ea_kPa':ea,'es_kPa':sat,'VPD_kPa':vpd,'u2_ms':u2,'Rn_MJ_m2_day':rn,'G_MJ_m2_day':heat,'ETo_mm_day':eto,'ETo_mm_month':eto*num})
  row['V20']=sum(etolist)
 c.append(row)
features=features.merge(pd.DataFrame(c),on=KEY,validate='one_to_one');features['V21']=features.V13-features.V20;features['V27']=np.nan
for col in IDS:
 if col not in features:features[col]=np.nan
qa=pd.DataFrame(qa);pd.DataFrame(curves).to_csv(OUT/'smoothed_vegetation_audit.csv',index=False);qa.to_csv(OUT/'satellite_feature_audit.csv',index=False)
if eto_audit:pd.DataFrame(eto_audit).to_csv(OUT/'eto_monthly_audit.csv',index=False)
for depth in ['0_20','20_50']:
 soil=read('soil_'+depth+'.csv').rename(columns={f'V{i:02}_{depth}':f'V{i:02}' for i in [23,24,25,26,28,29,30,31]})
 static=soil.merge(terrain,on='lga_name',validate='one_to_one').rename(columns={'V32_elevation_m':'V32','V33_slope_percent':'V33'})
 dynamic=features.drop(columns=[f'V{i:02}' for i in [23,24,25,26,28,29,30,31,32,33]])
 full=dynamic.merge(static,on='lga_name',validate='many_to_one')
 assert len(full)==105 and not full.duplicated(KEY).any()
 full.to_csv(OUT/('features_105_cells_soil_'+depth+'.csv'),index=False)
 for panel in ['main_min5','all_cells','exclude_above_10000_min5']:
  y=read(panel+'.csv');y['lga_name']=y.ADM2_NAME.replace({'Girie':'Girei','Teungo':'Toungo'});y=y.rename(columns={'year':'crop_year'})
  joined=y.merge(full,on=KEY,how='left',validate='one_to_one')
  assert len(joined)==len(y) and joined.V13.notna().all()
  joined.to_csv(OUT/(panel+'_features_soil_'+depth+'.csv'),index=False)
  joined[IDS].isna().sum().rename('missing_rows').to_csv(OUT/(panel+'_missingness_soil_'+depth+'.csv'))
summary=features[[i for i in IDS if int(i[1:])<=22]].agg(['count','min','max']).T.reset_index(names='feature_id');summary.to_csv(OUT/'dynamic_feature_summary.csv',index=False)
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(IN.glob('*.csv'))};(OUT/'input_hashes.json').write_text(json.dumps(manifest,indent=2))
print('Built 105 cells per soil depth; all three yield variants retained. No target-dependent processing or cross-cell imputation.')
print(summary.to_string(index=False))
