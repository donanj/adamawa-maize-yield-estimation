// MOD11A2 zero-QC mask repair: BOTH day and night. Five tasks.
var CREATE_EXPORTS=false;
var YEARS=[2010,2012,2015,2018,2023];
var regions=ee.FeatureCollection('FAO/GAUL/2015/level2')
 .filter(ee.Filter.eq('ADM0_NAME','Nigeria')).filter(ee.Filter.eq('ADM1_NAME','Adamawa'))
 .map(function(f){var n=ee.String(f.get('ADM2_NAME'));return f.set({ADM2_NAME_original:n,
  ADM2_NAME:ee.Algorithms.If(n.equals('Girie'),'Girei',ee.Algorithms.If(n.equals('Teungo'),'Toungo',n))});});
function prepare(im,period){
 var raw=im.select(period==='day'?'LST_Day_1km':'LST_Night_1km');
 var originalQC=im.select(period==='day'?'QC_Day':'QC_Night');
 // Fill QC zero ONLY within valid, non-fill LST support. Never fill temperature.
 var lstValid=raw.mask().and(raw.gt(0));
 var qc=originalQC.unmask(0).updateMask(lstValid);
 var strict=qc.bitwiseAnd(3).eq(0).and(qc.rightShift(2).bitwiseAnd(3).eq(0))
  .and(qc.rightShift(6).bitwiseAnd(3).eq(0));
 var value=raw.multiply(.02).subtract(273.15).updateMask(lstValid).updateMask(strict)
  .rename('LST_'+period+'_C');
 var recovered=originalQC.mask().unmask(0).eq(0).and(lstValid).and(strict)
  .rename(period+'_qc_zero_restored').selfMask();
 return {value:value,recovered:recovered};
}
function imageRows(obj,year,fc){
 var im=ee.Image(obj);var day=prepare(im,'day');var night=prepare(im,'night');
 var values=day.value.addBands(night.value);
 var recovered=day.recovered.addBands(night.recovered);
 var proj=im.select('LST_Day_1km').projection();
 return fc.map(function(f){
  var mean=values.reduceRegion({reducer:ee.Reducer.mean(),geometry:f.geometry(),crs:proj,scale:1000,tileScale:4,maxPixels:1e9});
  var count=values.reduceRegion({reducer:ee.Reducer.count(),geometry:f.geometry(),crs:proj,scale:1000,tileScale:4,maxPixels:1e9});
  var recovery=recovered.reduceRegion({reducer:ee.Reducer.count(),geometry:f.geometry(),crs:proj,scale:1000,tileScale:4,maxPixels:1e9});
  return ee.Feature(null,{ADM2_NAME:f.get('ADM2_NAME'),ADM2_NAME_original:f.get('ADM2_NAME_original'),
   ADM2_CODE:f.get('ADM2_CODE'),crop_year:year,date:ee.Date(im.get('system:time_start')).format('YYYY-MM-dd'),
   LST_day_C_mean:mean.get('LST_day_C'),LST_night_C_mean:mean.get('LST_night_C'),
   LST_day_C_count:count.get('LST_day_C'),LST_night_C_count:count.get('LST_night_C'),
   day_qc_zero_restored_count:recovery.get('day_qc_zero_restored'),
   night_qc_zero_restored_count:recovery.get('night_qc_zero_restored'),
   product:'MODIS/061/MOD11A2',scale_m:1000,
   qa_policy:'mandatory_0_data_0_error_le1K_QC_zero_restored_only_with_valid_LST',
   export_version:'day_night_QCmask_repaired_v3'});
 });
}
function table(year,fc){var c=ee.ImageCollection('MODIS/061/MOD11A2')
 .filterDate(ee.Date.fromYMD(year,5,1),ee.Date.fromYMD(year,11,1));
 return ee.FeatureCollection(c.toList(c.size()).map(function(im){return imageRows(im,year,fc);})).flatten();}
print('Expected 21 LGAs',regions.size());
print('Girei 2023 day/night preview',table(2023,regions.filter(ee.Filter.eq('ADM2_NAME','Girei'))));
if(CREATE_EXPORTS)YEARS.forEach(function(year){Export.table.toDrive({collection:table(year,regions),
 description:'Adamawa_MOD11A2_day_night_'+year+'_v3',folder:'Adamawa_Thesis_LST_v3',
 fileNamePrefix:'Adamawa_MOD11A2_day_night_'+year+'_v3',fileFormat:'CSV',
 selectors:['ADM2_NAME','ADM2_NAME_original','ADM2_CODE','crop_year','date','LST_day_C_mean','LST_night_C_mean',
 'LST_day_C_count','LST_night_C_count','day_qc_zero_restored_count','night_qc_zero_restored_count',
 'product','scale_m','qa_policy','export_version']});});
