// Run preview first, then enable the 11 supplementary export tasks.
var CREATE_EXPORTS = false;
var YEARS = [2010,2012,2015,2018,2023];
var FOLDER = 'Adamawa_Thesis_FeatureSupplements';
var lgas = ee.FeatureCollection('FAO/GAUL/2015/level2')
 .filter(ee.Filter.eq('ADM0_NAME','Nigeria')).filter(ee.Filter.eq('ADM1_NAME','Adamawa'))
 .map(function(f) {
  var n=ee.String(f.get('ADM2_NAME'));
  var name=ee.String(ee.Algorithms.If(n.equals('Girie'),'Girei',ee.Algorithms.If(n.equals('Teungo'),'Toungo',n)));
  return f.set({ADM2_NAME_original:n,ADM2_NAME:name,
    latitude_deg:f.geometry().centroid(100).coordinates().get(1)});
 });
function series(collection, year, kind, regions) {
 var list=collection.toList(collection.size());
 return ee.FeatureCollection(list.map(function(obj) {
  var im=ee.Image(obj);var date=ee.Date(im.get('system:time_start'));
  var qa,values;
  if (kind==='vegetation') {
   qa=im.select('SummaryQA').eq(0);
   values=im.select(['NDVI','EVI'],['NDVI_mean','EVI_mean']).multiply(.0001).updateMask(qa);
  } else {
   var q=im.select('QC_Night');
   qa=q.bitwiseAnd(3).eq(0).and(q.rightShift(2).bitwiseAnd(3).eq(0))
     .and(q.rightShift(6).bitwiseAnd(3).eq(0));
   values=im.select('LST_Night_1km').multiply(.02).subtract(273.15)
     .updateMask(qa).rename('LST_night_C_mean');
  }
  return values.reduceRegions({collection:regions,reducer:ee.Reducer.mean(),
   crs:values.projection(),scale:values.projection().nominalScale(),tileScale:4,maxPixelsPerRegion:1e9})
   .map(function(f) {return ee.Feature(null,f.toDictionary(kind==='vegetation' ? ['ADM2_NAME','ADM2_NAME_original','ADM2_CODE','NDVI_mean','EVI_mean'] : ['ADM2_NAME','ADM2_NAME_original','ADM2_CODE','LST_night_C_mean']))
     .set({crop_year:year,date:date.format('YYYY-MM-dd'),
      qa_policy:kind==='vegetation'?'SummaryQA_eq_0':'QC_Night_bits01_0_bits23_0_bits67_0',
      export_version:'feature_supplement_v1'});});
 })).flatten();
}
function aux(year,month,regions) {
 var start=ee.Date.fromYMD(year,month,1);var end=start.advance(1,'month');
 var c=ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start,end);
 var expected=end.difference(start,'day');
 var rain=c.sum().updateMask(c.count().eq(expected)).rename('rainfall_mm');
 var area=ee.Image.pixelArea();var sums=rain.multiply(area).rename('weighted_sum')
   .addBands(area.updateMask(rain.mask()).rename('valid_area_m2')).addBands(area.rename('grid_area_m2'));
 return sums.reduceRegions({collection:regions,reducer:ee.Reducer.sum(),crs:'EPSG:4326',
  crsTransform:[.05,0,-180,0,-.05,50],tileScale:4,maxPixelsPerRegion:1e9})
  .map(function(f) {
   var valid=ee.Number(ee.Algorithms.If(f.get('valid_area_m2'),f.get('valid_area_m2'),0));
   return ee.Feature(null,{ADM2_NAME:f.get('ADM2_NAME'),ADM2_NAME_original:f.get('ADM2_NAME_original'),
    ADM2_CODE:f.get('ADM2_CODE'),crop_year:year,month:month,latitude_deg:f.get('latitude_deg'),
    rainfall_mm:ee.Algorithms.If(valid.gt(0),ee.Number(f.get('weighted_sum')).divide(valid),null),
    valid_area_fraction:valid.divide(ee.Number(f.get('grid_area_m2'))),source_images:c.size(),expected_days:expected,
    latitude_method:'GAUL_polygon_centroid_100m_error',export_version:'feature_supplement_v1'});
  });
}
var first=2010;
var v=ee.ImageCollection('MODIS/061/MOD13Q1').filterDate('2010-04-01','2010-12-01');
var n=ee.ImageCollection('MODIS/061/MOD11A2').filterDate('2010-05-01','2010-11-01');
var sample=lgas.filter(ee.Filter.eq('ADM2_NAME','Girei'));
print('21 LGAs expected',lgas.size());
print('Buffered vegetation first date',series(v.limit(1),first,'vegetation',sample));
print('Nighttime LST first date',series(n.limit(1),first,'night',sample));
print('March rainfall and latitude',aux(first,3,sample));
if (CREATE_EXPORTS) {
 var auxiliaries=[];
 YEARS.forEach(function(year) {
  var start=ee.Date.fromYMD(year,4,1);var end=ee.Date.fromYMD(year,12,1);
  var vc=ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start,end);
  var nc=ee.ImageCollection('MODIS/061/MOD11A2').filterDate(ee.Date.fromYMD(year,5,1),ee.Date.fromYMD(year,11,1));
  Export.table.toDrive({collection:series(vc,year,'vegetation',lgas),description:'Adamawa_MOD13Q1_buffer_'+year,
   folder:FOLDER,fileNamePrefix:'Adamawa_MOD13Q1_buffer_'+year,fileFormat:'CSV',
   selectors:['ADM2_NAME','ADM2_NAME_original','ADM2_CODE','crop_year','date','NDVI_mean','EVI_mean','qa_policy','export_version']});
  Export.table.toDrive({collection:series(nc,year,'night',lgas),description:'Adamawa_MOD11A2_night_'+year,
   folder:FOLDER,fileNamePrefix:'Adamawa_MOD11A2_night_'+year,fileFormat:'CSV',
   selectors:['ADM2_NAME','ADM2_NAME_original','ADM2_CODE','crop_year','date','LST_night_C_mean','qa_policy','export_version']});
  auxiliaries.push(aux(year,3,lgas));auxiliaries.push(aux(year,4,lgas));
 });
 Export.table.toDrive({collection:ee.FeatureCollection(auxiliaries).flatten(),description:'Adamawa_feature_auxiliary_v1',
  folder:FOLDER,fileNamePrefix:'Adamawa_feature_auxiliary_v1',fileFormat:'CSV',
  selectors:['ADM2_NAME','ADM2_NAME_original','ADM2_CODE','crop_year','month','latitude_deg','rainfall_mm',
   'valid_area_fraction','source_images','expected_days','latitude_method','export_version']});
}
