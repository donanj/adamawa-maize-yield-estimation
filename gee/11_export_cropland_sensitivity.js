// Cropland sensitivity: matched controls, annual masks and coverage audits.
// Preview first. Set CREATE_EXPORTS=true after preview checks; 10 satellite export tasks; retain the original area-fractions CSV.
var CREATE_EXPORTS=false;
var YEARS=[2010,2012,2015,2018,2023];
var FOLDER='Adamawa_Cropland_Sensitivity';
var BOUNDARY='FAO/GAUL/2015/level2';
var LC_SOURCE='MODIS/061/MCD12Q1';
var MODES=['unmasked','cropland12','cropland12_14'];
var regions=ee.FeatureCollection(BOUNDARY)
 .filter(ee.Filter.eq('ADM0_NAME','Nigeria')).filter(ee.Filter.eq('ADM1_NAME','Adamawa'))
 .map(function(f){var n=ee.String(f.get('ADM2_NAME'));return f.set({ADM2_NAME_original:n,
 ADM2_NAME:ee.Algorithms.If(n.equals('Girie'),'Girei',ee.Algorithms.If(n.equals('Teungo'),'Toungo',n))});});
function annual(year){return ee.ImageCollection(LC_SOURCE).filterDate(ee.Date.fromYMD(year,1,1),ee.Date.fromYMD(year+1,1,1));}
function classification(year){var im=ee.Image(annual(year).first());var q=im.select('QC');return im.select('LC_Type1').updateMask(q.eq(0).or(q.eq(2)));}
function support(lc,mode){
 // Default categorical nearest-neighbour sampling on each satellite grid.
 // No bilinear resampling and no claimed subpixel crop purity.
 return mode==='unmasked'?ee.Image.constant(1):mode==='cropland12'?lc.eq(12).unmask(0):lc.eq(12).or(lc.eq(14)).unmask(0);
}
function safeNumber(dict,key){return ee.Number(ee.Algorithms.If(dict.get(key),dict.get(key),0));}
function fraction(n,d){return ee.Algorithms.If(d.gt(0),n.divide(d),null);}
function metadata(feature,year,lc,mode){return feature.set({crop_year:year,mask_mode:mode,
 landcover_source:LC_SOURCE,landcover_band:'LC_Type1',landcover_year:year,landcover_qc_policy:'QC_0_classified_land_or_2_classified_water',
 landcover_image:ee.Image(annual(year).first()).get('system:index'),
 classification_date:ee.Image(annual(year).first()).date().format('YYYY-MM-dd'),
 landcover_scale_m:lc.projection().nominalScale(),boundary_source:BOUNDARY,
 mask_sampling:'nearest_neighbour_on_product_reduction_grid',count_policy:'separate_centroid_counts_on_product_grid',export_version:'cropland_sensitivity_v2'});}
function fractions(year){
 var lc=classification(year);var area=ee.Image.pixelArea();var known=lc.gte(1).and(lc.lte(17)).unmask(0);
 var bands=area.rename('grid_area_m2')
 .addBands(area.updateMask(known).rename('classified_area_m2'))
 .addBands(area.updateMask(lc.eq(12)).rename('cropland12_area_m2'))
 .addBands(area.updateMask(lc.eq(14)).rename('mosaic14_area_m2'))
 .addBands(area.updateMask(known.and(lc.neq(12)).and(lc.neq(14))).rename('other_classified_area_m2'));
 return bands.reduceRegions({collection:regions,reducer:ee.Reducer.sum(),crs:lc.projection(),scale:lc.projection().nominalScale(),tileScale:4,maxPixelsPerRegion:1e9})
 .map(function(f){var dict=f.toDictionary();var total=safeNumber(dict,'grid_area_m2');var classified=safeNumber(dict,'classified_area_m2');var crop=safeNumber(dict,'cropland12_area_m2');var mosaic=safeNumber(dict,'mosaic14_area_m2');var other=safeNumber(dict,'other_classified_area_m2');
 var out=ee.Feature(null,{ADM2_NAME:f.get('ADM2_NAME'),ADM2_NAME_original:f.get('ADM2_NAME_original'),ADM2_CODE:f.get('ADM2_CODE'),
 grid_area_m2:total,classified_area_m2:classified,cropland12_area_m2:crop,mosaic14_area_m2:mosaic,other_classified_area_m2:other,
 cropland12_fraction: fraction(crop,total),mosaic14_fraction:fraction(mosaic,total),cropland12_14_fraction:fraction(crop.add(mosaic),total),
 noncropland12_fraction:fraction(other.add(mosaic),total),other_classified_fraction:fraction(other,total),unknown_fraction:fraction(total.subtract(classified),total)});
 return metadata(out,year,lc,'area_fractions');});
}
function prepareLST(im,period){
 var raw=im.select(period==='day'?'LST_Day_1km':'LST_Night_1km');var original=im.select(period==='day'?'QC_Day':'QC_Night');
 var valid=raw.mask().and(raw.gt(0));var qc=original.unmask(0).updateMask(valid);
 var good=qc.bitwiseAnd(3).eq(0).and(qc.rightShift(2).bitwiseAnd(3).eq(0)).and(qc.rightShift(6).bitwiseAnd(3).eq(0));
 return {value:raw.multiply(.02).subtract(273.15).updateMask(valid).updateMask(good).rename('LST_'+period+'_C'),
 restored:original.mask().unmask(0).eq(0).and(valid).and(good).selfMask().rename(period+'_qc_zero_restored')};
}
function imageRows(obj,year,kind,fc){
 var im=ee.Image(obj),lc=classification(year),values,recovered,proj,scale,names;
 if(kind==='vegetation'){
  values=im.select(['NDVI','EVI']).multiply(.0001).updateMask(im.select('SummaryQA').eq(0));
  recovered=null;proj=im.select('NDVI').projection();scale=proj.nominalScale();names=['NDVI','EVI'];
 }else{
  var day=prepareLST(im,'day'),night=prepareLST(im,'night');values=day.value.addBands(night.value);recovered=day.restored.addBands(night.restored);
  proj=im.select('LST_Day_1km').projection();scale=1000;names=['LST_day_C','LST_night_C'];
 }
 return ee.FeatureCollection(MODES.map(function(mode){var crop=support(lc,mode);
  var reducer=ee.Reducer.mean();
  var measured=values.updateMask(crop);
  return fc.map(function(f){
   var opts={geometry:f.geometry(),crs:proj,scale:scale,tileScale:4,maxPixels:1e9};
   var means=measured.reduceRegion({reducer:reducer,geometry:opts.geometry,crs:proj,scale:scale,tileScale:4,maxPixels:1e9});
   var counts=measured.reduceRegion({reducer:ee.Reducer.count(),geometry:opts.geometry,crs:proj,scale:scale,tileScale:4,maxPixels:1e9});
   // Derive full support from a source band, retaining its product grid.
   var grid=im.select(kind==='vegetation'?'NDVI':'LST_Day_1km').multiply(0).add(1).unmask(1).rename('eligible');
   var stats=ee.Dictionary({});
   // Dictionary.map omits null-valued entries; never call Dictionary.set with a null mean.
   var nonNullMeans=means.map(function(key,value){return value;});
   var meanKeys=nonNullMeans.keys();
   var renamedKeys=meanKeys.map(function(key){return ee.String(key).cat('_mean');});
   stats=ee.Dictionary.fromLists(renamedKeys,nonNullMeans.values(meanKeys));
   names.forEach(function(name){stats=stats.set(name+'_count',safeNumber(counts,name));});
   var eligible=grid.updateMask(crop).reduceRegion({reducer:ee.Reducer.count(),geometry:opts.geometry,crs:proj,scale:scale,tileScale:4,maxPixels:1e9});
   var denominator=safeNumber(eligible,'eligible');
   var out=ee.Feature(null,stats).set({ADM2_NAME:f.get('ADM2_NAME'),ADM2_NAME_original:f.get('ADM2_NAME_original'),ADM2_CODE:f.get('ADM2_CODE'),
    date:im.date().format('YYYY-MM-dd'),product:kind==='vegetation'?'MODIS/061/MOD13Q1':'MODIS/061/MOD11A2',
    reduction_scale_m:scale,eligible_grid_count:denominator,
    qa_policy:kind==='vegetation'?'SummaryQA_eq_0':'mandatory_0_data_0_error_le1K_QC_zero_restored_only_with_valid_LST'});
   names.forEach(function(name){out=out.set(name+'_valid_fraction',fraction(safeNumber(stats,name+'_count'),denominator));});
   if(recovered!==null){var rec=recovered.updateMask(crop).reduceRegion({reducer:ee.Reducer.count(),geometry:opts.geometry,crs:proj,scale:scale,tileScale:4,maxPixels:1e9});out=out.set({day_qc_zero_restored_count:safeNumber(rec,'day_qc_zero_restored'),night_qc_zero_restored_count:safeNumber(rec,'night_qc_zero_restored')});}
   return metadata(out,year,lc,mode);
  });
 })).flatten();
}
function series(year,kind,fc){var start=ee.Date.fromYMD(year,kind==='vegetation'?4:5,1);var end=ee.Date.fromYMD(year,kind==='vegetation'?12:11,1);
 var coll=ee.ImageCollection(kind==='vegetation'?'MODIS/061/MOD13Q1':'MODIS/061/MOD11A2').filterDate(start,end);
 return ee.FeatureCollection(coll.toList(coll.size()).map(function(im){return imageRows(im,year,kind,fc);})).flatten();}
var inventory=ee.FeatureCollection(YEARS.map(function(y){return ee.Feature(null,{crop_year:y,landcover_images:annual(y).size()});}));
print('Boundary count expected 21',regions.size());print('Corrected LGA names',regions.aggregate_array('ADM2_NAME').sort());print('Annual mask inventory expected one per year',inventory);
Map.centerObject(regions,7);Map.addLayer(classification(2023).eq(12).selfMask(),{palette:['yellow']},'2023 class 12 cropland');Map.addLayer(classification(2023).eq(14).selfMask(),{palette:['orange']},'2023 class 14 mosaic');
print('2023 LGA area fractions',fractions(2023));
var sample=regions.filter(ee.Filter.eq('ADM2_NAME','Girei'));
print('Girei vegetation first composite: three extraction modes',imageRows(ee.ImageCollection('MODIS/061/MOD13Q1').filterDate('2023-04-01','2023-12-01').first(),2023,'vegetation',sample));
print('Girei day/night first composite: three extraction modes',imageRows(ee.ImageCollection('MODIS/061/MOD11A2').filterDate('2023-05-01','2023-11-01').first(),2023,'thermal',sample));
if(CREATE_EXPORTS){
 ee.Dictionary({boundary_count:regions.size(),distinct_names:regions.aggregate_count_distinct('ADM2_NAME'),mask_counts:inventory.aggregate_array('landcover_images')}).evaluate(function(check,error){
  if(error||check.boundary_count!==21||check.distinct_names!==21||check.mask_counts.some(function(n){return n!==1;})){print('STOP resolve boundary or same-year mask availability',check,error);return;}
  YEARS.forEach(function(y){['vegetation','thermal'].forEach(function(kind){var name='Adamawa_cropland_'+kind+'_'+y+'_v2';Export.table.toDrive({collection:series(y,kind,regions),description:name,fileNamePrefix:name,folder:FOLDER,fileFormat:'CSV'});});});
  // Area fractions passed review; retain Adamawa_cropland_area_fractions_v1.csv.
 });
}
