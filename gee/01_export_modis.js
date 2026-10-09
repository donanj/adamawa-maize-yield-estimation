// Paste into the GEE JavaScript Code Editor. Run boundary check first.
// Candidate boundary source: GAUL 2015 (deprecated catalog asset).
// Replace with an approved LGA asset if unavailable or names/coverage are unsuitable.
var BOUNDARY_ASSET = 'FAO/GAUL/2015/level2';
var CREATE_EXPORTS = false; // Set true only after checking 21 distinct Adamawa LGAs.
var years = [2010, 2012, 2015, 2018, 2023];
var lgas = ee.FeatureCollection(BOUNDARY_ASSET)
  .filter(ee.Filter.eq('ADM0_NAME', 'Nigeria'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Adamawa'));
print('Boundary count (expected 21)', lgas.size());
print('LGA names: check against boundary_check.csv', lgas.aggregate_array('ADM2_NAME').sort());
Map.centerObject(lgas, 7);
Map.addLayer(lgas, {}, 'Candidate Adamawa LGA boundaries');
function maskVI(im) {
  var q = im.select('SummaryQA');
  // Strict good-quality pixels. No crop mask: LGA-wide extraction.
  return im.select(['NDVI','EVI']).multiply(0.0001)
    .updateMask(q.eq(0)).copyProperties(im, ['system:time_start']);
}
function maskLST(im) {
  var q = im.select('QC_Day');
  // Mandatory QA good (bits 0-1=0), data quality good (2-3=0),
  // LST error <=1 K (6-7=0). Daytime LST in degrees Celsius.
  var good = q.bitwiseAnd(3).eq(0)
    .and(q.rightShift(2).bitwiseAnd(3).eq(0))
    .and(q.rightShift(6).bitwiseAnd(3).eq(0));
  return im.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    .rename('LST_day_C').updateMask(good)
    .copyProperties(im, ['system:time_start']);
}
function rows(collection, scale, seasonYear, product) {
  var list = collection.toList(collection.size());
  return ee.FeatureCollection(list.map(function(obj) {
    var im = ee.Image(obj);
    var reduced = im.reduceRegions({collection:lgas,
      reducer:ee.Reducer.mean().combine({reducer2:ee.Reducer.count(), sharedInputs:true}),
      scale:scale, crs:im.projection(), tileScale:4});
    return reduced.map(function(f) {
      return ee.Feature(null, f.toDictionary()).set({
        crop_year:seasonYear, date:im.date().format('YYYY-MM-dd'),
        product:product, boundary_source:BOUNDARY_ASSET,
        qa_policy:product === 'MOD13Q1' ? 'SummaryQA_eq_0' : 'QC_Day_bits01_0_bits23_0_bits67_0'
      });
    });
  })).flatten();
}
if (CREATE_EXPORTS) {
  lgas.size().evaluate(function(n) {
    if (n !== 21) { print('STOP: boundary count is not 21. Resolve boundary source.'); return; }
    years.forEach(function(y) {
      var start=ee.Date.fromYMD(y,5,1), end=ee.Date.fromYMD(y,11,1);
      var vi=ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start,end).filterBounds(lgas).map(maskVI);
      var lst=ee.ImageCollection('MODIS/061/MOD11A2').filterDate(start,end).filterBounds(lgas).map(maskLST);
      Export.table.toDrive({collection:rows(vi,250,y,'MOD13Q1'),description:'Adamawa_MOD13Q1_'+y,
        folder:'Adamawa_Thesis',fileFormat:'CSV',selectors:['ADM2_CODE','ADM2_NAME','crop_year','date','NDVI_mean','NDVI_count','EVI_mean','EVI_count','product','boundary_source','qa_policy']});
      Export.table.toDrive({collection:rows(lst,1000,y,'MOD11A2'),description:'Adamawa_MOD11A2_'+y,
        folder:'Adamawa_Thesis',fileFormat:'CSV',selectors:['ADM2_CODE','ADM2_NAME','crop_year','date','LST_day_C_mean','LST_day_C_count','product','boundary_source','qa_policy']});
    });
  });
}
