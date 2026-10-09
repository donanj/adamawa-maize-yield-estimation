// CHIRPS v2 final: monthly rainfall totals averaged over each LGA.
// First run preview; set CREATE_EXPORTS=true only when numeric values appear.
var CREATE_EXPORTS=false;
var years=[2010,2012,2015,2018,2023];
var SOURCE='UCSB-CHG/CHIRPS/DAILY';
var BOUNDARY='FAO/GAUL/2015/level2';
var lgas=ee.FeatureCollection(BOUNDARY).filter(ee.Filter.eq('ADM0_NAME','Nigeria'))
 .filter(ee.Filter.eq('ADM1_NAME','Adamawa'));
var daily=ee.ImageCollection(SOURCE).select('precipitation');
var projection=ee.Image(daily.first()).projection();
var nameMap=ee.Dictionary({'Girie':'Girei','Teungo':'Toungo'});
function monthRows(y,m){
 var start=ee.Date.fromYMD(y,m,1),end=start.advance(1,'month');
 var expected=end.difference(start,'day');
 var subset=daily.filterDate(start,end).filterBounds(lgas);
 var n=subset.size();
 // Require all calendar days at each pixel; missing days are not zeros.
 var complete=subset.count().eq(expected);
 var total=subset.sum().rename('rain_mm').updateMask(complete)
  .updateMask(ee.Image.constant(n.eq(expected)));
 var area=ee.Image.pixelArea();
 var statsImage=total.multiply(area).rename('rain_x_area')
  .addBands(area.updateMask(total.mask()).rename('valid_area_m2'))
  .addBands(area.rename('lga_grid_area_m2'));
 return lgas.map(function(f){
  // Multiple bands + one statistic: known band keys, no assumed suffixes.
  var stats=statsImage.reduceRegion({reducer:ee.Reducer.sum(),geometry:f.geometry(),
   crs:'EPSG:4326',crsTransform:[0.05,0,-180,0,-0.05,50],maxPixels:1e8,tileScale:4});
  var valid=ee.Number(stats.get('valid_area_m2',0));
  var full=ee.Number(stats.get('lga_grid_area_m2',0));
  var original=ee.String(f.get('ADM2_NAME'));
  return ee.Feature(null,{
   ADM2_CODE:f.get('ADM2_CODE'),ADM2_NAME:nameMap.get(original,original),ADM2_NAME_original:original,
   crop_year:y,month:m,month_start:start.format('YYYY-MM-dd'),month_end_exclusive:end.format('YYYY-MM-dd'),
   rainfall_mm:ee.Algorithms.If(valid.gt(0),ee.Number(stats.get('rain_x_area',0)).divide(valid),null),
   expected_days:expected,source_images:n,calendar_complete:n.eq(expected),
   valid_area_m2:valid,lga_grid_area_m2:full,
   valid_area_fraction:ee.Algorithms.If(full.gt(0),valid.divide(full),null),
   product:'CHIRPS_v2_final_daily',boundary_source:BOUNDARY,
   aggregation:'daily_sum_then_area_weighted_LGA_mean',units:'mm',export_version:'CHIRPS_monthly_v1'
  });
 });
}
function yearRows(y){return ee.FeatureCollection(ee.List.sequence(5,10).map(function(m){return monthRows(y,ee.Number(m));})).flatten();}
print('Boundary count (expect 21)',lgas.size());
print('Native CHIRPS projection',projection);
var preview=monthRows(2023,8);
print('August 2023 preview: expand properties',preview.limit(5));
print('LGAs with numeric August rainfall (expect 21)',preview.filter(ee.Filter.notNull(['rainfall_mm'])).size());
if(CREATE_EXPORTS){
 lgas.size().evaluate(function(n){
  if(n!==21){print('STOP: expected 21 boundaries.');return;}
  years.forEach(function(y){
   Export.table.toDrive({collection:yearRows(y),description:'Adamawa_CHIRPS_monthly_'+y,
    folder:'Adamawa_Thesis_CHIRPS',fileFormat:'CSV',selectors:[
     'ADM2_CODE','ADM2_NAME','ADM2_NAME_original','crop_year','month','month_start','month_end_exclusive',
     'rainfall_mm','expected_days','source_images','calendar_complete','valid_area_m2','lga_grid_area_m2',
     'valid_area_fraction','product','boundary_source','aggregation','units','export_version']});
  });
 });
}
