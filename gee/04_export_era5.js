// ERA5-Land climate inputs. Preview first; ten tasks when enabled.
var CREATE_EXPORTS=false;
var years=[2010,2012,2015,2018,2023];
var SOURCE='ECMWF/ERA5_LAND/DAILY_AGGR';
var daily=ee.ImageCollection(SOURCE);
var hourly=ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY');
var lgas=ee.FeatureCollection('FAO/GAUL/2015/level2')
 .filter(ee.Filter.eq('ADM0_NAME','Nigeria')).filter(ee.Filter.eq('ADM1_NAME','Adamawa'));
var names=ee.Dictionary({'Girie':'Girei','Teungo':'Toungo'});
var proj=ee.Image(daily.first()).select('temperature_2m').projection();
var scale=proj.nominalScale(); // Numeric scale; no projection.transform() string.
var base=['temperature_2m','temperature_2m_max','temperature_2m_min','dewpoint_temperature_2m',
 'surface_pressure','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3'];
var outNames=['Tmean_C','Tmax_daily_mean_C','Tmin_daily_mean_C','dewpoint_C',
 'pressure_kPa','soil_water_0_7','soil_water_7_28','soil_water_28_100'];
function reduceRows(im,metadata){
 return lgas.map(function(f){
  var stats=im.reduceRegion({reducer:ee.Reducer.mean(),geometry:f.geometry(),crs:proj,
   scale:scale,maxPixels:1e8,tileScale:4});
  var original=ee.String(f.get('ADM2_NAME'));
  return ee.Feature(null,stats).set(metadata).set({ADM2_CODE:f.get('ADM2_CODE'),
   ADM2_NAME:names.get(original,original),ADM2_NAME_original:original,
   product:SOURCE,boundary_source:'FAO/GAUL/2015/level2',export_version:'ERA5_inputs_v1'});
 });
}
function monthlyRows(y,m){
 var start=ee.Date.fromYMD(y,m,1),end=start.advance(1,'month');
 var days=end.difference(start,'day'),sub=daily.filterDate(start,end).filterBounds(lgas);
 var means=sub.select(base).mean();
 var temps=means.select(base.slice(0,4)).subtract(273.15).rename(outNames.slice(0,4));
 var pressure=means.select('surface_pressure').divide(1000).rename('pressure_kPa');
 var soil=means.select(base.slice(5)).rename(outNames.slice(5));
 var rs=sub.select('surface_solar_radiation_downwards_sum').sum().divide(1e6).rename('solar_MJ_m2_month');
 // Average scalar hourly wind speeds, not magnitude of mean wind components.
 var hs=hourly.filterDate(start,end).filterBounds(lgas);
 var wind=hs.map(function(im){return im.select('u_component_of_wind_10m').pow(2)
  .add(im.select('v_component_of_wind_10m').pow(2)).sqrt().rename('wind10_ms');}).mean();
 var selected=base.concat(['surface_solar_radiation_downwards_sum']);
 var complete=sub.select(selected).count().reduce(ee.Reducer.min()).eq(days)
  .and(hs.select(['u_component_of_wind_10m','v_component_of_wind_10m']).count()
   .reduce(ee.Reducer.min()).eq(days.multiply(24)));
 var im=temps.addBands(pressure).addBands(soil).addBands(rs)
  .addBands(rs.divide(days).rename('solar_MJ_m2_day')).addBands(wind)
  .updateMask(complete).updateMask(ee.Image.constant(sub.size().eq(days).and(hs.size().eq(days.multiply(24)))));
 var coverage=ee.Image.pixelArea().updateMask(im.select('Tmean_C').mask()).rename('valid_area');
 var fullArea=ee.Image.pixelArea().rename('full_area');
 return reduceRows(im,{crop_year:y,month:m,month_start:start.format('YYYY-MM-dd'),
  expected_days:days,source_days:sub.size(),expected_hours:days.multiply(24),source_hours:hs.size(),
  season_role:ee.Algorithms.If(ee.Number(m).eq(4),'April_auxiliary','MayOct_predictor')}).map(function(f){
   var geom=ee.Feature(lgas.filter(ee.Filter.eq('ADM2_CODE',f.get('ADM2_CODE'))).first()).geometry();
   var areas=coverage.addBands(fullArea).reduceRegion({reducer:ee.Reducer.sum(),geometry:geom,
    crs:proj,scale:scale,maxPixels:1e8,tileScale:4});
   var valid=ee.Number(areas.get('valid_area',0)),full=ee.Number(areas.get('full_area',0));
   return f.set({valid_area_fraction:ee.Algorithms.If(full.gt(0),valid.divide(full),null)});
  });
}
function dailyTemperatureRows(y){
 var sub=daily.filterDate(ee.Date.fromYMD(y,5,1),ee.Date.fromYMD(y,11,1)).filterBounds(lgas);
 return ee.FeatureCollection(sub.toList(sub.size()).map(function(obj){
  var im=ee.Image(obj);return reduceRows(im.select(['temperature_2m','temperature_2m_max','temperature_2m_min'])
   .subtract(273.15).rename(['Tmean_C','Tmax_C','Tmin_C']),
   {crop_year:y,date:im.date().format('YYYY-MM-dd'),units:'degC'});
 })).flatten();
}
var preview=monthlyRows(2023,8);
print('Boundary count (expect 21)',lgas.size());print('ERA5 native projection and nominal metres',proj,scale);
print('August 2023 monthly inputs: expand properties',preview.limit(5));
print('Numeric Tmean LGAs (expect 21)',preview.filter(ee.Filter.notNull(['Tmean_C'])).size());
if(CREATE_EXPORTS){years.forEach(function(y){
 var monthly=ee.FeatureCollection(ee.List.sequence(4,10).map(function(m){return monthlyRows(y,ee.Number(m));})).flatten();
 Export.table.toDrive({collection:monthly,description:'Adamawa_ERA5_monthly_'+y,folder:'Adamawa_Thesis_ERA5',
  fileFormat:'CSV',selectors:['ADM2_CODE','ADM2_NAME','ADM2_NAME_original','crop_year','month','month_start',
   'Tmean_C','Tmax_daily_mean_C','Tmin_daily_mean_C','dewpoint_C','pressure_kPa',
   'soil_water_0_7','soil_water_7_28','soil_water_28_100','solar_MJ_m2_month','solar_MJ_m2_day','wind10_ms',
   'expected_days','source_days','expected_hours','source_hours','valid_area_fraction','season_role',
   'product','boundary_source','export_version']});
 Export.table.toDrive({collection:dailyTemperatureRows(y),description:'Adamawa_ERA5_daily_temperature_'+y,
  folder:'Adamawa_Thesis_ERA5',fileFormat:'CSV',selectors:['ADM2_CODE','ADM2_NAME','ADM2_NAME_original',
   'crop_year','date','Tmean_C','Tmax_C','Tmin_C','units','product','boundary_source','export_version']});
});}
