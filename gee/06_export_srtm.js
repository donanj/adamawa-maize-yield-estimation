// Adamawa: V32 elevation (m), V33 slope (%). Static terrain, 21 LGAs.
var CREATE_EXPORTS = false;
var SOURCE = 'USGS/SRTMGL1_003';
var dem = ee.Image(SOURCE).select('elevation');
// Calculate terrain BEFORE clipping so neighbour pixels outside LGAs remain available.
var slopeDegrees = ee.Terrain.slope(dem);
// Convert EACH pixel, then average; tan(mean angle) is not mean percent slope.
var slopePercent = slopeDegrees.multiply(Math.PI / 180).tan().multiply(100);
var lgas = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME','Nigeria'))
  .filter(ee.Filter.eq('ADM1_NAME','Adamawa'))
  .map(function(f) {
    var n = ee.String(f.get('ADM2_NAME'));
    var corrected = ee.String(ee.Algorithms.If(n.equals('Girie'),'Girei',
      ee.Algorithms.If(n.equals('Teungo'),'Toungo',n)));
    return f.set({lga_name:corrected,lga_original:n,gaul_code:f.get('ADM2_CODE')});
  });
var area = ee.Image.pixelArea();
function summarize(image, id, property, unit, regions) {
  var validArea = area.updateMask(image.mask()).rename('valid_area_m2');
  var sums = image.multiply(area).rename('value_area_sum')
    .addBands(validArea).addBands(area.rename('grid_area_m2'));
  return sums.reduceRegions({collection:regions,reducer:ee.Reducer.sum(),
    crs:dem.projection(),scale:dem.projection().nominalScale(),tileScale:4,
    maxPixelsPerRegion:1e9}).map(function(f) {
      var valid = ee.Number(ee.Algorithms.If(f.get('valid_area_m2'),f.get('valid_area_m2'),0));
      var full = ee.Number(f.get('grid_area_m2'));
      return ee.Feature(null,{
        lga_name:f.get('lga_name'),lga_original:f.get('lga_original'),gaul_code:f.get('gaul_code'),
        feature_id:id,property:property,unit:unit,
        mean_value:ee.Algorithms.If(valid.gt(0),ee.Number(f.get('value_area_sum')).divide(valid),null),
        valid_area_m2:valid,grid_area_m2:full,
        valid_area_fraction:ee.Algorithms.If(full.gt(0),valid.divide(full),null),
        source_asset:SOURCE,source_band:'elevation',source_version:'003',
        source_acquisition:'2000-02-11/2000-02-22',extraction_scale_m:dem.projection().nominalScale(),
        aggregation:'pixel_area_weighted_mean',boundary_asset:'FAO/GAUL/2015/level2',
        derivation:id === 'V33' ? '100*tan(ee.Terrain.slope(elevation)*pi/180),before_aggregation' : 'native_elevation_m'
      });
    });
}
print('LGA count: expected 21',lgas.size());
print('Corrected names',lgas.aggregate_array('lga_name').sort());
var previewRegions = lgas.filter(ee.Filter.eq('lga_name','Girei'));
print('Girei preview: elevation and percent slope',
  summarize(dem,'V32','elevation','m',previewRegions)
  .merge(summarize(slopePercent,'V33','slope','percent',previewRegions)));
Map.centerObject(lgas,8);
Map.addLayer(lgas,{},'Adamawa boundaries');
if (CREATE_EXPORTS) {
  var table = summarize(dem,'V32','elevation','m',lgas)
    .merge(summarize(slopePercent,'V33','slope','percent',lgas));
  Export.table.toDrive({collection:table,description:'Adamawa_SRTM_terrain_v1',
    folder:'Adamawa_Thesis_SRTM',fileNamePrefix:'Adamawa_SRTM_terrain_v1',fileFormat:'CSV',
    selectors:['lga_name','lga_original','gaul_code','feature_id','property','unit','mean_value',
      'valid_area_m2','grid_area_m2','valid_area_fraction','source_asset','source_band',
      'source_version','source_acquisition','extraction_scale_m','aggregation','boundary_asset','derivation']});
}
