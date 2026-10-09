// Adamawa iSDAsoil extraction. Paste into Google Earth Engine Code Editor.
// Static v1 soil maps; decoded pixel values are averaged BEFORE any panel join.
var CREATE_EXPORTS = false; // Change to true after checking the preview.
var FOLDER = 'Adamawa_Thesis_iSDAsoil';
var boundaries = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME', 'Nigeria'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Adamawa'));
var lgas = boundaries.map(function(f) {
  var original = ee.String(f.get('ADM2_NAME'));
  var corrected = ee.String(ee.Algorithms.If(original.equals('Girie'), 'Girei',
    ee.Algorithms.If(original.equals('Teungo'), 'Toungo', original)));
  return f.set({lga_name: corrected, lga_original: original, gaul_code: f.get('ADM2_CODE')});
});
// Texture values are percent as documented in the band tables; no exponential transform.
var properties = [
  {id:'V23', asset:'clay_content', unit:'percent', decode:'identity'},
  {id:'V24', asset:'sand_content', unit:'percent', decode:'identity'},
  {id:'V25', asset:'carbon_organic', unit:'g_per_kg', decode:'exp10'},
  {id:'V26', asset:'ph', unit:'pH', decode:'divide10'},
  {id:'V28', asset:'bulk_density', unit:'g_per_cm3', decode:'divide100'},
  {id:'V29', asset:'cation_exchange_capacity', unit:'cmol_charge_per_kg', decode:'exp10'},
  {id:'V30', asset:'nitrogen_total', unit:'g_per_kg', decode:'exp100'},
  {id:'V31', asset:'phosphorus_extractable', unit:'mg_per_kg', decode:'exp10'}
];
function decoded(raw, how) {
  if (how === 'exp10') return raw.divide(10).exp().subtract(1);
  if (how === 'exp100') return raw.divide(100).exp().subtract(1);
  if (how === 'divide10') return raw.divide(10);
  if (how === 'divide100') return raw.divide(100);
  return raw;
}
function extract(p, depth, regions) {
  var asset = 'ISDASOIL/Africa/v1/' + p.asset;
  var raw = ee.Image(asset).select('mean_' + depth);
  var value = decoded(raw, p.decode).rename('value');
  var area = ee.Image.pixelArea();
  var validArea = area.updateMask(value.mask()).rename('valid_area_m2');
  var sums = value.multiply(area).rename('value_area_sum')
    .addBands(validArea).addBands(area.rename('grid_area_m2'));
  var reduced = sums.reduceRegions({collection:regions, reducer:ee.Reducer.sum(),
    crs:raw.projection(), scale:raw.projection().nominalScale(), tileScale:4,
    maxPixelsPerRegion:1e9});
  return reduced.map(function(f) {
    var valid = ee.Number(ee.Algorithms.If(f.get('valid_area_m2'), f.get('valid_area_m2'), 0));
    var full = ee.Number(f.get('grid_area_m2'));
    return ee.Feature(null, {
      lga_name:f.get('lga_name'), lga_original:f.get('lga_original'), gaul_code:f.get('gaul_code'),
      feature_id:p.id, property:p.asset, depth_cm:depth.replace('_','-'), unit:p.unit,
      mean_value:ee.Algorithms.If(valid.gt(0), ee.Number(f.get('value_area_sum')).divide(valid), null),
      valid_area_m2:valid, grid_area_m2:full,
      valid_area_fraction:ee.Algorithms.If(full.gt(0),valid.divide(full),null),
      source_asset:asset, source_band:'mean_' + depth, decode:p.decode,
      map_version:'v1', extraction_scale_m:raw.projection().nominalScale(),
      aggregation:'pixel_area_weighted_mean_of_decoded_values', boundary_asset:'FAO/GAUL/2015/level2'
    });
  });
}
print('LGA count: expected 21', lgas.size());
print('Corrected LGA names', lgas.aggregate_array('lga_name').sort());
print('Preview: Girei topsoil organic carbon (one row)',
  extract(properties[2], '0_20', lgas.filter(ee.Filter.eq('lga_name','Girei'))));
Map.centerObject(lgas, 8);
Map.addLayer(lgas, {}, 'Adamawa boundaries');
var selectors = ['lga_name','lga_original','gaul_code','feature_id','property','depth_cm','unit',
  'mean_value','valid_area_m2','grid_area_m2','valid_area_fraction','source_asset','source_band',
  'decode','map_version','extraction_scale_m','aggregation','boundary_asset'];
if (CREATE_EXPORTS) {
  ['0_20', '20_50'].forEach(function(depth) {
    var collections = properties.map(function(p) { return extract(p, depth, lgas); });
    var table = ee.FeatureCollection(collections).flatten();
    Export.table.toDrive({collection:table, description:'Adamawa_iSDAsoil_' + depth + '_v1',
      folder:FOLDER, fileNamePrefix:'Adamawa_iSDAsoil_' + depth + '_v1',
      fileFormat:'CSV', selectors:selectors});
  });
}
// No V27: available water capacity requires a separately justified source/derivation.
// No standard-deviation bands: these are not LGA standard errors or yield uncertainty.
