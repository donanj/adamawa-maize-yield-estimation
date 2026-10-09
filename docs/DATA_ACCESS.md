# Data inventory and acquisition


| Data | Official access/catalogue | Local role |
|---|---|---|
| Nigeria LSMS-ISA/GHS Panel | https://microdata.worldbank.org/ | Aggregated maize-yield reference panels; original five wave CSV archives remain local |
| MOD13Q1 v6.1 | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1 | April–November buffered NDVI/EVI native time series |
| MOD11A2 v6.1 | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2 | May–October repaired-QC day/night LST v3 |
| CHIRPS daily | https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY | Monthly rainfall, plus March/April auxiliary rainfall |
| ERA5-Land daily aggregates | https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR | Monthly climate inputs built from daily data |
| ERA5-Land hourly | https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY | Hourly scalar wind-speed aggregation |
| iSDAsoil Africa v1 | https://developers.google.com/earth-engine/datasets/catalog/ISDASOIL_Africa_v1_nitrogen_total | Eight documented properties at two depths; nitrogen page is one example asset, not the whole soil inventory |
| SRTMGL1 v3 | https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003 | Elevation and pixelwise percent slope |
| FAO GAUL 2015 level 2 | https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level2 | 21 Adamawa extraction boundaries |

These catalogue URLs identify products, not a preselected Adamawa location. The GEE scripts filter the boundaries and perform Adamawa reductions. Preserve original GAUL names/codes and canonical Girei/Toungo name corrections. Soil assets and encodings are recorded in gee/05_export_isdasoil.js; V27 has no accepted source.

The bundled aggregated yield labels have LGA/year keys and audit counts, not household or plot linkage identifiers. Raw survey files and private plot audits are not bundled in this starter. To rebuild labels from those source files, place the five original NGA_*-W1_*_CSV.zip through W5 archives in data/private/lsms. Read the source audit code and verify harvest units, area measures and weighting before accepting changed labels. Do not upload archives, individual survey identifiers, credentials or private audit tables to GitHub.

Inputs called lst_day.csv and vegetation.csv support comparisons with earlier extracts; current features prefer all five repaired day/night v3 files and all five buffered vegetation files consistently. Old failed nighttime files are omitted. Soil CSVs already contain decoded physical units and area-weighted means; do not decode them again.
