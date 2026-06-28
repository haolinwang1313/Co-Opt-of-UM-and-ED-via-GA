# Data Sources

All geopackages in `data/gpkg/` originate from the processed layers used in the Xinwu (Wuxi) study.

## External downloads (Zenodo)
- `EULUC_China_20.gpkg` (Zenodo DOI: [10.5281/zenodo.15180905](https://zenodo.org/records/15180905)) provided the baseline land-use classification that was clipped/converted into `xinwu_land_use.gpkg` and `xinwu_water_osm.gpkg`.
- `RealEstateData.csv` and `PopulationDensityData.csv` from Zenodo DOI: [10.5281/zenodo.5747686](https://zenodo.org/records/5747686) were used during earlier experiments for socioeconomic calibration but are not redistributed here.

## Derived layers in this repo
- `xinwu_grid_250m.gpkg` / `xinwu_urban_form_250m.gpkg`: grids derived from the Xinwu district boundary (EPSG:32650) for morphology aggregation.
- `xinwu_built_env_250m.gpkg`, `xinwu_transportatio_250m.gpkg`, `xinwu_road_centerlines.gpkg`, `xinwu_road_integration.gpkg`: generated via scripts in `code/urban_form/` using the OSM/Zenodo inputs.
- `xinwu_energy_250m.gpkg`: outputs of `code/energy_sim/simulate_building_energy.py`, aggregating EnergyPlus simulations.
- `xinwu_subway_stations.gpkg`, `xinwu_bus_network.gpkg`: transit datasets cleaned from municipal feeds + OSM.

No raw building footprints are distributed; users should obtain equivalent 3D building datasets (e.g., 3D-GloBFP) for their own study area before running the scripts.
