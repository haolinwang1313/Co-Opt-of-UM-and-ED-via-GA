# 20×3 XGBoost Surrogate Package

This folder packages the XGBoost surrogate trained in `Paper03/Mapping/20x_3y/xgboost/` so it can be reused inside `Paper04` for co-optimization studies. The model maps 20 urban-form/built-environment indicators to three grid-level annual energy end uses that were simulated with EnergyPlus 24.1.

## What is included here?

| Path | Purpose |
| --- | --- |
| `models/xgb_*.json` | Booster files for each target (`cooling`, `heating`, `other_electricity`). |
| `data/dataset_20x3.csv` | Full training table (grid id + 20 inputs + 3 targets) copied from `Paper03`. |
| `data/xgb_metrics.json` | 5-fold cross-validation metrics recorded during training. |
| `data/xgb_feature_importance_*.csv` | Gain-based feature importance exports per target. |
| `data/feature_scalers.json` | 拟合自 `Paper03` 原始指标的 8/9 组归一化参数，供后续自定义候选方案沿用相同尺度。 |
| `predict_energy.py` | Batch inference CLI that loads the JSON boosters and appends predictions to any CSV with the 20 required inputs. |
| `feature_pipeline.py` | `compute_surrogate_features()`：统一计算 20 个输入特征（translator/GA 直接调用）。 |
| `README.md` (this file) | Describes inputs, outputs, model quality, and usage. |

> All files remain untouched copies of the Paper03 artifacts except for the new helper script and documentation.

## Feature reference (20 inputs)

The indicators are computed on a 250 m grid over Xinwu District. Unless noted otherwise, upstream logic lives in `Paper03/scripts`. Every column is ready to use without additional normalization.

### Urban form + land-use mix (`calc_urban_form.py`)

1. **`ci_norm`** – Compactness Index, i.e., the area-weighted gravitational clustering of building footprints inside a grid cell. Computed in `compute_ci_vci` and log-normalized between the 5th–95th percentiles.
2. **`vci_norm`** – Vertical Compactness Index, identical to CI but weighted by building volume (footprint × height) and 3D distance, then log-normalized.
3. **`lum_norm`** – Shannon-entropy land-use mix (residential/commercial/office/education/industrial) derived from 25 m sub-grid overlays, normalized to [0, 1].
4. **`lum_adjacency_norm`** – Share of 25 m sub-cells whose rook neighbors fall into the same use class; higher values mean more contiguous single-use clusters.
5. **`lum_intensity_norm`** – Dominance of the strongest land-use class (max area share) normalized between 0 and 1; high scores indicate mono-functional blocks.
6. **`lum_proximity_norm`** – Average inverse distance to the nearest *different* land-use class (1/(1+d)) aggregated over sub-cells, then normalized; highlights fine-grain mixing.
7. **`gi_norm`** – Mean global closeness centrality of the buffered road network nodes intersecting the cell (integration at city scale), normalized to [0,1].
8. **`li_norm`** – Mean local closeness centrality computed with a 500 m cut-off (captures block-scale accessibility), normalized to [0,1].

### Building program + land cover (`calc_built_environment.py`)

Floor-area-based variables are first summed in square meters and then divided by 10,000 to express hectares within each 250 m grid cell.

9. **`single_family_ha`** – Floor area used by single-family houses (building `type == 1`).  
10. **`multi_family_ha`** – Floor area used by multi-family houses (`type == 2`).  
11. **`facility_neighborhood_ha`** – Neighborhood/commercial service facilities (land-use classes 6/8/9, e.g., clinics, community centers).  
12. **`facility_sales_ha`** – Retail and sales facilities (land-use class 2).  
13. **`facility_office_ha`** – Office buildings (land-use class 1).  
14. **`facility_education_ha`** – Educational facilities (land-use class 7).  
15. **`facility_industrial_ha`** – Industrial facilities and plants (land-use class 3).  
16. **`parks_green_ha`** – Green/park areas (land-cover class 10) overlapped with the cell.  
17. **`water_area_ha`** – Inland water polygons intersecting the grid cell.  

### Transportation exposure (`calc_transportatio.py`)

18. **`road_area_ha`** – Area covered by buffered road centerlines (16 m full width) that fall inside the cell, converted to hectares.  
19. **`subway_influence_ha`** – Cell area within 250 m of a subway station (union of station buffers).  
20. **`bus_routes_cnt`** – Count of unique bus lines having at least one stop located inside the cell polygon.  

`grid_id_main` is retained in `data/dataset_20x3.csv` so you can join predictions back to spatial products, but it is not used as a feature.

## Targets (3 outputs)

All targets represent annual energy demand aggregated from the EnergyPlus simulations stored in `Paper03/Data/Processed/Energy/xinwu_energy_250m.gpkg` and normalized by each grid cell’s developed land area.

| Column | Description |
| --- | --- |
| `cooling_kwh_per_m2` | Total sensible cooling energy per square meter of developed land (kWh/m²·yr). |
| `heating_kwh_per_m2` | Space-heating demand per square meter (kWh/m²·yr). |
| `other_electricity_kwh_per_m2` | Remaining end uses (lighting, equipment, fans, misc. electric loads) per square meter (kWh/m²·yr). |

## 8/9/3 物理含义、量纲与归一化方式

下表用于论文描述，也可直接引用到方法部分。所有阈值均源自 `data/feature_scalers.json`，确保 translator/GA 在 Paper04 中构造新的场景时与 Paper03 模型一致。

### 8 个 Form 指标

| 指标 | 物理含义 | 原始量纲 | 归一化 |
| --- | --- | --- | --- |
| `ci_norm` | 建筑足迹两两的“引力”紧凑度：取每对建筑的面积乘积 (ha²) 除以平面距离 (km)，再对全体平均，量纲体现为面积-距离综合。 | ha²/km（经 log1p 处理） | 先对 `ci` 做 log1p，再按 5%/95% 分位截断至 `[0.038, 46.463]`，随后 min-max 映射至 [0,1]。 |
| `vci_norm` | 与 CI 相同，但面积乘以高度形成体量，距离改为三维距离，体现垂直聚合度。 | (ha·m)²/km（经 log1p 处理） | log1p 后按 5%/95% 分位截断至 `[0.132, 6.586]`，再做 min-max。 |
| `lum_norm` | 25 m 子网格的土地利用 Shannon 熵，描述用途多样性。 | 无量纲（0–ln5） | 直接以训练集的最小值 0 与最大值 0.842 做 min-max。 |
| `lum_adjacency_norm` | 子网格中与同类用地 rook 邻接的比例，衡量单一用途的连通性。 | 无量纲（0–1） | 原始指标范围 [1/3, 1]，直接 min-max 映射。 |
| `lum_intensity_norm` | 优势用地（面积占比最高的类别）的比例，数值越大表示越单一。 | 无量纲（0–1） | 直接 min-max。 |
| `lum_proximity_norm` | 子网格到最近异类用地的平均逆距离 `1/(1+d)`，体现功能混合的细粒度程度。 | 无量纲（0.0035–0.0278） | 直接 min-max。 |
| `gi_norm` | 道路网络的全局 closeness centrality（以米为单位），衡量节点到全域的平均最短路距离倒数。 | 1/m | 以 5%/95% 分位（0, 2.96×10⁻⁵）截断后 min-max。 |
| `li_norm` | 500 m 截止半径的局部 closeness centrality。 | 1/m | 以 5%/95% 分位（0, 4.30×10⁻³）截断后 min-max。 |

### 9 个 Built 指标

| 指标 | 物理含义 | 原始量纲 | 处理方式 |
| --- | --- | --- | --- |
| `single_family_ha` | 单户住宅建筑面积（建筑类型 `type=1`）。 | m² | 汇总后除以 10,000 → ha，仅截断负值。 |
| `multi_family_ha` | 多户住宅建筑面积 (`type=2`)。 | m² | → ha。 |
| `facility_neighborhood_ha` | 邻里/社区服务设施面积（土地分类 6/8/9）。 | m² | → ha。 |
| `facility_sales_ha` | 零售商业设施面积。 | m² | → ha。 |
| `facility_office_ha` | 办公设施面积。 | m² | → ha。 |
| `facility_education_ha` | 教育设施面积。 | m² | → ha。 |
| `facility_industrial_ha` | 工业设施面积。 | m² | → ha。 |
| `parks_green_ha` | 公园与绿地覆盖面积。 | m² | → ha。 |
| `water_area_ha` | 水体覆盖面积。 | m² | → ha。 |

所有建成环境变量没有进一步归一化，保留“每格多少公顷”的物理量纲，方便在协同优化中直观约束面积。

### 3 个交通指标

| 指标 | 物理含义 | 原始量纲 | 处理方式 |
| --- | --- | --- | --- |
| `road_area_ha` | 以 8 m 半宽缓冲的道路走廊面积与网格交集。 | m² | → ha。 |
| `subway_influence_ha` | 距地铁站 250 m 缓冲区与网格的覆盖面积。 | m² | → ha。 |
| `bus_routes_cnt` | 网格内独立公交线路数。 | 条 | 计数后取整数并截断负值。 |

> 若 translator/GA 以其它尺度生成候选指标，可通过更新 `data/feature_scalers.json`（或另行提供归一化参数）以保持尺度一致。

## Model training + performance

- Training script: `Paper03/Mapping/20x_3y/xgboost/train_xgboost.py`
- Dataset rows: 3,999 grid cells (4000 lines incl. header).
- Model spec: XGBRegressor (`tree_method="hist"`, depth 4, 600 trees) per target with 5-fold CV for reporting.
- Cross-validated metrics (from `data/xgb_metrics.json`):

| Target | RMSE (kWh/m²) | MAE (kWh/m²) | R² |
| --- | --- | --- | --- |
| Cooling | 2.63 | 1.33 | 0.938 |
| Heating | 3.08 | 1.15 | 0.968 |
| Other electricity | 2.74 | 1.41 | 0.837 |

Feature-importance CSV files (gain-based) are mirrored in `data/` and align with the SHAP study documented in `Paper03/Mapping/xai/Allmodel/shap_analysis.md`.

## Using the surrogate in Paper04

1. **激活/准备虚拟环境**  
   ```bash
   cd /home/dogtoree/Paper04
   source activate_surrogate.sh
   ```
   > 该脚本会复用 `Paper03/.venv_geo` 中已经配置好的 GeoEnergy 虚拟环境，因此无需重新安装依赖（环境中包含 pandas/numpy/xgboost 等包）。如果你更倾向于单独创建环境，可参考下列包清单手动安装。
2. **（可选）在新环境中安装依赖**  
   ```bash
   pip install pandas numpy xgboost
   ```
3. **Prepare input CSV** containing at least the 20 feature columns listed above. You may keep extra columns (e.g., `grid_id_main`); they will be preserved in the output.
4. **Run the helper script**:
   ```bash
   cd /home/dogtoree/Paper04/surrogate
   python predict_energy.py \
     --input data/dataset_20x3.csv \
     --output predictions.csv \
     --report-metrics
   ```
   The script validates required columns, loads `models/xgb_*.json`, appends `*_pred` columns, and optionally prints RMSE/MAE/R² if ground truth columns exist.

## 统一的 20 维特征计算接口

为方便 “translator” 与 GA 在 Paper04 中直接生成 surrogate 所需特征，我们在 `feature_pipeline.py` 中封装了完整的 8/9/3 处理流程（代码即复制了 Paper03 的归一化逻辑，并可复用 `data/feature_scalers.json` 中的阈值）。

```python
import pandas as pd
from feature_pipeline import compute_surrogate_features

urban_form_df = pd.read_csv("path/to/urban_form.csv")
built_env_df = pd.read_csv("path/to/built_env.csv")
transport_df = pd.read_csv("path/to/transport.csv")

features = compute_surrogate_features(urban_form_df, built_env_df, transport_df)
# features 包含 ['grid_id_main', ci_norm, ..., bus_routes_cnt] 共 21 列
```

### 输入要求

- `urban_form_df` 至少包含 `grid_id_main`，以及以下任一组合：  
  * 已归一化列：`ci_norm`, `vci_norm`, `lum_norm`, `lum_adjacency_norm`, `lum_intensity_norm`, `lum_proximity_norm`, `gi_norm`, `li_norm`；  
  * 或对应的原始列：`ci`, `vci`, `lum`, `lum_adjacency`, `lum_intensity`, `lum_proximity`, `gi_mean`, `li_mean`（函数会自动套用 `feature_scalers.json` 中的 log/分位裁剪参数）。  
- `built_env_df` 需提供 `grid_id_main`，以及任意 `_ha` 或 `_m2` 版本的 9 个建成环境面积列（函数会统一转换为 ha）。  
- `transport_df` 需包含 `grid_id_main`、`road_area_*`、`subway_influence_*`（m² 或 ha）以及 `bus_routes_cnt`。

函数最终返回 `['grid_id_main', *20 features]`，与 `data/dataset_20x3.csv` 完全对齐。在原始数据（Paper03 processed CSV）上测试时，生成结果与训练集 20 个特征的最大差异仅为 3.6×10⁻¹⁵（浮点误差），可放心用于新一轮的代理评估。

> 若 translator/GA 直接操作单个候选（而非整张表），可传入仅包含一行的 DataFrame；`compute_surrogate_features` 会保持输入顺序返回结果。

### Integrating with co-optimization workflows

- Treat `predict_energy.py` as a reference implementation. For programmatic use, import the module and call `load_models()` + `predict()` to embed the surrogate directly in your optimization loop.
- The JSON boosters can be loaded via the XGBoost Python API or any language binding that supports the `Booster` format (e.g., `xgboost4j`).
- If you generate new candidate forms, keep feature engineering consistent with the upstream scripts referenced above to avoid domain shift.

### Retraining or extending the surrogate

1. Update or regenerate the 20 indicators/targets within `Paper03`.
2. Run `python Mapping/20x_3y/xgboost/train_xgboost.py` from the Paper03 root.
3. Copy the new `dataset.csv`, `xgb_metrics.json`, `xgb_feature_importance_*.csv`, and `xgb_*.json` files into this `surrogate/` directory (overwriting the existing artifacts).

This package now centralizes documentation + inference utilities so Paper04 can call the surrogate without re-opening the Paper03 workspace.
