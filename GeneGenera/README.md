# GeneGenera 模块

该目录用于定义 1 km 研究窗口中 16 个 250 m 网格的基因结构，并提供“基因 → 20 个代理模型特征”的译码框架。目标是复用 `surrogate/feature_pipeline.py` 中的输入逻辑，但通过可解释的分层/混合编码生成这些输入，而不是直接操作 20 个特征。

## 数据依赖

- `RawData/gpkg/xinwu_grid_250m.gpkg`: 获取 grid_id_main、边界和面积（默认为 250×250 m）。
- `RawData/gpkg/xinwu_urban_form_250m.gpkg`: 作为 `ci`、`vci`、`lum_*` 的基线统计，且用于校准形态模板。
- `RawData/gpkg/xinwu_built_env_250m.gpkg`: 提供各类建筑面积（m²），用来推导单/多户、公共设施、产业面积等基线值。
- `RawData/gpkg/xinwu_transportatio_250m.gpkg` + `xinwu_road_centerlines.gpkg` + `xinwu_bus_network.gpkg` + `xinwu_subway_stations.gpkg`: 作为交通供给指标（road_area、subway_influence、bus_routes）基线。
- `surrogate/data/dataset_20x3.csv`: 包含 20 个特征 + 3 个能耗输出；译码器初期可直接读取该表作为 baseline。

## 分层基因模块

| 模块 | 变量 | 作用 | 影响的代理特征 |
| --- | --- | --- | --- |
| M1 形态模板 | `shape_id`, `orientation_deg`, `coverage_factor` | 选择建筑底图原型并调整覆盖率/FAR | `ci_norm`, `lum_adjacency_norm`, `lum_proximity_norm` 等 |
| M2 垂向密度 | `floor_base`, `floor_gradient`, `podium_ratio` | 控制楼层数/裙房，对体形系数与体量分配加权 | `vci_norm`, `built_env` 九项面积、`FAR`/`BSF` 约束 |
| M3 土地功能配比 | `mix_seed` (3 元素)，`res_type_split`, `service_focus` | 将总建筑面积分摊到住宅/混合/产业/公共设施 | `lum_*`、`single_family_ha`, `multi_family_ha`, `facility_*_ha` |
| M4 生态与水体 | `green_ratio`, `blue_ratio`, `green_pattern` | 控制开放绿地/水体面积及空间模式 | `parks_green_ha`, `water_area_ha`, `lum_adjacency_norm` |
| M5 交通供给 | `road_width_factor`, `road_grid_density`, `bus_route_toggle`, `transit_hub_flag` | 调整道路宽度/密度、公交线路、新增地铁口 | `road_area_ha`, `subway_influence_ha`, `bus_routes_cnt`, `gi_norm`, `li_norm` |
| M6 窗口级协调 | `target_far`, `industrial_quota`, `green_network_continuity`, `bus_ring_strength` | 约束 16 个网格总体 FAR/产业占比/绿地连通/公交成环度 | 通过归一化系数影响所有单元的 m² 分配与交通增量 |

## 译码流程

1. **初始化 baseline**：读取 `surrogate/data/dataset_20x3.csv`，以 `grid_id_main` 为索引，作为所有特征的基线值。
2. **解析染色体**：将 1×(16×CellGene + GlobalGene) 的向量映射到 dataclass。CellGene 附带 `grid_id` 以保证与 baseline 对齐。
3. **计算中间量**：
   - 以 `coverage_factor × floor_base` 估算 raw FAR，并根据 `GlobalGene.target_far` 生成缩放系数，使 16 个网格的平均 FAR 满足目标；
   - 用 `mix_seed` 通过 Dirichlet 映射得到 {R, M, B, P} 四类功能占比；
   - 结合 `res_type_split`、`service_focus` 把总可建面积拆分为 9 类 built_env 指标；
   - `green_ratio/blue_ratio/green_pattern` 决定开放空间面积及其对 `lum_adjacency_norm` 的加成；
   - 交通模块在 baseline 指标上执行缩放或增量，并以 `bus_ring_strength` 控制线路连接性。
4. **生成 20 特征表**：对每个 250 m 网格输出包含 `grid_id_main` + 20 列的 DataFrame，可直接送入 `feature_pipeline.compute_surrogate_features` 做校验或直接喂给 surrogate。
5. **约束与罚项**：若任何特征超出 `dataset_20x3.csv` 的 1–99 百分位，记录到日志供 GA 作为约束或罚分，确保结果不偏离训练分布。

## 当前实现

- `decoder.py`：实现 `CellGene`、`GlobalGene` dataclass 和 `GeneDecoder`，支持：
  - 从 baseline 中加载 20 个特征；
  - 应用分层基因模块生成新的特征；
  - 对输出做裁剪与格式化。
- `shape_effects` 等常量为占位符，后续可通过真实建筑库或 VAE 模板重估。

## 后续工作

1. **模板库标定**：使用 `RawData/gpkg/xinwu_buildings.gpkg` 抽取代表性轮廓、体量和 land-use 区块，替换 `decoder.py` 中的默认 shape effect。
2. **与 25 m 数据联动**：当需要更精准的 lum/交通指标时，将译码器输出的几何写入新的 25 m 栅格，再调用 Paper03 的指标脚本重新计算。
3. **GA 集成**：包装 `GeneDecoder.decode()` 供 NSGA-II 调用（例如 `modeling/define_var.py` 中的 translator），形成 “染色体 → 特征 → surrogate 预测” 的全链路。

### 校准工具

- `calibrate.py`：读取 `surrogate/data/dataset_20x3.csv`，自动推导形态模板权重、服务用地权重与绿地模式系数，结果写入 `calibration.json`。译码器会在运行时自动加载该文件。运行前需安装 `scikit-learn`。
- `evaluator.py`：封装 `GeneDecoder` 与 surrogate 预测，输入一组 `CellGene`/`GlobalGene`，输出 20 个特征、三类能耗预测、1 km 级汇总指标及超界特征提醒。首次运行需安装 `xgboost`。
