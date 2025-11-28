# RealDis

代表方案与真实 1 km 区块的匹配工具。

## 数据准备
- `dataset.csv`（未公开）：250 m 栅格特征，示例见 `dataset.csv.example`。假设 `grid_id_main` 连续编号，每 16 个格子组成一个 1 km block（block_id = grid_id_main // 16）。
- 必要字段示例：`grid_id_main`, `cooling_kwh_per_m2`, `heating_kwh_per_m2`, `other_electricity_kwh_per_m2`, `multi_family_ha`, `facility_industrial_ha`, `road_area_ha`, `parks_green_ha` 等。

## 匹配脚本
- `match_reals.py`：读取代表解（`CoOpt/results/final_outputs/representatives_grids/*.json`）与真实 block，做特征标准化（z-score），计算欧氏距离，输出 top-N 相似 block。
- 运行：`python RealDis/match_reals.py`（要求已准备好 `dataset.csv`）。
- 输出：`match_results.csv`，包含每个代表解的前 5 个最相似的真实 block 及其特征。

## 注意
- 本目录不包含真实数据，仅提供占位示例。使用者需自行准备合法的 250 m 数据。
