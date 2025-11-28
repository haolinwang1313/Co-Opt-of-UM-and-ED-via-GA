# Co-Opt-of-UM-and-ED-via-GA

开源版仅包含核心代码与可公开的示例/占位数据，不含原始数据、可视化产出、工作日志。

## 内容概览
- `CoOpt/`：基于 NSGA-II 的城市形态优化流程，包含参数配置、解码与评估调用、前沿分析等脚本。
- `GeneGenera/`：基因层定义、解码与评估的底层实现。
- `surrogate/`：代理模型训练/推理代码（模型权重可公开的放在 `models/` 或提供下载链接）。
- `RealDis/`：代表方案与真实 1 km 区块的匹配脚本（提供 `dataset.csv.example` 占位）。
- `RawData/`：仅留 README 说明，原始数据未公开。

未包含的内容：
- `Visualization/` 下的可视化脚本与生成图像（不公开）。
- `work_logs/` 工作日志（不公开）。
- `RawData/` 中的原始矢量/栅格数据（不公开）。

## 环境与依赖
- Python 3.9+，建议使用 conda/venv 管理。
- 主要依赖：numpy、pandas、matplotlib、seaborn、scikit-learn、geopandas（如需处理 GPKG）。
- 可根据需要补充 `requirements.txt` 或 `environment.yml`。

## 基本运行流程
1. **代理模型**（可选）  
   - 在 `surrogate/` 下训练或加载已有模型（如有权重或下载链接，放置于 `surrogate/models/`）。
2. **优化（NSGA-II）**  
   - 参考 `CoOpt/config.py` 配置参数范围与场景（S1/S2）。  
   - 小规模试跑：`python CoOpt/nsga_small.py`。  
   - 正式运行（多代/多种群）：`python CoOpt/run_nsga.py` 或 `run_multi_seeds.py`。  
   - 前沿分析：`python CoOpt/analyze_frontier.py`。
3. **真实区块匹配**  
   - 在 `RealDis/` 准备 `dataset.csv`（结构见 `dataset.csv.example`），每 16 个连续 `grid_id_main` 组成 1 km block。  
   - 运行：`python RealDis/match_reals.py`，生成 `match_results.csv`（代表解与真实区块的相似度 Top-N）。

## 数据说明
- 原始数据与可视化产出未公开。  
- `RealDis/dataset.csv.example` 仅提供字段示例；真实数据需根据你自己的 250m 栅格数据生成。  
- 若需重新生成代理模型训练数据，请使用你持有的数据按 surrogate 流程预处理。

## 许可
- 本仓库使用 MIT 许可证（见 LICENSE）。  
- 数据部分如有额外限制，请遵循数据提供方的许可与合规要求。
