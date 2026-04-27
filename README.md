# 基于深度强化学习的最优清算策略研究

本项目实现了一个可复现的最优清算实验框架。在 Almgren-Chriss 风格的市场冲击模型下，项目将限定期限内的大额资产卖出问题建模为连续控制任务，并比较以下四类执行策略：

- Almgren-Chriss 理论基准
- TWAP 等量清算基准
- DDPG
- PPO

实验流水线支持智能体训练、统一评估、参数敏感性分析、多随机种子对比，并导出 CSV 表格与 PNG 图表。

## 项目结构

```text
optimal-liquidation-drl/
|-- src/
|   |-- config.py
|   |-- env_execution.py
|   |-- execution_reference.py
|   |-- baseline_ac.py
|   |-- baseline_twap.py
|   |-- agent_ddpg.py
|   |-- agent_ppo.py
|   |-- experiment_runner.py
|   `-- utils.py
|-- results/
|   |-- *.csv
|   `-- *.png
|-- requirements.txt
`-- README.md
```

## 环境安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 macOS 或 Linux，虚拟环境激活命令为：

```bash
source .venv/bin/activate
```

## 运行实验

```bash
cd src
python experiment_runner.py
```

默认实验参数位于 `src/config.py`。运行后会在 `results/` 下生成训练日志、方法对比结果、敏感性分析结果、图表、模型检查点和版本化输出。其中检查点和版本化输出已在 `.gitignore` 中忽略，避免大体积生成文件进入版本库。

## 已包含结果

`results/` 目录中保留了一份已清理的代表性实验结果，便于直接查看：

- `eval_compare.csv`：各方法的汇总评估指标
- `eval_compare_detail.csv`：逐回合评估明细
- `train_ddpg.csv` 和 `train_ppo.csv`：训练日志
- `multi_seed_compare.csv` 和 `multi_seed_summary.csv`：多随机种子对比结果
- `sensitivity_lambda.csv` 和 `sensitivity_fee.csv`：参数敏感性分析结果
- `*.png`：训练曲线、方法对比图和敏感性分析图

