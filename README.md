# AlphaLab

一个面向A股市场的完整多因子量化研究框架。从原始股票数据出发，依次完成因子计算、因子评价、因子合成、行业轮动、ETF组合构建与回测，并提供完整的稳健性检验模块。

---

## 项目简介

本项目以沪深300成分股为股票池，构建了一套从数据获取到策略回测的完整量化流程：

- 计算101+个Alpha因子，并通过IC分析和分层测试筛选有效因子
- 使用IR加权方式合成综合因子
- 将股票层面的因子信号聚合为申万一级行业得分
- 根据行业得分轮动配置对应的行业ETF
- 回测引擎支持T+1执行、手续费、滑点、风控等完整的交易模拟
- 通过参数敏感性、滚动窗口和蒙特卡洛等方法验证策略稳健性

---

## 整体流程

```
原始股票数据（沪深300）
        ↓
Alpha因子计算（101个 / 191个 / 外部因子）
        ↓
因子预处理（MAD去极值 / 行业市值中性化 / 标准化）
        ↓
因子评价（IC分析 / 分层测试）
        ↓
因子合成（IR加权）
        ↓
行业得分聚合（Top-K均值）
        ↓
ETF组合构建（等权 / 得分加权 / 风险平价）
        ↓
回测引擎（T+1执行 / 风控 / 交易成本）
        ↓
绩效评估与稳健性检验
```

---

## 功能特性

- **因子计算**：支持101个WorldQuant风格Alpha因子、191个扩展因子及外部基本面因子
- **因子评价**：Spearman IC均值、ICIR计算，支持日/周/月频
- **分层测试**：五分位分层收益分析，验证因子单调性
- **行业轮动**：将股票因子信号聚合至申万一级30个行业
- **ETF映射**：行业得分直接映射至可交易的A股行业ETF
- **组合构建**：支持等权、得分加权、风险平价三种权重方式，单行业权重上限40%
- **风险控制**：单行业止损、组合回撤保护、波动率过滤、流动性保护
- **交易成本**：万分之一佣金、1个tick滑点、日均成交额5%冲击成本限制
- **T+1执行**：信号T日生成，T+1日开盘价执行
- **稳健性检验**：参数敏感性测试、滚动窗口样本外验证、蒙特卡洛模拟、不同市况分析

---

## 文件结构

```
AlphaLab/
├── config.py                     # 全局配置与路径管理
├── utils.py                      # 公共工具函数
├── rename.py                     # ETF数据文件重命名工具
│
├── data_fetcher.py               # 通过Insight SDK获取原始数据
├── datas.py                      # 数据管理与存储
├── data_transformer.py           # 原始数据预处理与转换
│
├── factor_alphas.py              # 101个Alpha因子定义
├── factor_alphas_191.py          # 191个扩展Alpha因子定义
├── factor_utils.py               # 因子工具函数（ts_rank、decay_linear等）
├── factor_utils_191.py           # 191因子专用工具函数
├── factor_calculator.py          # 因子计算主流程
├── factor_calculator_191.py      # 191因子计算流程
├── factor_calculator_external.py # 外部/基本面因子计算
│
├── factor_portfolio_DWM.py       # 因子合成（支持日/周/月频）
│
├── ic_analysis_DWM.py            # IC分析（支持日/周/月频）
├── layer_analysis_DWM.py         # 分层测试（支持日/周/月频）
│
├── industry_rotation.py          # 股票因子 → 行业得分聚合
├── etf_mapping.py                # 行业 → ETF代码映射表(需自己完善)
├── portfolio_construction.py     # 行业得分 → ETF目标权重
│
├── backtest_DWM_ETF_终版.py      # 回测引擎
│
└── robustness_test.py            # 稳健性与参数敏感性测试
```

---

## 环境依赖

```
python >= 3.9
pandas
numpy
scipy
matplotlib
seaborn
py7zr
python-dateutil
```

安装依赖：
```bash
pip install pandas numpy scipy matplotlib seaborn py7zr python-dateutil
```

---

## 使用步骤

```bash
# 第一步：获取原始数据（需要Insight SDK账号）
python data_fetcher.py

# 第二步：计算Alpha因子
python factor_calculator.py          # 101个因子
python factor_calculator_191.py      # 191个因子（可选）
python factor_calculator_external.py # 外部因子（可选）

# 第三步：数据格式转换
python data_transformer.py

# 第四步：因子评价
python ic_analysis_DWM.py
python layer_analysis_DWM.py

# 第五步：因子合成（选择调仓频率）
python factor_portfolio_DWM.py       # 在文件内设置 freq = 'D' / 'W' / 'M'

# 第六步：运行ETF回测
python backtest_DWM_ETF_终版.py

# 第七步：稳健性检验
python robustness_test.py
```

---

## 配置说明

修改 `config.py` 中的关键参数：

```python
# 数据时间范围
START_DATE       = "2020-01-14"   # 因子计算起始日
START_TEST_DATE  = "2022-01-14"   # 回测起始日
END_DATE         = "2026-02-20"   # 回测结束日

# 数据路径
STOCK_DAILY_DIR  = "data_bfq"                      # 股票日线数据目录
ETF_DAILY_DIR    = r"D:\ETF数据\行情数据-1d"        # ETF日线数据目录
RESULT_DIR       = "factor_sum_results"             # 结果输出目录
```

---

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `freq` | `'W'` | 调仓频率：`'D'`日频 / `'W'`周频 / `'M'`月频 |
| `n_top` | `5` | 持仓行业数量（3-5个） |
| `weighting` | `'score'` | 权重方式：`'equal'` / `'score'` / `'risk_parity'` |
| `commission` | `0.0001` | 单边佣金率（万分之一） |
| `min_holding_days` | `2` | 最小持仓期（交易日） |
| `stop_loss_single` | `0.10` | 单行业止损阈值 |
| `drawdown_half` | `0.25` | 组合回撤超过此值降仓至50% |
| `drawdown_clear` | `0.40` | 组合回撤超过此值全部平仓 |

---

## 数据来源

- **股票数据**：Insight SDK（沪深300成分股日线行情、行业分类）
- **ETF数据**：本地1分钟行情压缩包，聚合为日线数据
- **基准**：沪深300指数（000300）

---

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。历史回测表现不代表未来实际收益。
