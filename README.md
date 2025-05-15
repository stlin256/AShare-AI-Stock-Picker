# A-shares Intelligent Stock Picker v1.0
<a href="#cn">中文介绍</a>
## Project Overview
This is a machine learning-based A-shares stock selection system designed to help users make investment decisions by predicting future stock price trends. The system utilizes the LightGBM model combined with multiple technical indicators for prediction, and employs Optuna for hyperparameter optimization to ensure optimal model performance.

### Core Features
- **Data Acquisition**: Retrieve A-shares historical data from `akshare`  
- **Feature Engineering**: Calculate multiple technical indicators such as RSI, MACD, OBV  
- **Model Training**: Perform binary classification prediction (stock price movement over 5 days) using LightGBM  
- **Hyperparameter Optimization**: Automatic parameter optimization through Optuna  
- **Model Caching**: Support model saving/loading to accelerate subsequent runs  
- **Prediction & Recommendations**: Generate stock recommendation ratings (A/B/C) and export to Excel  

## Technology Stack & Dependencies
- **Python Version**: Recommended 3.12  
- **Key Libraries**:  
  - `akshare`: Retrieve stock historical data  
  - `talib`: Calculate technical indicators  
  - `lightgbm`: Machine learning model training  
  - `optuna`: Hyperparameter optimization  
  - `pandas`/`numpy`: Data processing and analysis  
  - `sklearn`: Cross-validation & model evaluation  
  - `tqdm`: Display training/prediction progress bars  
  - `openpyxl`: Export results to Excel  

### Installation
Run the following commands in the project root directory:  
```bash
pip install requirements.txt -r
pip install TA_Lib-0.4.28-cp312-cp312-win_amd64.whl
```

## Directory Structure
```text
stock_prediction_project/
├── config/
│   └── config.py          # Global configuration parameters (e.g., feature columns, model paths)
├── data/
│   └── stock_models/      # Model cache directory
├── features/
│   └── feature_engineering.py  # Feature engineering & technical indicator calculations
├── models/
│   ├── __init__.py
│   └── model_trainer.py   # Model training & optimization logic
├── main.py                # Main program entry point
└── README.md              # Project documentation
```

## Function Modules

### Data Acquisition
- **Source**: A-shares historical data via `akshare`  
- **Preprocessing**: Column renaming, filtering invalid data (missing values or outliers)  
- **Custom Range**: Support user-defined time range specification  

### Feature Engineering
- **Technical Indicators**: Calculate common indicators including:  
  - RSI (Relative Strength Index)  
  - MACD (Moving Average Convergence/Divergence)  
  - OBV (On-Balance Volume)  
  - CCI (Commodity Channel Index)  
  - ATR (Average True Range)  
  - ADX (Average Directional Index)  
- **Prediction Target**: Generate 5-day future returns (pct_chg_5d) as the model target variable  

### Model Training
- **Model**: LightGBM for binary classification (predicting positive/negative 5-day returns)  
- **Optimization**: Bayesian hyperparameter optimization via Optuna  
- **Caching**: Trained models saved in `data/stock_models/` for reuse  

### Prediction & Recommendations
- **Prediction Flow**: Predict price movement probability for each stock  
- **Recommendation Ratings**:  
  - **Grade A**: Probability ≥ 0.75 (High-confidence gain)  
  - **Grade B**: 0.6 ≤ Probability < 0.75 (Medium-confidence gain)  
  - **Grade C**: Probability < 0.6 (Low-confidence gain)  
- **Output**: Excel file containing stock code, name, limit-up status, prediction probability, and recommendation grade  

## Usage Instructions

### Runtime Environment
- Ensure Python 3.12 and all dependencies are installed  
- Virtual environment recommended to avoid dependency conflicts  

### Execution Steps
1. Clone/download the project:  
   ```bash
   git clone https://github.com/stlin256/AShare-AI-Stock-Picker.git  
   ```

2. Install dependencies (see "Technology Stack" section)

3. Run the main program:  
   ```bash
   python main.py
   ```

4. Choose whether to load cached models:  
   - First-time run requires model training (~5h on 8C16T R9-7940hs)  
   - Subsequent runs can load existing models  

5. System will automatically process data and generate recommendations  

### Output Files
- Saved as `stock_recommend_YYYYMMDD.xlsx` with date stamp  
- Example output:  

  | 代码   | 名称     | 是否涨停 | 预测概率 | 推荐评级 | 收盘价  | 更新日期   |
  | ------ | -------- | -------- | -------- | -------- | ------- | ---------- |
  | 600519 | 贵州茅台 | 否       | 0.464573218     | C        | 1487.02 | 2025-03-03 |

## Model & Prediction
- **Model Selection**: LightGBM - efficient gradient boosting tree model for large datasets  
- **Prediction Target**: Binary classification of 5-day returns (1: gain, 0: loss)  
- **Rating Criteria**:  
  - Grade A: High-confidence gain  
  - Grade B: Medium-confidence gain  
  - Grade C: Low-confidence gain  

## Notes
- **Data Updates**: Stock data changes rapidly - regular program execution recommended  
- **Model Updates**: Force retraining via `force_retrain=True` in main.py  
- **Stock Filtering**: Default filters for ST stocks, delisted stocks, and STAR/GEM stocks (adjustable in main.py)  

## Contribution & Feedback
- **Code Contributions**: Submit optimizations or new features via Pull Requests  
- **Bug Reports**: Use GitHub Issues for feedback and suggestions  

---

<div id="cn"></div>

# A股智能选股系统 v1.0

## 项目概述
这是一个基于机器学习的 A 股选股系统，旨在通过预测股票未来涨跌趋势帮助用户进行投资决策。系统利用 LightGBM 模型结合多种技术指标进行预测，并通过 Optuna 进行超参数优化，确保模型性能最佳。

### 核心功能
- **数据获取**：从 `akshare` 获取 A 股历史数据。
- **特征工程**：计算多种技术指标，如 RSI、MACD、OBV 等。
- **模型训练**：使用 LightGBM 进行二分类预测（股票未来5日涨跌）。
- **超参数优化**：通过 Optuna 自动优化模型参数。
- **模型缓存**：支持模型保存和加载，加速后续运行。
- **预测与推荐**：生成股票推荐评级（A/B/C）并导出为 Excel 文件。

## 技术栈与依赖
- **Python 版本**：推荐3.12。
- **主要库**：
  - `akshare`：用于获取股票历史数据。
  - `talib`：计算技术指标。
  - `lightgbm`：机器学习模型训练。
  - `optuna`：超参数优化。
  - `pandas`/`numpy`：数据处理和分析。
  - `sklearn`：交叉验证与模型评估。
  - `tqdm`：显示训练和预测进度条。
  - `openpyxl`：支持导出结果到 Excel 文件。

### 安装依赖
在项目根目录下运行以下命令安装所需库：
```bash
pip install requirements.txt -r
pip install TA_Lib-0.4.28-cp312-cp312-win_amd64.whl
```

目录结构

```text
stock_prediction_project/
├── config/
│   └── config.py          # 全局配置参数（如特征列、模型路径）
├── data/
│   └── stock_models/      # 模型缓存文件夹
├── features/
│   └── feature_engineering.py  # 特征工程和技术指标计算
├── models/
│   ├── __init__.py
│   └── model_trainer.py   # 模型训练和优化逻辑
├── main.py                # 主程序入口
└── README.md              # 项目说明文档
```

功能模块详解

数据获取

- **来源**：通过 akshare 获取 A 股历史数据。
- **预处理**：包括重命名列、过滤无效数据（如缺失值或异常值）等。
- **自定义范围**：支持用户指定数据的时间范围。

特征工程

- **技术指标**：计算多种常见技术指标，包括但不限于：
  - RSI（相对强弱指数）
  - MACD（指数平滑异同平均线）
  - OBV（成交量净额）
  - CCI（商品通道指数）
  - ATR（真实波幅）
  - ADX（平均趋向指数）
- **预测目标**：生成未来5日收益率（pct_chg_5d）作为模型的目标变量。

模型训练

- **模型**：使用 LightGBM 进行二分类预测（预测股票未来5日收益率是否为正）。
- **优化**：通过 Optuna 进行贝叶斯超参数优化，提升模型性能。
- **缓存**：训练好的模型保存在 data/stock_models/ 目录下，支持加载已有模型以避免重复训练。

预测与推荐

- **预测流程**：对每只股票进行预测，输出涨跌概率。
- **推荐评级**：
  - **A级**：预测概率 ≥ 0.75（高概率上涨）
  - **B级**：0.6 ≤ 预测概率 < 0.75（中等概率上涨）
  - **C级**：预测概率 < 0.6（低概率上涨）
- **输出**：结果导出为 Excel 文件，包含股票代码、名称、是否涨停、预测概率、推荐评级等信息。

使用说明

运行环境

- 确保安装 Python 3.12 及所有依赖库。

- 建议使用虚拟环境以避免依赖冲突：

运行步骤

1. 克隆或下载项目到本地：

   ```bash
   git clone https://github.com/stlin256/AShare-AI-Stock-Picker.git
   ```

2. 安装依赖（见“技术栈与依赖”部分）。

3. 运行主程序：

   ```bash
   python main.py
   ```

4. 根据提示选择是否加载缓存模型：

   - 首次运行需训练模型，耗时较长。(使用8C16T的R9-7940hs进行训练，耗时约5h)
   - 非首次运行可加载已有模型，加速流程。

5. 系统将自动处理数据并生成推荐结果。

输出文件

- 推荐结果保存为 stock_recommend_YYYYMMDD.xlsx，文件名包含日期戳。

- 示例输出：

  | 代码   | 名称     | 是否涨停 | 预测概率 | 推荐评级 | 收盘价  | 更新日期   |
  | ------ | -------- | -------- | -------- | -------- | ------- | ---------- |
  | 600519 | 贵州茅台 | 否       | 0.464573218     | C        | 1487.02 | 2025-03-03 |

模型与预测

- **模型选择**：LightGBM，一种高效的梯度提升树模型，适合处理大规模数据集。
- **预测目标**：股票未来5日收益率是否为正（1：涨，0：跌）。
- **评级标准**：
  - A级：高置信度上涨。
  - B级：中等置信度上涨。
  - C级：低置信度上涨。

注意事项

- **数据更新**：股票数据实时变化，建议定期运行程序以获取最新预测结果。
- **模型更新**：如需强制重新训练模型，可在 main.py 中设置 force_retrain=True。
- **股票过滤**：默认过滤 ST、退市及科创板/创业板股票，用户可在 main.py 中调整过滤条件。

贡献与反馈

- **代码贡献**：欢迎通过 Pull Request 提交代码优化或新增功能。
- **问题反馈**：请通过 GitHub Issues 报告 bug 或提出建议。


------

