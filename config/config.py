# -*- coding: utf-8 -*-
"""全局配置参数模块"""

import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 特征列配置
FEATURE_COLS = [
    'close', 'volume', 'pct_chg',
    'EMA20_ratio', 'RSI14', 'MACD',
    'OBV', 'CCI', 'ATR', 'ADX'
]

# 预测目标天数
TARGET_DAYS = 2

# 模型存储目录
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'stock_models')