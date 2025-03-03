# -*- coding: utf-8 -*-
"""数据获取模块"""
import pandas as pd
import akshare as ak
from features.feature_engineering import calculate_technical_indicators


def get_stock_data(code):
    """
    获取股票数据并计算技术指标

    Args:
        code (str): 股票代码

    Returns:
        pd.DataFrame: 包含技术指标的股票数据
    """
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
            '最低': 'low', '成交量': 'volume', '涨跌幅': 'pct_chg'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # 计算技术指标
        df = calculate_technical_indicators(df)
        return df
    except Exception as e:
        print(f"获取股票{code}数据失败: {str(e)}")
        return None