# -*- coding: utf-8 -*-
"""特征工程和技术指标计算模块"""
import pandas as pd
import numpy as np
import talib


def calculate_technical_indicators(df):
    """
    计算技术指标并添加到数据框

    Args:
        df (pd.DataFrame): 包含OHLCV数据的DataFrame

    Returns:
        pd.DataFrame: 包含技术指标的DataFrame
    """
    df = df.copy()

    # 计算EMA20_ratio
    df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
    df['EMA20_ratio'] = df['close'] / df['EMA20']

    # 计算RSI
    df['RSI14'] = talib.RSI(df['close'], timeperiod=14)

    # 计算MACD
    macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd

    # 计算OBV
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    # 计算CCI
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

    # 计算ATR
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # 计算ADX
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # 计算未来5日收益率作为目标
    df['pct_chg_5d'] = df['close'].pct_change(periods=5).shift(-5)

    return df