# -*- coding: utf-8 -*-
"""主程序入口"""
import os
import glob
from datetime import datetime
import pandas as pd
import akshare as ak
from tqdm import tqdm
import numpy as np
from config.config import MODEL_DIR, FEATURE_COLS
from data.data_fetcher import get_stock_data
from models.model_trainer import train_and_save_model


def main():
    """程序主函数"""
    print("""
 █████╗       ███████╗██╗  ██╗ █████╗ ██████╗ ███████╗
██╔══██╗      ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔════╝
███████║█████╗███████╗███████║███████║██████╔╝█████╗  
██╔══██║╚════╝╚════██║██╔══██║██╔══██║██╔══██╗██╔══╝  
██║  ██║      ███████║██║  ██║██║  ██║██║  ██║███████╗
╚═╝  ╚═╝      ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝                  
         🚀 启动A股智能选股系统 v1.0
    """)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 缓存处理
    load_cache = input("是否加载缓存模型？(y/n): ").lower() == 'y'
    if load_cache:
        model_files = glob.glob(os.path.join(MODEL_DIR, '*_model.txt'))
        if model_files:
            latest_mtime = max(os.path.getmtime(f) for f in model_files)
            print(f"最新缓存模型更新时间：{datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("⚠️ 未找到缓存模型，将重新训练")
            load_cache = False

    # 获取股票列表
    stock_list = ak.stock_zh_a_spot_em().rename(columns={
        '代码': 'code',
        '名称': 'name',
        '最新价': 'price',
        '涨跌幅': 'change_pct'
    })
    stock_list['code'] = stock_list['code'].apply(lambda x: str(x).zfill(6))
    stock_list = stock_list[~stock_list['name'].str.contains('ST|退')]
    stock_list = stock_list[~stock_list['code'].str.startswith(('300', '688', '8'))]
    #stock_list = stock_list[:10] #调试时取消注释加快测试速度

    results = []
    pbar = tqdm(stock_list['code'], desc="处理股票", ncols=100)

    for code in pbar:
        pbar.set_postfix_str(f"正在处理：{code}")
        try:
            # 获取并训练模型
            booster = train_and_save_model(code, force_retrain=not load_cache)
            if not booster:
                continue

            # 获取最新数据
            df = get_stock_data(code)
            if df is None or len(df) < 500:
                continue

            # 生成预测
            X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            latest_features = X.iloc[[-1]].values.astype(np.float32)
            prob = booster.predict(latest_features)[0]

            # 记录结果
            latest_pct = df['pct_chg'].iloc[-1]
            results.append({
                '代码': code,
                '名称': stock_list.loc[stock_list['code'] == code, 'name'].values[0],
                '是否涨停': "是" if latest_pct >= 9.9 else "否",
                '预测概率': prob,
                '收盘价': df['close'].iloc[-1],
                '更新日期': datetime.today().strftime('%Y-%m-%d')
            })

        except Exception as e:
            print(f"\n处理{code}时发生错误: {str(e)}")
            continue

    # 生成推荐结果
    if results:
        result_df = pd.DataFrame(results)
        result_df['推荐评级'] = pd.cut(result_df['预测概率'],
                                       bins=[0, 0.6, 0.75, 1],
                                       labels=['C', 'B', 'A'])
        result_df = result_df[['代码', '名称', '是否涨停', '预测概率', '推荐评级', '收盘价', '更新日期']]

        result_file = f'stock_recommend_{datetime.today().strftime("%Y%m%d")}.xlsx'
        result_df.to_excel(result_file, index=False, engine='openpyxl')
        print(f"\n✅ 分析完成！共处理{len(results)}只股票，推荐结果已保存至 {result_file}")
        print(result_df.head(10))
    else:
        print("\n⚠️ 未找到有效股票数据，请检查数据源或过滤条件")


if __name__ == '__main__':
    main()