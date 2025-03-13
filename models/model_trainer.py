# -*- coding: utf-8 -*-
"""模型训练与优化模块"""
import os
import lightgbm as lgb
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import logging
from config.config import MODEL_DIR, FEATURE_COLS
from data.data_fetcher import get_stock_data
import warnings

warnings.filterwarnings("ignore")

def optimize_hyperparameters(X, y):
    """
    使用Optuna进行超参数优化

    Args:
        X (pd.DataFrame): 特征数据
        y (pd.Series): 目标变量

    Returns:
        dict: 最佳超参数
    """
    logging.getLogger('optuna').setLevel(logging.ERROR)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='auc',
                      verbose= False,
                      callbacks=[])

            y_pred = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    return study.best_params


def train_and_save_model(code, force_retrain=False):
    """
    训练并保存模型

    Args:
        code (str): 股票代码
        force_retrain (bool): 是否强制重新训练

    Returns:
        lgb.Booster: 训练好的模型，或None（如果训练失败）
    """
    model_path = os.path.join(MODEL_DIR, f'{code}_model.txt')

    # 尝试加载已有模型
    if not force_retrain and os.path.exists(model_path):
        try:
            return lgb.Booster(model_file=model_path)
        except Exception as e:
            print(f"模型{code}加载失败，重新训练... 错误：{str(e)}")

    # 获取训练数据
    df = get_stock_data(code)
    if df is None or len(df) < 500:
        return None

    try:
        # 数据预处理
        df = df.dropna(subset=['pct_chg_5d'])
        X = df[FEATURE_COLS]
        y = (df['pct_chg_5d'] > 0).astype(int)

        # 时间序列分割
        split_idx = int(len(X) * 0.8)
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

        # 超参数优化
        best_params = optimize_hyperparameters(X_train, y_train)

        # 全量数据训练
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X, y, eval_metric='auc', callbacks=[])

        # 保存模型
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.booster_.save_model(model_path)
        return model.booster_

    except Exception as e:
        print(f"训练{code}模型失败: {str(e)}")
        return None
