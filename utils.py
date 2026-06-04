# utils.py
import pandas as pd
import os
from datetime import datetime
 # 公共辅助函数（日期处理、数据转换等）

def ensure_dir(path):
    """确保目录存在，若不存在则创建"""
    os.makedirs(path, exist_ok=True)

def normalize_date(df, date_col='date'):
    """将日期列标准化为 YYYY-MM-DD 字符串格式"""
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    return df

def strip_stock_suffix(code_series):
    """去除股票代码中的 .SH/.SZ 后缀"""
    return code_series.astype(str).str.split('.').str[0]