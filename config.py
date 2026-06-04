# config.py
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ================== Insight SDK 登录凭证 ==================
INSIGHT_USER = "~"   # 请使用自己的用户名和密码
INSIGHT_PASSWORD = "~" 

# ================== 数据时间范围 ==================
START_DATE = "2020-01-14"
START_TEST_DATE = "2022-01-14"
END_DATE = "2026-02-20"
TRADING_DAY_START = datetime.strptime(START_DATE, '%Y-%m-%d')
TRADING_DAY_END = datetime.strptime(END_DATE, '%Y-%m-%d')

# ================== 指数成分股 ==================
INDEX_HTSC_CODE = '000300'
INDEX_NAME = '沪深300'

# ================== 因子回测区间 ==================
# 用于 IC 测试和分层测试
IC_START_DATES = ['2022-01-08', '2023-01-06', '2024-01-05']
IC_REF_DATES   = ['2023-01-01', '2024-01-01', '2025-11-30']
IC_END_DATES   = ['2023-12-30', '2024-12-29', '2025-12-30']
# ================== 数据存储路径 ==================
RAW_DATA_DIR        = os.path.join(BASE_DIR, 'raw_data')
FACTOR_DATA_DIR     = os.path.join(BASE_DIR, 'data')
FACTOR_LONG_DIR     = os.path.join(BASE_DIR, 'data1')
INDEX_COMPONENT_DIR = os.path.join(BASE_DIR, 'index')
STOCK_DAILY_DIR     = os.path.join(BASE_DIR, 'data_bfq')
IC_RESULT_DIR       = os.path.join(BASE_DIR, 'ic_results')
LAYER_RESULT_DIR    = os.path.join(BASE_DIR, 'fenceng_results')
RESULT_DIR          = os.path.join(BASE_DIR, 'factor_sum_results')
STOCK_INDUSTRY_FILE = os.path.join(BASE_DIR, 'raw_data', 'industry.csv')

# ================== 自动创建目录 ==================
_ALL_DIRS = [
    RAW_DATA_DIR,
    FACTOR_DATA_DIR,
    FACTOR_LONG_DIR,
    INDEX_COMPONENT_DIR,
    STOCK_DAILY_DIR,
    IC_RESULT_DIR,
    LAYER_RESULT_DIR,
    RESULT_DIR
]
# ================== ETF 数据目录 ==================
ETF_DAILY_DIR = r'D:\ETF数据\行情数据-1d'  


def ensure_directories():
    """创建所有配置中需要的文件夹（若不存在）"""
    for d in _ALL_DIRS:
        os.makedirs(d, exist_ok=True)
