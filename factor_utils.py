# factor_utils.py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import numpy.linalg as la
from scipy.stats import linregress

# ---------- 行业中性化 ----------
def neutral(data, ind):
    stocks = list(data.index)
    data_ = pd.DataFrame(ind.l1_name, index=['INDUSTRY'], columns=ind.htsc_code).T
    data_med = pd.get_dummies(data_, columns=['INDUSTRY'])
    X = np.array(data_med)
    y = data.values
    beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y)
    residual = y - X.dot(beta_ols)
    return residual

def IndNeutralize(vwap, ind):
    vwap_ = vwap.fillna(0)
    for i in range(len(vwap_)):
        vwap_.iloc[i] = neutral(vwap_.iloc[i], ind)
    return vwap_

# ---------- 时序函数 ----------
def ts_sum(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).sum()

def sma(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).mean()

def stddev(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).std()

def correlation(x, y, window):
    x = ensure_dataframe(x)
    y = ensure_dataframe(y)
    return x.rolling(window).corr(y)

def covariance(x, y, window):
    x = ensure_dataframe(x)
    y = ensure_dataframe(y)
    return x.rolling(window).cov(y)

def rolling_rank(na):
    na = ensure_dataframe(na)
    return rankdata(na)[-1]

def ts_rank(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    na = ensure_dataframe(na)
    return np.prod(na)

def product(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).min()

def ts_max(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).max()

def delta(df, period):
    df = ensure_dataframe(df)
    return df.diff(period)

def delay(df, period):
    df = ensure_dataframe(df)
    return df.shift(period)

def rank(df):
    df = ensure_dataframe(df)
    return df.rank(pct=True, axis=1)

def scale(df, k=1):
    df = ensure_dataframe(df)
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).apply(np.argmax) + 1

def ts_argmin(df, window):
    df = ensure_dataframe(df)
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period):
    df = ensure_dataframe(df)
    if df.isnull().values.any():
        df = df.ffill().bfill().fillna(0)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.values
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = np.dot(x.T, y)
    return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)

# ---------- 191用到的函数  --------
def ensure_dataframe(x):
    """将输入转为 DataFrame，若已是 DataFrame 则直接返回（零拷贝）。"""
    if isinstance(x, pd.DataFrame):
        return x
    elif isinstance(x, pd.Series):
        return x.to_frame()
    else:  # np.ndarray
        # 对于 2D 数组，保留形状；1D 转为单列
        if x.ndim == 1:
            return pd.DataFrame(x.reshape(-1, 1))
        else:
            return pd.DataFrame(x)

def ewm_mean(series, com=None, span=None, alpha=None, halflife=None, adjust=False):
    """指数加权平均，自动处理输入类型。"""
    df = ensure_dataframe(series)
    return df.ewm(com=com, span=span, alpha=alpha, halflife=halflife, adjust=adjust).mean()

def rolling_slope(df, window):
    """滚动线性回归斜率（对自然索引 0..window-1）。"""
    df = ensure_dataframe(df)
    def _slope(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        return linregress(x, y)[0]
    return df.rolling(window).apply(_slope, raw=True)

def rolling_apply(df, window, func, raw=False):
    """通用滚动 apply，自动转换输入。"""
    df = ensure_dataframe(df)
    return df.rolling(window).apply(func, raw=raw)

# 原有的辅助函数（从 alpha191.py 移入）
def high_day(high: pd.DataFrame, window: int) -> pd.DataFrame:
    """返回窗口内最高价距当前的天数（0表示当天）"""
    return ts_argmax(high, window) - 1

def low_day(low: pd.DataFrame, window: int) -> pd.DataFrame:
    return ts_argmin(low, window) - 1
def accumulation(data, func):
    """
    对 DataFrame 的每一列独立进行累积计算。
    
    参数
    ----
    data : pd.DataFrame
        输入数据，行代表时间，列代表不同股票。
    func : callable
        累积函数，签名：func(prev_value, curr_value) -> new_value
        其中 prev_value 和 curr_value 均为标量（float/int）。
    
    返回
    ----
    pd.DataFrame
        与 data 形状相同的累积结果。
    """
    # 确保输入为 DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    elif not isinstance(data, pd.DataFrame):
        data = ensure_dataframe(data)   # 如果 ensure_dataframe 存在，否则用 pd.DataFrame(data)
    
    result = data.copy()
    for col in data.columns:
        col_series = data[col]           # 单列 Series
        res_col = result[col]            # 结果列
        # 从第2行开始累积（第1行保持原值）
        for i in range(1, len(col_series)):
            prev = res_col.iloc[i-1]     # 标量
            curr = col_series.iloc[i]    # 标量
            res_col.iloc[i] = func(prev, curr)
    return result

def rank_ts(x, window):
    """时序百分比排名，等价于 mrank(x, true, window)"""
    x = ensure_dataframe(x)
    return ts_rank(x, window) / window