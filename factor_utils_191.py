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
    return df.rolling(window).mean()

def stddev(df, window):
    return df.rolling(window).std()

def correlation(x, y, window):
    return x.rolling(window).corr(y)

def covariance(x, y, window):
    return x.rolling(window).cov(y)

def rolling_rank(na):
    return rankdata(na)[-1]

def ts_rank(df, window):
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    return np.prod(na)

def product(df, window):
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window):
    return df.rolling(window).min()

def ts_max(df, window):
    return df.rolling(window).max()

def delta(df, period):
    return df.diff(period)

def delay(df, period):
    return df.shift(period)

def rank(df):
    return df.rank(pct=True, axis=1)

def scale(df, k=1):
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window):
    return df.rolling(window).apply(np.argmax) + 1

def ts_argmin(df, window):
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period):
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

def accumulation(series: pd.Series, func):
    """自累积计算，用于因子 143"""
    result = series.copy()
    for i in range(1, len(series)):
        result.iloc[i] = func(result.iloc[i-1], series.iloc[i])
    return result

def rank_ts(x, window):
    """时序百分比排名，等价于 mrank(x, true, window)"""
    return ts_rank(x, window) / window