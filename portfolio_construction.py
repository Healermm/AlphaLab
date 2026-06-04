'''
portfolio_construction.py  仓位构建层
负责：行业排序 ——> 目标ETF权重

修复说明：
  1. 40% 上限改为迭代裁剪，避免归一化后再次超限
  2. 全负信号时返回空仓（不强行持仓）
  3. 新增 risk_parity 权重方式（需传入 etf_vol_dict）
  4. 新增 n_min 参数，持仓不足时返回空仓或现金仓位
'''

import pandas as pd
import numpy as np
from etf_mapping import INDUSTRY_ETF_MAP


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def clip_weights_iterative(weights: pd.Series, max_w: float = 0.4, max_iter: int = 50) -> pd.Series:
    """
    迭代裁剪权重，确保每个权重 ≤ max_w 且归一化后仍满足约束。

    原因：一次 clip + 归一化后，被压制的行业权重会"扩散"到其他行业，
    可能导致其他行业再次超限，因此需要迭代直到收敛。
    """
    w = weights.copy().astype(float)
    for _ in range(max_iter):
        w = w.clip(upper=max_w)
        total = w.sum()
        if total <= 0:
            # 理论上不应走到这里，保底等权
            return pd.Series([1.0 / len(w)] * len(w), index=w.index)
        w = w / total
        if (w <= max_w + 1e-9).all():
            break
    return w


def calc_risk_parity_weights(etf_codes: list, etf_vol_dict: dict) -> pd.Series:
    """
    风险平价权重：weight_i ∝ 1 / vol_i
    etf_vol_dict: {etf_code: annualized_volatility}，由调用方传入
    缺失波动率的标的用所有标的的均值填充。
    """
    vols = pd.Series({code: etf_vol_dict.get(code, np.nan) for code in etf_codes})
    mean_vol = vols.mean()  # 先算均值（可能含 nan）
    vols = vols.fillna(mean_vol)
    # 防止 vol = 0 导致除零
    vols = vols.replace(0, mean_vol if mean_vol > 0 else 1e-6)
    inv_vol = 1.0 / vols
    return inv_vol / inv_vol.sum()


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def construct_portfolio(
        industry_score_today: pd.DataFrame,
        n_top: int = 5,
        n_min: int = 3,
        max_weight: float = 0.4,
        score_threshold: float = 0.0,
        weighting: str = 'score',
        etf_vol_dict: dict = None,
        allowed_codes: set = None, 
) -> dict:
    """
    构建行业ETF目标组合。

    Parameters
    ----------
    industry_score_today : pd.DataFrame
        单日行业得分，列：['industry', 'industry_score']
    n_top : int
        最多持有行业数，默认 5
    n_min : int
        最少持有行业数，不足则空仓，默认 3
    max_weight : float
        单行业最大权重，默认 40%
    score_threshold : float
        得分低于此值的行业不入选，默认 0.0（过滤负信号）
    weighting : str
        权重方式：'equal' | 'score' | 'risk_parity'
    etf_vol_dict : dict, optional
        {etf_code: vol}，risk_parity 模式下必须传入

    Returns
    -------
    dict : {etf_code: target_weight}，空仓时返回 {}
    """

    # ── 1. 排序 ──────────────────────────────
    df = industry_score_today.copy().sort_values('industry_score', ascending=False)

    # ── 2. 得分过滤（去除负信号行业）─────────
    df = df[df['industry_score'] > score_threshold]

    # ── 3. 取 Top-N ──────────────────────────
    df = df.head(n_top)

    # ── 4. 信号数量不足 → 空仓（保留现金）───
    if len(df) < n_min:
        return {}

    # ── 5. 行业 → ETF 代码 ───────────────────
    df = df.copy()
    df['etf_code'] = df['industry'].map(INDUSTRY_ETF_MAP)
    df = df.dropna(subset=['etf_code'])
    if allowed_codes is not None and len(allowed_codes) > 0:
        df = df[df['etf_code'].isin(allowed_codes)]

    if len(df) < n_min:
        return {}

    # ── 6. 权重计算 ───────────────────────────
    if weighting == 'equal':
        df['weight'] = 1.0 / len(df)

    elif weighting == 'score':
        scores = df['industry_score'].clip(lower=0)
        if scores.sum() <= 0:
            # 无正信号：空仓，不强行持仓
            return {}
        df['weight'] = scores / scores.sum()

    elif weighting == 'risk_parity':
        if etf_vol_dict is None:
            raise ValueError("weighting='risk_parity' 时必须传入 etf_vol_dict")
        rp_weights = calc_risk_parity_weights(df['etf_code'].tolist(), etf_vol_dict)
        df['weight'] = df['etf_code'].map(rp_weights).values

    else:
        raise ValueError(f'不支持 weighting={weighting}，可选: equal / score / risk_parity')

    # ── 7. 迭代裁剪，确保单行业 ≤ max_weight ─
    df['weight'] = clip_weights_iterative(df['weight'], max_w=max_weight).values

    # ── 8. 输出 dict ──────────────────────────
    portfolio = dict(zip(df['etf_code'], df['weight']))
    return portfolio