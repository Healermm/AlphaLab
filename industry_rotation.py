'''
行业聚合层 industry_rotation.py
负责：股票因子 ——> 行业得分
输入: date | stock | alpha_sum
输出: date | industry | industry_score
'''
import pandas as pd
import numpy as np

def build_industry_scores(factor_df: pd.DataFrame,ind_df: pd.DataFrame,method: str = 'mean'):
    """
    股票因子 -> 行业得分
    Parameters: factor_df: date, codes, alpha_sum
                ind_df: codes, l1_name
                method: mean, median, topk_mean
    Returns: industry_score_df
    """
 
    df = factor_df.merge(ind_df[['codes', 'l1_name']],on='codes',how='left')
    df = df.dropna(subset=['l1_name'])
    # 行业聚合
    if method == 'mean':
        industry_score_df = (
            df.groupby(['date', 'l1_name'])['alpha_sum']
            .mean()
            .reset_index()
            .rename(columns={
                'l1_name': 'industry',
                'alpha_sum': 'industry_score'
            })
        )
    elif method == 'median':
        industry_score_df = (
            df.groupby(['date', 'l1_name'])['alpha_sum']
            .median()
            .reset_index()
            .rename(columns={
                'l1_name': 'industry',
                'alpha_sum': 'industry_score'
            })
        )
    elif method == 'topk_mean':
        def calc_topk(subdf, k=10):
            topk = subdf.nlargest(k, 'alpha_sum')
            return topk['alpha_sum'].mean()
        industry_score_df = (
            df.groupby(['date', 'l1_name'])
            .apply(calc_topk,include_groups=False)
            .reset_index()
            .rename(columns={
                0: 'industry_score',
                'l1_name': 'industry'
            })
        )
    else:
        raise ValueError(f'不支持 method={method}')

    return industry_score_df