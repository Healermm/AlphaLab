# factor_portfolio.py
"""
因子组合构建、预处理与合成
"""
import os
import pandas as pd
import numpy as np
import numpy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import spearmanr
import config
from utils import ensure_dir, strip_stock_suffix
import config

# ==================== 数据加载 ====================
def load_factor_data(factor_ids: List[int]) -> pd.DataFrame:
    dfs = []
    for fid in factor_ids:
        fpath = os.path.join(config.FACTOR_LONG_DIR, f'data_alpha{fid}.csv')
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"因子文件不存在: {fpath}")
        
        df = pd.read_csv(fpath)
        
        # 明确只取需要的三列，防止多余列干扰
        col_name = f'alpha{fid}'
        df = df[['date', 'codes', col_name]].copy()
        
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df['codes'] = strip_stock_suffix(df['codes'].astype(str))
        df = df.set_index(['date', 'codes'])
        dfs.append(df)

    combined = dfs[0].join(dfs[1:], how='inner').reset_index()
    return combined


def load_auxiliary_data():
    # 行业分类
    ind_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'industry.csv'))
    ind_df = ind_df[['htsc_code', 'l1_name']].rename(columns={'htsc_code': 'codes'})
    ind_df['codes'] = strip_stock_suffix(ind_df['codes'])

    # 股票日频数据（流通市值）
    daily_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'daily_basic.csv'))
    daily_df['trading_day'] = pd.to_datetime(daily_df['trading_day']).dt.normalize()
    daily_df = daily_df.rename(columns={'trading_day': 'date', 'htsc_code': 'codes'})
    daily_df['codes'] = strip_stock_suffix(daily_df['codes'])
    daily_df = daily_df[['date', 'codes', 'floating_market_val']]

    # 股票状态
    state_df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'stock_state.csv'))
    state_df['trading_day'] = pd.to_datetime(state_df['trading_day']).dt.normalize()
    state_df['in_date'] = pd.to_datetime(state_df['in_date']).dt.normalize()
    state_df['out_date'] = pd.to_datetime(state_df['out_date']).dt.normalize()

    if 'stock_code' not in state_df.columns:
        state_df = state_df.rename(columns={'codes': 'stock_code'})
    state_df['stock_code'] = strip_stock_suffix(state_df['stock_code'].astype(str))
    return ind_df, daily_df, state_df

def load_all_daily_returns(bfq_dir: str, codes_list: List[str]) -> pd.DataFrame:
    """加载所有股票的日度收益率和交易状态"""
    all_data = []
    for code in codes_list:
        fpath = os.path.join(bfq_dir, f"{code}.csv")
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df['codes'] = code
        # 涨跌幅（百分比）转小数
        if 'pctChg' in df.columns:
            df['ret'] = df['pctChg'] / 100.0
        else:
            raise KeyError(f"{fpath} 缺少 pctChg 列")
        # 交易状态：假设 1=正常交易，0或其它=停牌/退市
        if 'trading_state' in df.columns:
            df['tradable'] = df['trading_state'] == 1
        else:
            df['tradable'] = True
        all_data.append(df[['date', 'codes', 'ret', 'tradable']])
    if not all_data:
        raise ValueError("未加载到任何股票数据")
    return pd.concat(all_data, ignore_index=True)

# ==================== 数据预处理 ====================
def extreme_process_MAD(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """
    MAD 去极值（按日期分组处理）
    :param df: 包含因子列和 date 列的 DataFrame
    :param factor_cols: 需要处理的因子列名列表
    :return: 处理后的 DataFrame
    """
    df = df.copy()
    for date, group in df.groupby('date'):
        for col in factor_cols:
            x = group[col]
            median = x.median()
            mad = (x - median).abs().median()
            upper = median + 3 * 1.4826 * mad
            lower = median - 3 * 1.4826 * mad
            group[col] = x.clip(lower=lower, upper=upper)
        df.loc[group.index, factor_cols] = group[factor_cols]
    return df


def data_scale_neutral(df: pd.DataFrame, factor_cols: List[str],
                       daily_df: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
    """
    行业市值中性化（按日期分组处理）
    :param df: 因子数据，含 date, codes 和因子列
    :param factor_cols: 因子列名列表
    :param daily_df: 含 date, codes, floating_market_val
    :param ind_df: 含 codes, l1_name
    :return: 中性化后的 DataFrame
    """
    df = df.copy()
    # 合并市值和行业
    df = df.merge(daily_df, on=['date', 'codes'], how='left')
    df = df.merge(ind_df, on='codes', how='left')
    df = df.dropna(subset=['l1_name'])

    for date, group in df.groupby('date'):
        if group.empty:
            continue
        # 行业哑变量
        dummies = pd.get_dummies(group['l1_name'], prefix='ind')
        X = dummies.values
        # 可选：加入市值平方项等，这里只做行业中性
        for col in factor_cols:
            y = group[col].values
            try:
                beta = la.inv(X.T @ X) @ X.T @ y
                residual = y - X @ beta
            except np.linalg.LinAlgError:
                residual = y  # 如果奇异，跳过中性化
            df.loc[group.index, col] = residual

    # 删除辅助列
    df = df.drop(columns=['floating_market_val', 'l1_name'])
    return df


def standardize(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """
    截面标准化（按日期分组）
    """
    df = df.copy()
    for date, group in df.groupby('date'):
        for col in factor_cols:
            x = group[col]
            std = x.std()
            if std > 0:
                group[col] = (x - x.mean()) / std
            else:
                group[col] = 0
        df.loc[group.index, factor_cols] = group[factor_cols]
    return df


def data_process(df: pd.DataFrame, factor_cols: List[str],
                 daily_df: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
    """
    完整的因子预处理流程：MAD -> 行业市值中性化 -> 标准化
    """
    df = extreme_process_MAD(df, factor_cols)
    df = data_scale_neutral(df, factor_cols, daily_df, ind_df)
    df = standardize(df, factor_cols)
    return df
# ==================== 未来收益准备（支持多频率） ====================
def prepare_future_returns(factor_df: pd.DataFrame, daily_ret_df: pd.DataFrame,
                           state_df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
    """
    根据给定的频率，为每个因子截面日期计算未来收益。
    :param factor_df: 因子宽表（含 date, codes 及各因子列），date 需为交易日（与频率匹配）
    :param daily_ret_df: 日度收益率表（含 date, codes, ret）
    :param state_df: 股票状态表
    :param freq: 'D' 日频, 'W' 周频, 'M' 月频
    """
    factor_df = factor_df.copy()
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    daily_ret_df = daily_ret_df.copy()
    daily_ret_df['date'] = pd.to_datetime(daily_ret_df['date'])

    if freq == 'D':
        # 日频：未来收益 = 下一个交易日的 ret
        # 使用 shift(-1) 对齐
        daily_ret_df = daily_ret_df.sort_values(['codes', 'date'])
        daily_ret_df['future_ret'] = daily_ret_df.groupby('codes')['ret'].shift(-1)
        # 合并：因子截面日期对应的未来收益 = 当天的 future_ret
        aligned = factor_df.merge(
            daily_ret_df[['date', 'codes', 'future_ret']],
            on=['date', 'codes'],
            how='inner'
        )
        aligned = aligned.dropna(subset=['future_ret'])
        # 过滤股票池（按当天）
        all_valid = []
        for dt in aligned['date'].unique():
            pool = get_stocks(dt, state_df, daily_ret_df)
            sub = aligned[aligned['date'] == dt]
            sub = sub[sub['codes'].isin(pool)]
            all_valid.append(sub)
        result = pd.concat(all_valid, ignore_index=True)
        return result

    elif freq == 'W':
        # 周频：未来收益 = 下周五的周收益（累计）
        daily_ret_df['week'] = daily_ret_df['date'].dt.to_period('W-FRI')
        weekly_ret = daily_ret_df.groupby(['codes', 'week'])['ret'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        weekly_ret['date'] = weekly_ret['week'].dt.end_time.dt.normalize()
        # 因子截面日期假设为周五，未来收益对应下周五
        factor_df['next_week_date'] = factor_df['date'] + pd.Timedelta(days=7)
        aligned = factor_df.merge(
            weekly_ret[['codes', 'date', 'ret']],
            left_on=['codes', 'next_week_date'],
            right_on=['codes', 'date'],
            how='inner'
        ).rename(columns={'ret': 'future_ret'}).drop(columns=['date_y', 'next_week_date'])
        # 过滤股票池
        all_valid = []
        for dt in aligned['date_x'].unique():
            pool = get_stocks(dt, state_df, daily_ret_df)
            sub = aligned[aligned['date_x'] == dt]
            sub = sub[sub['codes'].isin(pool)]
            all_valid.append(sub)
        result = pd.concat(all_valid, ignore_index=True).rename(columns={'date_x': 'date'})
        return result.dropna(subset=['future_ret'])

    elif freq == 'M':
        # 月频：未来收益 = 下个月最后一个交易日的月累计收益
        # 首先将日收益聚合为月收益（以月末最后交易日为准）
        daily_ret_df['month'] = daily_ret_df['date'].dt.to_period('M')
        monthly_ret = daily_ret_df.groupby(['codes', 'month'])['ret'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        monthly_ret['date'] = monthly_ret['month'].dt.end_time.dt.normalize()
        # 因子截面日期假设为月末，未来收益对应下月末
        factor_df['next_month_date'] = factor_df['date'] + pd.offsets.MonthEnd(1)
        aligned = factor_df.merge(
            monthly_ret[['codes', 'date', 'ret']],
            left_on=['codes', 'next_month_date'],
            right_on=['codes', 'date'],
            how='inner'
        ).rename(columns={'ret': 'future_ret'}).drop(columns=['date_y', 'next_month_date'])
        # 过滤股票池
        all_valid = []
        for dt in aligned['date_x'].unique():
            pool = get_stocks(dt, state_df, daily_ret_df)
            sub = aligned[aligned['date_x'] == dt]
            sub = sub[sub['codes'].isin(pool)]
            all_valid.append(sub)
        result = pd.concat(all_valid, ignore_index=True).rename(columns={'date_x': 'date'})
        return result.dropna(subset=['future_ret'])

    else:
        raise ValueError(f"不支持的频率: {freq}")

# ==================== 因子合成 ====================
def factor_sum(df: pd.DataFrame, factor_cols: List[str],
               weights: List[float] = None) -> pd.DataFrame:
    """
    等权或加权合成综合因子
    :param df: 包含因子列的 DataFrame
    :param factor_cols: 参与合成的因子列名列表
    :param weights: 权重列表，默认等权
    :return: 增加 'alpha_sum' 列的 DataFrame
    """
    df = df.copy()
    if weights is None:
        weights = [1.0 / len(factor_cols)] * len(factor_cols)
    else:
        weights = np.array(weights) / np.sum(weights)

    df['alpha_sum'] = 0.0
    for col, w in zip(factor_cols, weights):
        df['alpha_sum'] += df[col] * w
    return df

# ==================== 股票池筛选 ====================
def get_stocks(date: pd.Timestamp, state_df: pd.DataFrame, daily_ret_df: pd.DataFrame = None) -> List[str]:
    """
    筛选可投资股票池：上市满6个月 + 未退市 + (可选)当天可交易
    :param date: 日期
    :param state_df: 股票状态表（含 stock_code, in_date, out_date, 可能还有 trading_state）
    :param daily_ret_df: 日度数据（含 date, codes, tradable），若提供则增加可交易过滤
    :return: 股票代码列表
    """
    # 上市满6个月且未退市
    mask = (state_df['in_date'] <= date - pd.DateOffset(months=6)) & \
           ((state_df['out_date'].isna()) | (state_df['out_date'] >= date))
    valid_stocks = state_df.loc[mask, 'stock_code'].unique().tolist()
    
    # 如果提供了日度交易状态，则进一步过滤当天可交易的股票
    if daily_ret_df is not None:
        tradable_today = daily_ret_df[(daily_ret_df['date'] == date) & daily_ret_df['tradable']]['codes'].unique()
        valid_stocks = list(set(valid_stocks) & set(tradable_today))
    
    return valid_stocks
# ==================== 计算权重 ====================
def IR_weights(future_data: pd.DataFrame, factor_cols: List[str], lookback: int = 12) -> np.ndarray:
    dates = sorted(future_data['date'].unique())
    if len(dates) < lookback:
        raise ValueError(f"历史数据不足：需要 {lookback} 周，实际 {len(dates)} 周")
    dates = dates[-lookback:]

    ic_matrix = []
    for dt in dates:
        sub = future_data[future_data['date'] == dt]
        ic_row = []
        for col in factor_cols:
            valid = sub[[col, 'future_ret']].dropna()
            if len(valid) > 5:
                ic, _ = spearmanr(valid[col], valid['future_ret'])
                ic_row.append(ic if not np.isnan(ic) else 0.0)
            else:
                ic_row.append(0.0)
        ic_matrix.append(ic_row)

    ic_matrix = np.array(ic_matrix)
    ic_matrix = np.nan_to_num(ic_matrix, nan=0.0)
    cov = np.cov(ic_matrix.T)
    mean_ic = ic_matrix.mean(axis=0)
    return cov @ mean_ic

def rate_weights(future_data: pd.DataFrame, factor_cols: List[str], 
                      lookback: int = 12, top_pct: float = 0.1) -> np.ndarray:
    dates = sorted(future_data['date'].unique())
    if len(dates) < lookback:
        raise ValueError(f"历史数据不足：需要 {lookback} 周，实际 {len(dates)} 周")
    dates = dates[-lookback:]

    rate_matrix = []
    for dt in dates:
        sub = future_data[future_data['date'] == dt]
        rate_row = []
        for col in factor_cols:
            valid = sub[[col, 'future_ret']].dropna()
            if len(valid) == 0:
                rate_row.append(0.0)
                continue
            n_top = max(1, int(len(valid) * top_pct))
            top_stocks = valid.nlargest(n_top, col)
            rate_row.append(top_stocks['future_ret'].mean())
        rate_matrix.append(rate_row)

    rate_matrix = np.array(rate_matrix)
    return rate_matrix.mean(axis=0)


# ==================== 可视化 ====================
def plot_correlation_heatmap(df: pd.DataFrame, factor_cols: List[str],
                             save_path: str = None):
    """绘制因子相关系数热力图"""
    corr = df[factor_cols].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, linewidths=0.05, linecolor='white',
                annot_kws={'size': 8, 'weight': 'bold'})
    plt.title('Factor Correlation Heatmap')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 主流程（支持频率选择） ====================
def main(freq: str = 'W'):
    ensure_dir(config.RESULT_DIR)

    # 1. 选择因子组合
    selected_factors = [40, 95, 69, 24, 19]  # 示例，可改为其他组合
    factor_cols = [f'alpha{fid}' for fid in selected_factors]

    # 2. 加载因子数据
    df = load_factor_data(selected_factors)
    print(f"加载因子数据完成，形状: {df.shape}")

    # 3. 加载辅助数据
    ind_df, daily_df, state_df = load_auxiliary_data()

    # 3.1 加载所有股票的日度收益率
    if not hasattr(config, 'STOCK_DAILY_DIR'):
        raise AttributeError("请在 config.py 中设置 STOCK_DAILY_DIR")
    all_codes = df['codes'].unique()
    daily_ret_df = load_all_daily_returns(config.STOCK_DAILY_DIR, all_codes)
    print(f"加载日度收益率完成，形状: {daily_ret_df.shape}")

    # 3.2 准备未来收益率数据（根据频率）
    future_data = prepare_future_returns(df, daily_ret_df, state_df, freq=freq)
    print(f"未来收益率数据准备完成（频率={freq}），形状: {future_data.shape}")

    # 3.3 计算权重（IR 加权）
    weights = IR_weights(future_data, factor_cols, lookback=12)
    # weights = rate_weights(future_data, factor_cols, lookback=12, top_pct=0.1)
    print(f"计算得到的权重: {weights}")

    # 4. 因子预处理（去极值、中性化、标准化）
    df_processed = data_process(df, factor_cols, daily_df, ind_df)

    # 5. 因子合成（可选用计算出的权重或等权）
    # df_combined = factor_sum(df_processed, factor_cols)               # 等权
    df_combined = factor_sum(df_processed, factor_cols, weights=weights.tolist())  # 加权
    print("因子合成完成")

    # 6. 保存预处理后的综合因子
    output_path = os.path.join(config.RESULT_DIR, f'alpha_combined_{freq}.csv')
    df_combined.to_csv(output_path, index=False)
    print(f"综合因子已保存至 {output_path}")

    # 7. 绘制相关热力图
    plot_correlation_heatmap(df_processed, factor_cols,
                             save_path=os.path.join(config.RESULT_DIR, f'factor_corr_heatmap_{freq}.png'))


if __name__ == '__main__':
    main(freq='W')  # 周频
    # main(freq='M')  # 月频
    # main(freq='D')  # 日频
