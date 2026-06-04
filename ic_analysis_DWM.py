# ic_analysis.py
import os
import re
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import config
from utils import ensure_dir, strip_stock_suffix
# ---------- 1. 获取交易日 ----------
def get_all_trading_days(start_date, end_date, data_dir=config.STOCK_DAILY_DIR):
    """通过参考股票 000001 的日线数据获取区间内所有交易日"""
    ref_stock = '000001'
    file_path = os.path.join(data_dir, f"{ref_stock}.csv")
    if not os.path.exists(file_path):
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError(f"在 {data_dir} 中没有找到任何股票日线数据文件")
        ref_stock = files[0].replace('.csv', '')
        file_path = os.path.join(data_dir, f"{ref_stock}.csv")
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        all_days = df['date'].sort_values().unique()
        result = pd.DatetimeIndex(all_days)
        if len(result) == 0:
            raise ValueError(f"在区间 [{start_date}, {end_date}] 内没有找到交易日")
        return result
    except Exception as e:
        print(f"读取参考股票 {ref_stock} 的日线文件失败: {e}")
        raise

# ---------- 2. 生成调仓日（买入日）序列 ----------
def get_rebalance_dates(start_date, end_date, freq, data_dir=config.STOCK_DAILY_DIR):
    all_days = get_all_trading_days(start_date, end_date, data_dir)
    if freq == 'D':
        return all_days
    elif freq == 'W':
        # 每周五（若周五非交易日则取本周最后一个交易日）
        s = pd.Series(all_days, index=all_days)
        weekly = s.groupby(pd.Grouper(freq='W-FRI')).max()
        return pd.DatetimeIndex(weekly)
    elif freq == 'M':
        s = pd.Series(all_days, index=all_days)
        monthly = s.groupby(pd.Grouper(freq='M')).max()
        return pd.DatetimeIndex(monthly)
    else:
        raise ValueError(f"不支持的频率: {freq}，请使用 'D', 'W', 'M'")

# ---------- 3. 收益率计算函数 ----------
def compute_forward_returns_to_dates(stock_code, buy_dates, sell_dates, data_dir):
    """
    根据给定的买入日期和对应卖出日期计算区间收益率。
    buy_dates 和 sell_dates 长度相同，一一对应。
    返回 Series，index = buy_dates，值为 sell_date / buy_date - 1。
    """
    file_path = os.path.join(data_dir, f"{stock_code}.csv")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()
    close = df['close']
    ret = pd.Series(index=buy_dates, dtype=float)
    for i, (b, s) in enumerate(zip(buy_dates, sell_dates)):
        if b not in close.index or s not in close.index:
            ret.iloc[i] = np.nan
            continue
        ret.iloc[i] = close[s] / close[b] - 1
    ret = ret.replace([np.inf, -np.inf], np.nan)  # 将无穷值替换为 NaN
    return ret

def compute_forward_returns_fixed_hold(stock_code, buy_dates, hold_days, data_dir):
    """固定持有 hold_days 个交易日的收益率（仅用于日频）"""
    file_path = os.path.join(data_dir, f"{stock_code}.csv")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()
    close = df['close']
    forward_ret = pd.Series(index=buy_dates, dtype=float)
    for i, buy_date in enumerate(buy_dates):
        if buy_date not in close.index:
            forward_ret.iloc[i] = np.nan
            continue
        pos = close.index.get_loc(buy_date)
        if pos + hold_days >= len(close):
            forward_ret.iloc[i] = np.nan
            continue
        sell_date = close.index[pos + hold_days]
        forward_ret.iloc[i] = close[sell_date] / close[buy_date] - 1
    forward_ret = forward_ret.replace([np.inf, -np.inf], np.nan)  # 将无穷值替换为 NaN
    return forward_ret

# ---------- 4. 构建宽表 ----------
def build_forward_returns_wide_fixed_hold(buy_dates, stock_list, hold_days, data_dir):
    """固定持有期（日频专用）"""
    ret_wide = pd.DataFrame(index=buy_dates, columns=stock_list, dtype=float)
    for code in stock_list:
        ret_series = compute_forward_returns_fixed_hold(code, buy_dates, hold_days, data_dir)
        ret_wide[code] = ret_series
    return ret_wide

def build_forward_returns_wide_with_dates(buy_dates, sell_dates, stock_list, data_dir):
    """使用指定的卖出日期（周/月频专用）"""
    ret_wide = pd.DataFrame(index=buy_dates, columns=stock_list, dtype=float)
    for code in stock_list:
        ret_series = compute_forward_returns_to_dates(code, buy_dates, sell_dates, data_dir)
        ret_wide[code] = ret_series
    return ret_wide
# ----------5. 其他工具函数----------
def zhuanhuan_ret(wide_df, rebalance_dates, value_name='NEXT_RET'):
    wide_df.index = rebalance_dates
    long_df = wide_df.stack().to_frame(name=value_name)
    long_df.index.names = ['date', 'codes']
    return long_df

def build_stock_valid_df(rebalance_dates, comp_dir=config.INDEX_COMPONENT_DIR, data_dir=config.STOCK_DAILY_DIR):
    """判断每个调仓日，成分股是否有当日的日线数据"""
    stock_valid = []
    for date in rebalance_dates:
        year = date.year
        comp_file = os.path.join(comp_dir, f'hs300_components_{year}.csv')
        if not os.path.exists(comp_file):
            stock_valid.append([])
            continue
        comp_df = pd.read_csv(comp_file)
        all_stocks = strip_stock_suffix(comp_df['code']).tolist()
        valid = []
        target_date = pd.Timestamp(date)   # 统一转为 Timestamp
        for code in all_stocks:
            fpath = os.path.join(data_dir, f'{code}.csv')
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            df['date'] = pd.to_datetime(df['date'])
            if target_date in df['date'].values:
                valid.append(code)
        stock_valid.append(valid)
    return pd.DataFrame({'stocks': stock_valid}, index=rebalance_dates)

def ic_fenxi(df, alpha, stock_valid_df):
    dates = df.index.get_level_values('date').unique().sort_values()
    ic_list = []
    for date in dates:
        try:
            valid_stocks = stock_valid_df.loc[date].iloc[0]
        except KeyError:
            continue
        df_date = df.xs(date, level='date')
        available = df_date.index.intersection(valid_stocks)
        if len(available) < 2:
            continue
        sub = df_date.loc[available]
        ic = sub['NEXT_RET'].corr(sub[alpha], method='spearman')
        if not np.isnan(ic):
            ic_list.append(ic)
    ic_series = pd.Series(ic_list)
    pos_rate = (ic_series >= 0).mean()
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
    return [ic_mean, ic_std, ic_ir, pos_rate]

# ---------- 6. 主分析函数 ----------
def run_ic_analysis(freq='W', hold_days=None):
    # 日频需要 hold_days，周/月频忽略 hold_days，使用相邻调仓日
    if freq == 'D' and hold_days is None:
        hold_days = 1
    ensure_dir(config.IC_RESULT_DIR)
    next_ret_list = []
    for i in range(3):
        print(f"处理区间 {i+1}/3: {config.IC_START_DATES[i]} 至 {config.IC_END_DATES[i]}, 频率={freq}")
        rebalance_dates = get_rebalance_dates(config.IC_START_DATES[i], config.IC_END_DATES[i], freq)
        if len(rebalance_dates) < 2:
            print(f"警告: 区间 {i+1} 交易日不足，跳过")
            continue
        buy_dates = rebalance_dates[:-1]
        sell_dates = rebalance_dates[1:]   # 下一个调仓日作为卖出日
        
        year = config.IC_REF_DATES[i].split('-')[0]
        comp_df = pd.read_csv(os.path.join(config.INDEX_COMPONENT_DIR, f'hs300_components_{year}.csv'))
        stocks = strip_stock_suffix(comp_df['code']).tolist()
        
        if freq == 'D':
            # 日频：固定持有1个交易日（实际就是下一个交易日）
            ret_wide = build_forward_returns_wide_fixed_hold(buy_dates, stocks, hold_days, config.STOCK_DAILY_DIR)
        else:
            # 周频或月频：使用相邻调仓日收益率（严格遵循日历周期）
            ret_wide = build_forward_returns_wide_with_dates(buy_dates, sell_dates, stocks, config.STOCK_DAILY_DIR)
        
        next_ret = zhuanhuan_ret(ret_wide, buy_dates)
        next_ret_list.append(next_ret)
    
    if not next_ret_list:
        print("没有有效区间，退出")
        return
    
    next_ret = pd.concat(next_ret_list)
    all_dates = next_ret.index.get_level_values('date').unique().sort_values()
    stock_valid_df = build_stock_valid_df(all_dates)
    
    factor_files = [f for f in os.listdir(config.FACTOR_LONG_DIR) if re.match(r'^data_alpha\d+\.csv$', f)]
    summary = []
    for file in factor_files:
        df = pd.read_csv(os.path.join(config.FACTOR_LONG_DIR, file))
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df['codes'] = strip_stock_suffix(df['codes'])
        df.set_index(['date', 'codes'], inplace=True)
        factor_name = file.replace('.csv', '')
        df = df[[df.columns[0]]]
        df.columns = [factor_name]
        combined = df.join(next_ret[['NEXT_RET']], how='inner')
        if combined.empty:
            continue
        stats = ic_fenxi(combined, factor_name, stock_valid_df)
        summary.append([factor_name] + stats)
        print(f"{factor_name}: {stats}")
    
    summary_df = pd.DataFrame(summary, columns=['factor', 'IC_mean', 'IC_std', 'IC_IR', 'pos_rate'])
    out_file = os.path.join(config.IC_RESULT_DIR, f'ic_summary_{freq}.csv')
    summary_df.to_csv(out_file, index=False)
    print(f"IC 分析完成，结果保存至 {out_file}")

# ---------- 命令行入口 ----------
if __name__ == '__main__':
    run_ic_analysis(freq='D')   # 日频
    #run_ic_analysis(freq='W')   # 周频
    #run_ic_analysis(freq='M')   # 月频