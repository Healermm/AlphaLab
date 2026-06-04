# layer_analysis.py
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import ensure_dir, strip_stock_suffix
# 从新版 ic_analysis 中导入正确的函数
from ic_analysis_DWM import (
    get_rebalance_dates,
    build_forward_returns_wide_with_dates,
    build_forward_returns_wide_fixed_hold,
    zhuanhuan_ret,
    build_stock_valid_df
)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def return_fenxi(df, dates, num, alpha, stock_valid_df):
    """
    返回：
        culmu      : 每层最终累积收益率列表
        return_s   : 每层的累积收益率序列列表 (list of list)
        eff_dates  : 实际参与计算的调仓日期列表（顺序与 return_s 中的列一致）
    """
    return_s = []
    eff_dates = []          # 记录有效调仓日
    for date in dates:
        try:
            valid_stocks = stock_valid_df.loc[date].iloc[0]
        except KeyError:
            continue
        df_date = df.xs(date, level='date')
        available = df_date.index.intersection(valid_stocks)
        if len(available) == 0:
            continue
        dff = df_date.loc[available].copy()
        x = dff[alpha]
        if x.sum() == 0:
            continue
        df_i = dff.sort_values(alpha)
        layer_rets = []
        for j in range(num):
            n1 = int(len(df_i) * j / num)
            n2 = int(len(df_i) * (j + 1) / num)
            layer_rets.append(df_i.iloc[n1:n2]['NEXT_RET'].mean() + 1)
        return_s.append(layer_rets)
        eff_dates.append(date)      # 记录有效日期

    if not return_s:
        return [], [], []

    # 转置为 [组数, 期数]
    return_s = np.array(return_s).T.tolist()
    for seq in return_s:
        for j in range(1, len(seq)):
            seq[j] *= seq[j-1]
    culmu = [seq[-1] for seq in return_s]
    return culmu, return_s, eff_dates

def fencengceshi(factor_data, next_ret, num, stock_valid_df):
    """
    对多个因子进行分层测试
    返回：
        cum_seqs   : 各因子的分组累计收益序列列表 (list of list of list)
        eff_dates  : 各因子的有效日期列表 (list of list)，因不同因子有效日期可能不同
    """
    rebalance_dates = next_ret.index.get_level_values('date').unique().sort_values()
    cum_seqs = []
    all_eff_dates = []      # 新增：记录每个因子的有效日期
    for df in factor_data:
        if df.empty:
            cum_seqs.append([])
            all_eff_dates.append([])
            continue
        factor_name = df.columns[0]
        combined = df.join(next_ret[['NEXT_RET']], how='inner')
        if combined.empty:
            cum_seqs.append([])
            all_eff_dates.append([])
            continue
        _, seqs, eff_dates = return_fenxi(combined, rebalance_dates, num, factor_name, stock_valid_df)
        cum_seqs.append(seqs)
        all_eff_dates.append(eff_dates)
    return cum_seqs, all_eff_dates

# ---------- 主分析函数（支持频率选择，调用 ic_analysis 中的方法）----------
def run_layer_analysis(freq='W'):
    ensure_dir(config.LAYER_RESULT_DIR)

    # 生成未来收益率数据（完全复用 ic_analysis 中的方法）
    next_ret_list = []
    for i in range(3):
        print(f"处理区间 {i+1}/3: {config.IC_START_DATES[i]} 至 {config.IC_END_DATES[i]}, 频率={freq}")
        # 调仓日（买入日）序列
        rebalance_dates = get_rebalance_dates(config.IC_START_DATES[i], config.IC_END_DATES[i], freq)
        if len(rebalance_dates) < 2:
            print(f"警告: 区间 {i+1} 交易日不足，跳过")
            continue
        buy_dates = rebalance_dates[:-1]
        sell_dates = rebalance_dates[1:]   # 周/月频的卖出日；日频中未使用但保留

        year = config.IC_REF_DATES[i].split('-')[0]
        comp_df = pd.read_csv(os.path.join(config.INDEX_COMPONENT_DIR, f'hs300_components_{year}.csv'))
        stocks = strip_stock_suffix(comp_df['code']).tolist()

        if freq == 'D':
            # 日频：固定持有1个交易日
            ret_wide = build_forward_returns_wide_fixed_hold(buy_dates, stocks, 1, config.STOCK_DAILY_DIR)
        else:
            # 周频/月频：使用相邻调仓日收益率
            ret_wide = build_forward_returns_wide_with_dates(buy_dates, sell_dates, stocks, config.STOCK_DAILY_DIR)

        next_ret = zhuanhuan_ret(ret_wide, buy_dates)
        next_ret_list.append(next_ret)

    if not next_ret_list:
        print("没有有效区间，退出")
        return

    next_ret = pd.concat(next_ret_list)
    all_dates = next_ret.index.get_level_values('date').unique().sort_values()
    stock_valid_df = build_stock_valid_df(all_dates)

    # 加载因子数据
    factor_files = [f for f in os.listdir(config.FACTOR_LONG_DIR) if re.match(r'^data_alpha\d+\.csv$', f)]
    factor_data = []
    factor_names = []
    for file in factor_files:
        df = pd.read_csv(os.path.join(config.FACTOR_LONG_DIR, file))
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df['codes'] = strip_stock_suffix(df['codes'])
        df.set_index(['date', 'codes'], inplace=True)
        name = file.replace('.csv', '')
        df = df[[df.columns[0]]]
        df.columns = [name]
        factor_data.append(df)
        factor_names.append(name)

    num = 5   # 分5组
    layer_seqs, layer_dates_list = fencengceshi(factor_data, next_ret, num, stock_valid_df)
    bench_seqs, _ = fencengceshi(factor_data, next_ret, 1, stock_valid_df)

    # 绘图
    for idx, seqs in enumerate(layer_seqs):
        if not seqs:
            continue
        eff_dates = layer_dates_list[idx]
        if len(eff_dates) != len(seqs[0]):
            print(f"警告：因子 {factor_names[idx]} 日期长度与序列长度不一致，跳过绘图")
            continue

        bench_seq = bench_seqs[idx][0] if bench_seqs[idx] else None
        if bench_seq is None or len(bench_seq) != len(eff_dates):
            bench_seq = [1] * len(eff_dates)

        plt.figure(figsize=(15, 5))
        plt.plot(eff_dates, seqs[0], label='Lowest Group')
        plt.plot(eff_dates, seqs[num//2], label='Middle Group')
        plt.plot(eff_dates, seqs[-1], label='Highest Group')
        plt.plot(eff_dates, bench_seq, '--', label='Equal Weight')
        plt.legend()
        plt.title(f'Factor {factor_names[idx]} 分层测试 (5组) 频率={freq}')
        plt.xlabel('回测区间')
        plt.ylabel('净值')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        out_path = os.path.join(config.LAYER_RESULT_DIR, f'{factor_names[idx]}_fenceng_{freq}.png')
        plt.savefig(out_path)
        plt.close()
        print(f"已保存: {out_path}")

    print(f"分层测试图像生成完成，频率={freq}")

# ---------- 命令行入口 ----------
if __name__ == '__main__':
    # run_layer_analysis(freq='D')
    # run_layer_analysis(freq='M')
    run_layer_analysis(freq='W')  # 默认周频