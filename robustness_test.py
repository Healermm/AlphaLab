from backtest_DWM_ETF import run_backtest
from datetime import date
import pandas as pd
import numpy as np
import os
import config
import matplotlib.pyplot as plt

from industry_rotation import build_industry_scores
from utils import strip_stock_suffix

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#参数敏感性测试
def sensitivity_test(base_params):
    test_params = {
        'n_top':                   [3, 4, 5, 6, 7],
        'min_holding_days':        [1, 2, 3],
        'weight_drift_threshold':  [0.03, 0.05, 0.07, 0.10],
        'stop_loss_single':        [0.05, 0.10, 0.15, 0.20],
    }
    results = []
    for param_name, values in test_params.items():
        for val in values:
            params = base_params.copy()
            params[param_name] = val
            _, perf = run_backtest(**params)
            row = {'param': param_name, 'value': val}
            row.update(perf)
            results.append(row)
    return pd.DataFrame(results)

#滚动窗口测试，滚动2年训练 + 6个月验证，每次向前滚动6个月
def rolling_window_test(all_start='2022-01-01', all_end='2026-02-20',
                        train_years=2, test_months=6):
    from dateutil.relativedelta import relativedelta
    results = []
    
    train_start = pd.Timestamp(all_start)
    while True:
        train_end = train_start + relativedelta(years=train_years)
        test_start = train_end
        test_end = test_start + relativedelta(months=test_months)
        if test_end > pd.Timestamp(all_end):
            break
        
        print(f'训练: {train_start.date()}~{train_end.date()} | 验证: {test_start.date()}~{test_end.date()}')
        _, perf = run_backtest(start_date=str(test_start.date()), 
                               end_date=str(test_end.date()))
        perf['test_start'] = test_start.date()
        perf['test_end'] = test_end.date()
        results.append(perf)
        
        train_start += relativedelta(months=test_months)  # 向前滚动
    
    return pd.DataFrame(results)

#蒙特卡洛检验，随机打乱信号时序，跑1000次，看真实策略是否显著优于随机
def monte_carlo_test(nav_df, industry_score_df, n_sim=1000):
    real_ret = nav_df.set_index('date')['nav'].pct_change().dropna().values
    real_sharpe = real_ret.mean() / real_ret.std() * np.sqrt(252)
    
    sim_sharpes = []
    dates = industry_score_df['date'].unique()
    
    for _ in range(n_sim):
        # 打乱行业得分的日期映射
        shuffled_dates = np.random.permutation(dates)
        date_map = dict(zip(dates, shuffled_dates))
        shuffled_score_df = industry_score_df.copy()
        shuffled_score_df['date'] = shuffled_score_df['date'].map(date_map)
        
        # 用打乱后的信号跑回测
        nav_sim, _ = run_backtest(industry_score_df_override=shuffled_score_df)
        if nav_sim is not None:
            ret_sim = nav_sim.set_index('date')['nav'].pct_change().dropna().values
            s = ret_sim.mean() / ret_sim.std() * np.sqrt(252)
            sim_sharpes.append(s)
        
    
    sim_sharpes = np.array(sim_sharpes)
    p_value = (sim_sharpes >= real_sharpe).mean()
    
    print(f'真实夏普: {real_sharpe:.3f}')
    print(f'随机夏普均值: {sim_sharpes.mean():.3f}')
    print(f'p值: {p_value:.3f} (< 0.05 说明策略显著优于随机)')
    
    # 画分布图
    plt.figure(figsize=(10, 5))
    plt.hist(sim_sharpes, bins=50, alpha=0.7, label='随机夏普分布')
    plt.axvline(real_sharpe, color='red', linewidth=2, label=f'真实夏普={real_sharpe:.3f}')
    plt.title('蒙特卡洛检验：策略夏普比率 vs 随机分布')
    plt.xlabel('夏普比率')
    plt.legend()
    plt.savefig(os.path.join(config.RESULT_DIR, 'monte_carlo.png'), dpi=150)
    plt.show()
    
    return p_value

#不同市况分析
def market_regime_test():
    """按牛熊震荡市分段统计绩效"""
    periods = {
        '熊市(2022)':    ('2022-01-01', '2022-10-31'),
        '震荡(2023-24)': ('2023-01-01', '2024-09-30'),
        '牛市(2024Q4)':  ('2024-10-01', '2025-05-31'),
    }
    results = {}
    for name, (s, e) in periods.items():
        _, perf = run_backtest(start_date=s, end_date=e)
        results[name] = perf
        print(f'\n=== {name} ===')
        for k, v in perf.items():
            print(f'  {k}: {v}')
    return results

#因子衰减分析，测试不同滞后期（1日、5日、10日、20日）因子对未来收益的预测能力
def factor_decay_test(factor_df, daily_ret_df, factor_cols, lags=[1, 5, 10, 20]):
    """测试因子在不同滞后期的IC"""
    results = []
    for lag in lags:
        # 计算lag日后的收益率
        daily_ret_df = daily_ret_df.sort_values(['codes', 'date'])
        daily_ret_df[f'ret_{lag}d'] = daily_ret_df.groupby('codes')['ret'].transform(
            lambda x: x.shift(-lag).rolling(lag).sum()
        )
        merged = factor_df.merge(
            daily_ret_df[['date', 'codes', f'ret_{lag}d']],
            on=['date', 'codes'], how='inner'
        ).dropna()
        
        ic_list = []
        for dt, grp in merged.groupby('date'):
            for col in factor_cols:
                valid = grp[[col, f'ret_{lag}d']].dropna()
                if len(valid) > 5:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(valid[col], valid[f'ret_{lag}d'])
                    if not np.isnan(ic):
                        ic_list.append(ic)
        
        ic_mean = np.mean(ic_list)
        ic_ir = np.mean(ic_list) / np.std(ic_list) if np.std(ic_list) > 0 else np.nan
        results.append({'lag': lag, 'IC_mean': ic_mean, 'IC_IR': ic_ir})
        print(f'滞后{lag}日: IC均值={ic_mean:.4f}, ICIR={ic_ir:.4f}')
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    base_params = dict(freq='W', n_top=5, stop_loss_single=0.10,
                       drawdown_half=0.25, drawdown_clear=0.40,
                       weight_drift_threshold=0.05, min_holding_days=2)

    # 1. 参数敏感性
    sens_df = sensitivity_test(base_params)
    sens_df.to_csv(os.path.join(config.RESULT_DIR, 'sensitivity.csv'), index=False)
    print('参数敏感性测试完成')

    # 2. 滚动窗口
    roll_df = rolling_window_test()
    roll_df.to_csv(os.path.join(config.RESULT_DIR, 'rolling_window.csv'), index=False)
    print('滚动窗口测试完成')

    # 3. 蒙特卡洛
    nav_df, _ = run_backtest()  # 只调用一次

    factor_df = pd.read_csv(os.path.join(config.RESULT_DIR, 'alpha_combined_W.csv'))
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    factor_df['codes'] = factor_df['codes'].astype(str).str.zfill(6)

    ind_df = pd.read_csv(config.STOCK_INDUSTRY_FILE)
    ind_df = ind_df.rename(columns={'htsc_code': 'codes'})
    ind_df['codes'] = ind_df['codes'].astype(str).str.split('.').str[0].str.zfill(6)

    industry_score_df = build_industry_scores(factor_df, ind_df, method='topk_mean')
    monte_carlo_test(nav_df, industry_score_df, n_sim=50)  # 先50次

    # 4. 不同市况
    market_regime_test()