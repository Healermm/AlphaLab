# data_transformer.py
import os
import pandas as pd
from datetime import datetime
from insight_python.com.insight.query import get_trading_days, get_index_component
import config
from utils import ensure_dir
from insight_python.com.insight import common
def login_insight():
    """登录 Insight，返回是否成功"""
    from insight_python.com.insight.market_service import market_service
    class DummyService(market_service):
        def on_subscribe_tick(self, result): pass
        def on_subscribe_kline(self, result): pass
        def on_subscribe_trans_and_order(self, result): pass
    markets = DummyService()
    result = common.login(markets, config.INSIGHT_USER, config.INSIGHT_PASSWORD, login_log=False)
    print("Insight 登录结果:", result)
    return result is not None and 'success' in str(result).lower()


def zhuanhuan(alpha_df, dates, stocks, columns=['alpha']):
    """将因子宽表转换为双索引长表 (date, codes)"""
    data = alpha_df.iloc[:, 1:].copy()
    data.index = alpha_df.iloc[:, 0]
    data = data.fillna(0)
    data_reindexed = data.reindex(index=dates, columns=stocks, fill_value=0)
    stacked = data_reindexed.stack()
    result = pd.DataFrame(stacked, columns=columns)
    result.index.names = ['date', 'codes']
    return result

def main():
    if not login_insight():
        print("登录失败，程序终止")
        return
    ensure_dir(config.FACTOR_LONG_DIR)
    
    date_splits = ['2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01', '2026-02-20']
    df_raw = pd.read_csv(os.path.join(config.FACTOR_DATA_DIR, 'alpha.csv'))
    
    # 按年份切分
    for i in range(5):
        start, end = date_splits[i], date_splits[i+1]
        mask = (df_raw['trading_day'] >= start) & (df_raw['trading_day'] < end)
        part = df_raw[mask].copy()
        part.to_csv(os.path.join(config.FACTOR_DATA_DIR, f'data_alpha_{i+2022}.csv'), index=False)
    
    # 转换为长表并按因子保存
    for i in range(5):
        part = pd.read_csv(os.path.join(config.FACTOR_DATA_DIR, f'data_alpha_{i+2022}.csv'))
        
        n = len(part) // 101
        dates = sorted(list(get_trading_days(
            trading_day=[datetime.strptime(date_splits[i], '%Y-%m-%d'),
                         datetime.strptime(date_splits[i+1], '%Y-%m-%d')],
            exchange='XSHG'
        )[1].str[:10]))
        stocks = list(get_index_component(
            htsc_code=config.INDEX_HTSC_CODE,
            stock_code='',
            name=config.INDEX_NAME,
            trading_day=datetime.strptime(dates[-1], '%Y-%m-%d')
        )['stock_code'])
        
        for j in range(101):
            sub = part.iloc[n*j:n*(j+1)]
            long_df = zhuanhuan(sub, dates, stocks, columns=[f'alpha{j+1}'])
            long_df.to_csv(os.path.join(config.FACTOR_LONG_DIR, f'data_alpha{j+1}_{i+2022}.csv'))
    
    # 合并不同年份的同一因子
    for j in range(1, 102):
        combined = pd.concat([
            pd.read_csv(os.path.join(config.FACTOR_LONG_DIR, f'data_alpha{j}_{i+2022}.csv'))
            for i in range(5)
        ])
        combined.to_csv(os.path.join(config.FACTOR_LONG_DIR, f'data_alpha{j}.csv'), index=False)
    
    print("数据转换完成")

if __name__ == '__main__':
    main()