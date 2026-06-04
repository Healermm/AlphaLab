# data_fetcher.py
import os
import pandas as pd
from datetime import datetime
from insight_python.com.insight import common
from insight_python.com.insight.query import get_trading_days, get_index_component, get_daily_basic, get_industry
import config
from utils import ensure_dir, normalize_date

def login_insight():
    """登录 Insight SDK"""
    from insight_python.com.insight.market_service import market_service
    class InsightMarketService(market_service):
        def on_subscribe_tick(self, result): pass
        def on_subscribe_kline(self, result): pass
        def on_subscribe_trans_and_order(self, result): pass
    markets = InsightMarketService()
    result = common.login(markets, config.INSIGHT_USER, config.INSIGHT_PASSWORD, login_log=False)
    print("Insight 登录结果:", result)
    return markets
def fetch_index_components():
    """获取每个交易日的沪深300成分股列表"""
    trading_days = get_trading_days(
        trading_day=[config.TRADING_DAY_START, config.TRADING_DAY_END],
        exchange='XSHG'
    )[1]
    
    frames = []
    for day_str in trading_days:
        day = datetime.strptime(day_str[:10], '%Y-%m-%d')
        day_data = get_index_component(
            htsc_code=config.INDEX_HTSC_CODE,
            stock_code='',
            name=config.INDEX_NAME,
            trading_day=day
        )
        if day_data is not None and not day_data.empty:
            frames.append(day_data)
        else:
            print(f"警告: 日期 {day} 无成分股数据，返回值为 {day_data}")
    
    if not frames:
        return pd.DataFrame()
    
    stock_data = pd.concat(frames, axis=0, ignore_index=True)
    stock_data = stock_data[['stock_code', 'trading_day', 'in_date', 'out_date']]
    stock_data = normalize_date(stock_data, 'trading_day')
    stock_data = normalize_date(stock_data, 'in_date')
    stock_data = normalize_date(stock_data, 'out_date')
    return stock_data

def fetch_daily_basic(stock_list):
    """获取股票日频基础数据（价量）"""
    all_frames = []  # 用列表收集
    for code in stock_list:
        df = get_daily_basic(code, trading_day=[config.TRADING_DAY_START, config.TRADING_DAY_END])
        if df.empty:
            continue  # 跳过无数据的股票
        df['pctChg'] = df['close'].pct_change() * 100
        df['vwap'] = df['value'] / df['volume']
        all_frames.append(df)
    
    if not all_frames:
        return pd.DataFrame()  # 全部无数据时返回空 DataFrame
    
    all_data = pd.concat(all_frames, axis=0, ignore_index=True)
    all_data = normalize_date(all_data, 'trading_day')
    return all_data

def fetch_industry(stock_list):
    """获取股票申万一级行业分类"""
    ind_data = pd.DataFrame()
    for code in stock_list:
        ind = get_industry(code, 'sw')
        ind_data = pd.concat([ind_data, ind], axis=0)
    return ind_data

def save_raw_data():
    """主流程：获取并保存原始数据"""
    ensure_dir(config.RAW_DATA_DIR)
    markets = login_insight()
    
    # 成分股变动
    print("获取指数成分股...")
    comp_df = fetch_index_components()
    comp_df.to_csv(os.path.join(config.RAW_DATA_DIR, 'index_components.csv'), index=False)
    
    stock_list = comp_df['stock_code'].unique().tolist()
    print(f"共有 {len(stock_list)} 只股票")
    
    # 日频价量
    print("获取日频基础数据...")
    daily_df = fetch_daily_basic(stock_list)
    daily_df.to_csv(os.path.join(config.RAW_DATA_DIR, 'daily_basic.csv'), index=False)
    
    # 行业分类
    print("获取行业分类...")
    ind_df = fetch_industry(stock_list)
    ind_df.to_csv(os.path.join(config.RAW_DATA_DIR, 'industry.csv'), index=False)
    
    # 股票交易状态合并
    state_df = daily_df[['htsc_code', 'trading_day', 'trading_state']].copy()
    state_df = normalize_date(state_df, 'trading_day')
    state_df['stock_code'] = state_df['htsc_code']
    state_df.drop('htsc_code', axis=1, inplace=True)
    merged_state = pd.merge(
        comp_df[['stock_code', 'trading_day', 'in_date', 'out_date']],
        state_df,
        on=['stock_code', 'trading_day'],
        how='left'
    )
    merged_state.to_csv(os.path.join(config.RAW_DATA_DIR, 'stock_state.csv'), index=False)
    print("原始数据保存完成")

if __name__ == '__main__':
    save_raw_data()