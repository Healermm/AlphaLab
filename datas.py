#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import config
import os
import pandas as pd
from datetime import datetime
from insight_python.com.insight import common
from insight_python.com.insight.query import get_kline, get_index_component, get_trading_days, get_daily_basic, get_industry
from insight_python.com.insight.market_service import market_service



class insightmarketservice(market_service):
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.logged_in = False
    
    def login(self):
        """登录insight平台"""
        # 直接调用 common.login，忽略返回值（或仅打印）
        common.login(self, self.username, self.password, login_log=False)
        # 假设登录成功（因为回调显示 OnLoginSuccess）
        self.logged_in = True
        print("登录成功")
        return True

    def logout(self):
        """登出并释放资源"""
        common.fini()
        print("已登出")

    def get_hs300_components(self, start_date):
        """
        获取沪深300指数在指定日期范围内的成分股列表
        :param start_date: datetime对象，起始日期
        :param end_date: datetime对象，结束日期
        :return: list of stock codes (带交易所后缀，如 '600000.SH')
        """
        # 指数代码：沪深300为 000300（华泰内部可能为 000300.SH，但接口通常用数字）
        result = get_index_component(htsc_code='000300', trading_day=start_date)
        print(f"获取到的成分股数据：{result}")
        if result is None:
            print(f"未获取到成分股数据")
            return []
        # 返回的字段中通常包含 'stock_code' 列
        if 'stock_code' in result.columns:
            codes = result['stock_code'].tolist()
            print(f"获取到 {len(codes)} 只成分股")
            return codes
        else:
            print("返回数据中无 stock_code 字段")
            return []

    def download_index_daily(self, index_code, start_date, end_date, save_path):
        """
        下载指数日线行情，优先尝试 get_daily_basic，失败则用 get_kline
        :param index_code: 指数代码，如 '000300.SH'
        :param start_date: datetime
        :param end_date: datetime
        :param save_path: 保存路径（含文件名）
        """
        # 尝试用 get_daily_basic
        result = get_daily_basic(htsc_code=index_code, trading_day=[start_date, end_date])
        df = None
        if isinstance(result, tuple) and len(result) >= 2:
            error_code, data = result[0], result[1]
            if error_code == 0 and data is not None:
                df = data
        elif isinstance(result, pd.DataFrame):
            df = result

        # 如果 get_daily_basic 无数据，回退到 get_kline
        if df is None or df.empty:
            print(f"指数 {index_code} 用 get_daily_basic 无数据，尝试 get_kline...")
            result = get_kline(htsc_code=[index_code], time=[start_date, end_date],
                            frequency='daily', fq='none')
            if isinstance(result, tuple) and len(result) >= 2:
                error_code, data = result[0], result[1]
                if error_code == 0 and data is not None:
                    df = data
            elif isinstance(result, pd.DataFrame):
                df = result

        if df is None or df.empty:
            print(f"指数 {index_code} 无数据")
            return

        # 确保数据按日期排序
        if 'trading_day' in df.columns:
            df = df.sort_values('trading_day')
            df.rename(columns={'trading_day': 'date'}, inplace=True)
        elif 'time' in df.columns:
            df = df.sort_values('time')
            df.rename(columns={'time': 'date'}, inplace=True)

        # 重命名列
        rename_dict = {}
        if 'open' in df.columns:
            rename_dict['open'] = 'open'
        if 'close' in df.columns:
            rename_dict['close'] = 'close'
        if 'high' in df.columns:
            rename_dict['high'] = 'high'
        if 'low' in df.columns:
            rename_dict['low'] = 'low'
        if 'volume' in df.columns:
            rename_dict['volume'] = 'volume'
        df.rename(columns=rename_dict, inplace=True)

        # 选择需要的列
        keep_cols = ['date', 'open', 'close', 'high', 'low', 'volume']
        available_cols = [col for col in keep_cols if col in df.columns]
        df = df[available_cols].copy()

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"指数数据已保存至 {save_path}，共 {len(df)} 条记录")

    def download_stock_daily(self, stock_code, start_date, end_date, save_dir):
        """
        使用 get_daily_basic 下载个股日线行情（含成交额、换手率等）
        """
        result = get_daily_basic(htsc_code=stock_code, trading_day=[start_date, end_date])
        
        # 处理返回值（可能是元组或DataFrame）
        df = None
        if isinstance(result, tuple) and len(result) >= 2:
            error_code, data = result[0], result[1]
            if error_code == 0 and data is not None:
                df = data
            else:
                print(f"  {stock_code} 查询失败，错误码: {error_code}")
                return
        elif isinstance(result, pd.DataFrame):
            df = result
        else:
            print(f"  {stock_code} 返回未知类型: {type(result)}")
            return

        if df is None or df.empty:
            print(f"  {stock_code} 无数据")
            return

        # 确保数据按日期排序
        df = df.sort_values('trading_day')
        
        # 重命名列以匹配后续处理
        df.rename(columns={
            'trading_day': 'date',
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'volume': 'volume',
            'value': 'amount',                # 成交额
            'turnover_rate': 'turnover',       # 换手率（%）
            # 涨跌幅可以用 day_change 字段，但可能不是百分比，建议用 close 计算
        }, inplace=True)

        # 计算涨跌幅（使用收盘价）
        df['pctChg'] = df['close'].pct_change()*100  # 百分比形式
        # ---------- 新增：计算 vwap（均价） ----------
        # 根据你的数据样本，volume 单位为股，amount 单位为元，因此均价 = amount / volume
        df['vwap'] = df['amount'] / df['volume']

        # 原始 turnover 如 0.5907 表示 0.5907%，除以100得到 0.005907
        df['turnover'] = df['turnover'] 
        # 选择需要的列（中文列名用于后续因子计算）
        # 如果你的后续处理期望中文列名，可以保留中文；如果期望英文，则用英文
        # 这里为了与原有代码兼容，我们保存为英文列名，并在 get_stocks_data 中映射
        keep_cols = ['date','trading_state', 'open', 'close', 'high', 'low', 'volume', 'amount',
                 'pctChg', 'turnover', 'vwap']
        available_cols = [col for col in keep_cols if col in df.columns]
        df = df[available_cols].copy()

        # 提取纯数字代码作为文件名
        code_num = stock_code.split('.')[0]
        file_path = os.path.join(save_dir, f"{code_num}.csv")
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"  {stock_code} 已保存，共 {len(df)} 条记录")

    def get_hs300_stocks(time):
        """
        从本地文件读取沪深300成分股列表
        :param time: 年份或日期字符串，用于定位文件（例如 '2019-01-01'，实际只用年份）
        :return: (list of codes, DataFrame)  codes为纯数字代码列表，DataFrame为原始成分股信息（此处可留空）
        """
        # 从 time 中提取年份（假设传入的是 '2019-01-01' 格式）
        year = time.split('-')[0]
        file_path = f'index/hs300_components_{year}.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"成分股文件 {file_path} 不存在，请先运行数据下载脚本。")
        
        df = pd.read_csv(file_path)
        # 假设文件有一列 'code'，内容为带交易所后缀的代码，如 '600000.SH'
        codes = df['code'].tolist()
        # 转换为纯数字代码（去掉后缀），以匹配 data_bfq 中的文件名
        numeric_codes = [c.split('.')[0] for c in codes]
        return numeric_codes, df

    def get_first_trading_day(self, year):
        """获取指定年份的第一个交易日"""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        print(f"调用 get_trading_days: start={start_date}, end={end_date}")
        # 调用接口，返回 (状态信息, 交易日数组)
        result = get_trading_days(exchange='XSHG', trading_day=[start_date, end_date])
        print(f"get_trading_days 返回: {result}")
        # 解析返回值
        if isinstance(result, tuple) and len(result) >= 2:
            status, days_array = result[0], result[1]
            # 通常 status 为 '0' 表示成功，可根据需要检查
            if days_array is not None and len(days_array) > 0:
                # 数组中的元素可能是 numpy.datetime64 或字符串，转换为 datetime 对象
                first_day = pd.to_datetime(days_array[0]).to_pydatetime()
                return first_day
            else:
                print("未获取到交易日历（数组为空）")
                return None
        else:
            print("返回值格式异常:", result)
            return None
    def run(self, start_year, end_year):
        """
        主流程：登录 -> 获取所有年份成分股 -> 下载个股日线 -> 下载指数日线 -> 登出
        :param start_year: 起始年份，如 2020
        :param end_year: 结束年份，如 2026
        """
        if not self.login():
            return
        index_dir = 'index'
         # 创建 index 目录
        os.makedirs(index_dir, exist_ok=True)  
        # 计算整体数据时间范围（覆盖 start_year-1 到 end_year+1 的完整年份）
        data_start = datetime(start_year - 1, 1, 1)
        data_end   = datetime(end_year + 1, 12, 31)
        print(f"整体数据时间范围：{data_start.strftime('%Y-%m-%d')} 至 {data_end.strftime('%Y-%m-%d')}")

        # 收集所有年份的成分股（去重）
        all_components = set()
        for year in range(start_year, end_year + 1):
            query_date = self.get_first_trading_day(year)
            if query_date is None:
                print(f"警告：{year}年无交易日信息，跳过该年成分股获取")
                continue
            print(f"获取 {year} 年成分股（查询日期：{query_date.strftime('%Y-%m-%d')}）")
            components = self.get_hs300_components(query_date)
            if components:
                all_components.update(components)
                # 可选：保存该年的成分股列表
                comp_df = pd.DataFrame({'code': components})
                comp_file = os.path.join('index', f'hs300_components_{year}.csv')
                comp_df.to_csv(comp_file, index=False, encoding='utf-8-sig')
                print(f"{year}年成分股列表已保存至 {comp_file}")
            else:
                print(f"警告：{year}年未获取到成分股")

        if not all_components:
            print("未获取到任何成分股，退出")
            self.logout()
            return

        components_list = list(all_components)
        print(f"合并去重后共 {len(components_list)} 只股票")

        # 创建个股保存目录
        stock_dir = 'data_bfq'
        os.makedirs(stock_dir, exist_ok=True)
        stock_ind = pd.DataFrame()
        # 下载每只股票的日线数据（整体时间范围）
        for i, code in enumerate(components_list):
            print(f"处理 {i+1}/{len(components_list)}: {code}")
            self.download_stock_daily(code, data_start, data_end, stock_dir)
            htsc_code = components_list[i]
            ind_value = get_industry(htsc_code, 'sw')
            stock_ind = pd.concat([stock_ind,ind_value],axis=0)
        
        stock_ind.to_csv(os.path.join(index_dir, f'stock_industry_{start_year}_{end_year}.csv'), index=False, encoding='utf-8-sig')   

        # 下载指数数据（整体时间范围）
        index_dir = 'index'
        os.makedirs(index_dir, exist_ok=True)
        index_save_path = os.path.join(index_dir, '000300.csv')
        self.download_index_daily('000300.SH', data_start, data_end, index_save_path)

        # 保存合并后的总成分股列表
        all_comp_df = pd.DataFrame({'code': components_list})
        all_comp_file = os.path.join('index', f'hs300_components_{start_year}_{end_year}.csv')
        all_comp_df.to_csv(all_comp_file, index=False, encoding='utf-8-sig')
        print(f"合并成分股列表已保存至 {all_comp_file}")
        

        self.logout()
        print("全部任务完成！")

def get_hs300_stocks(start_year,end_year):
        
        file_path = f'index/hs300_components_{start_year}_{end_year}.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"成分股文件 {file_path} 不存在，请先运行数据下载脚本。")
        df = pd.read_csv(file_path)
        codes = df['code'].tolist()
        numeric_codes = [c.split('.')[0] for c in codes]
        return numeric_codes, df

def get_all_date_data(start_time, end_time, list_assets):
    """
    从 data_bfq 读取指定股票列表的日线数据，拼接并返回长表。
    适用于华泰 insight 下载的数据（列名为英文小写）。
    """
    data_path = 'data_bfq'
    list_all = []
    for c in list_assets:
        df = pd.read_csv(f'{data_path}/{c}.csv')
        # 统一列名为小写，避免大小写问题
        df.columns = [col.lower().strip() for col in df.columns]
        df['asset'] = c
        # 筛选时间范围（假设日期列名为 'date'）
        df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
        list_all.append(df)

    print(f"读取了 {len(list_all)} 只股票的数据")

    if not list_all:
        return pd.DataFrame()

    df_all = pd.concat(list_all, ignore_index=True)

    # 选择需要的列（原函数只返回这些，你可按需添加 amount, turnover 等）
    keep_cols = ['asset', 'date', 'open', 'close', 'high', 'low', 'volume', 'vwap', 'pctChg']
    # 如果希望保留 amount 和 turnover，可以取消下面注释
    # extra_cols = ['amount', 'turnover']
    # for col in extra_cols:
    #     if col in df_all.columns and col not in keep_cols:
    #         keep_cols.append(col)

    final_cols = [col for col in keep_cols if col in df_all.columns]
    df_all = df_all[final_cols].copy()
    return df_all

if __name__ == '__main__':
    USERNAME = config.INSIGHT_USER
    PASSWORD = config.INSIGHT_PASSWORD

    # 设置要下载的年份范围
    START_YEAR = 2021
    END_YEAR = 2026

    downloader = insightmarketservice(USERNAME, PASSWORD)
    downloader.run(START_YEAR, END_YEAR)