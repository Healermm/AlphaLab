# factor_calculator.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
import config
from factor_alphas import *  # 导入全部因子函数
from utils import ensure_dir

def pivot_data(df, value_col):
    """将长表数据透视为宽表（日期行，股票列）"""
    return df.pivot(index='trading_day', columns='htsc_code', values=value_col)

def main():
    ensure_dir(config.FACTOR_DATA_DIR)
    
    # 读取原始数据
    daily = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'daily_basic.csv'))
    daily['trading_day'] = pd.to_datetime(daily['trading_day'])
    industry = pd.read_csv(os.path.join(config.RAW_DATA_DIR, 'industry.csv'))
    
    # 透视表
    close   = pivot_data(daily, 'close')
    returns = pivot_data(daily, 'pctChg')
    Open    = pivot_data(daily, 'open')
    low     = pivot_data(daily, 'low')
    high    = pivot_data(daily, 'high')
    vwap    = pivot_data(daily, 'vwap')
    cap     = pivot_data(daily, 'floating_market_val')
    volume  = pivot_data(daily, 'volume')
    ind     = industry[['htsc_code', 'l1_name']]
    
    # 计算全部因子
    start_time = datetime.now()
    
    alpha_list = []

    # alpha1 ~ alpha101 严格按顺序
    alpha_list.append(alpha1(close, returns))                     # 1
    alpha_list.append(alpha2(Open, close, volume))                # 2
    alpha_list.append(alpha3(Open, volume))                       # 3
    alpha_list.append(alpha4(low))                                # 4
    alpha_list.append(alpha5(Open, vwap, close))                  # 5
    alpha_list.append(alpha6(Open, volume))                       # 6
    alpha_list.append(alpha7(volume, close))                      # 7
    alpha_list.append(alpha8(Open, returns))                      # 8
    alpha_list.append(alpha9(close))                              # 9
    alpha_list.append(alpha10(close))                             # 10
    alpha_list.append(alpha11(vwap, close, volume))               # 11
    alpha_list.append(alpha12(volume, close))                     # 12
    alpha_list.append(alpha13(volume, close))                     # 13
    alpha_list.append(alpha14(Open, volume, returns))             # 14
    alpha_list.append(alpha15(high, volume))                      # 15
    alpha_list.append(alpha16(high, volume))                      # 16
    alpha_list.append(alpha17(volume, close))                     # 17
    alpha_list.append(alpha18(close, Open))                       # 18
    alpha_list.append(alpha19(close, returns))                    # 19
    alpha_list.append(alpha20(Open, high, close, low))            # 20
    alpha_list.append(alpha21(volume, close))                     # 21
    alpha_list.append(alpha22(high, volume, close))               # 22
    alpha_list.append(alpha23(high, close))                       # 23
    alpha_list.append(alpha24(close))                             # 24
    alpha_list.append(alpha25(volume, returns, vwap, high, close))# 25
    alpha_list.append(alpha26(volume, high))                      # 26
    alpha_list.append(alpha27(volume, vwap))                      # 27
    alpha_list.append(alpha28(volume, high, low, close))          # 28
    alpha_list.append(alpha29(close, returns))                    # 29
    alpha_list.append(alpha30(close, volume))                     # 30
    alpha_list.append(alpha31(close, low, volume))                # 31
    alpha_list.append(alpha32(close, vwap))                       # 32
    alpha_list.append(alpha33(Open, close))                       # 33
    alpha_list.append(alpha34(close, returns))                    # 34
    alpha_list.append(alpha35(volume, close, high, low, returns)) # 35
    alpha_list.append(alpha36(Open, close, volume, returns, vwap))# 36
    alpha_list.append(alpha37(Open, close))                       # 37
    alpha_list.append(alpha38(close, Open))                       # 38
    alpha_list.append(alpha39(volume, close, returns))            # 39
    alpha_list.append(alpha40(high, volume))                      # 40
    alpha_list.append(alpha41(high, low, vwap))                   # 41
    alpha_list.append(alpha42(vwap, close))                       # 42
    alpha_list.append(alpha43(volume, close))                     # 43
    alpha_list.append(alpha44(high, volume))                      # 44
    alpha_list.append(alpha45(close, volume))                     # 45
    alpha_list.append(alpha46(close))                             # 46
    alpha_list.append(alpha47(volume, close, high, vwap))         # 47
    alpha_list.append(alpha48(close, ind))                        # 48 ⚠️ 中性化，但位置不变
    alpha_list.append(alpha49(close))                             # 49
    alpha_list.append(alpha50(volume, vwap))                      # 50
    alpha_list.append(alpha51(close))                             # 51
    alpha_list.append(alpha52(returns, volume, low))              # 52
    alpha_list.append(alpha53(close, high, low))                  # 53
    alpha_list.append(alpha54(Open, close, high, low))            # 54
    alpha_list.append(alpha55(high, low, close, volume))          # 55
    alpha_list.append(alpha56(returns, cap))                      # 56
    alpha_list.append(alpha57(close, vwap))                       # 57
    alpha_list.append(alpha58(vwap, volume, ind))                 # 58 ⚠️ 中性化
    alpha_list.append(alpha59(vwap, volume, ind))                 # 59 ⚠️ 中性化
    alpha_list.append(alpha60(close, high, low, volume))          # 60
    alpha_list.append(alpha61(volume, vwap))                      # 61
    alpha_list.append(alpha62(volume, high, low, Open, vwap))     # 62
    alpha_list.append(alpha63(volume, close, vwap, Open, ind))    # 63 ⚠️ 中性化
    alpha_list.append(alpha64(high, low, Open, volume, vwap))     # 64
    alpha_list.append(alpha65(volume, vwap, Open))                # 65
    alpha_list.append(alpha66(vwap, low, Open, high))             # 66
    alpha_list.append(alpha67(volume, vwap, high, ind))           # 67 ⚠️ 中性化
    alpha_list.append(alpha68(high, low, vwap))             # 68 (与alpha41相同)
    alpha_list.append(alpha69(volume, vwap, ind, close))          # 69 ⚠️ 中性化
    alpha_list.append(alpha70(close, ind, vwap, volume))          # 70 ⚠️ 中性化
    alpha_list.append(alpha71(volume, close, low, Open, vwap))    # 71
    alpha_list.append(alpha72(volume, high, low, vwap))           # 72
    alpha_list.append(alpha73(vwap, Open, low))                   # 73
    alpha_list.append(alpha74(volume, close, high, vwap))         # 74
    alpha_list.append(alpha75(volume, vwap, low))                 # 75
    alpha_list.append(alpha76(volume, vwap, low, ind))            # 76 ⚠️ 中性化
    alpha_list.append(alpha77(volume, high, low, vwap))           # 77
    alpha_list.append(alpha78(volume, low, vwap))                 # 78
    alpha_list.append(alpha79(volume, close, Open, ind, vwap))    # 79 ⚠️ 中性化
    alpha_list.append(alpha80(Open,high,ind,volume))              # 80 ⚠️ 中性化
    alpha_list.append(alpha81(volume, vwap))                      # 81
    alpha_list.append(alpha82(Open, volume, ind))                 # 82 ⚠️ 中性化
    alpha_list.append(alpha83(high,low,close,volume,vwap))          # 83
    alpha_list.append(alpha84(vwap, close))                       # 84
    alpha_list.append(alpha85(volume, high, close, low))          # 85
    alpha_list.append(alpha86(volume, close, Open, vwap))         # 86
    alpha_list.append(alpha87(volume,close,vwap,ind))             # 87 ⚠️ 中性化
    alpha_list.append(alpha88(volume, Open, low, high, close))    # 88
    alpha_list.append(alpha89(low, vwap, ind, volume))            # 89 ⚠️ 中性化
    alpha_list.append(alpha90(volume, close, ind, low))           # 90 ⚠️ 中性化
    alpha_list.append(alpha91(close, ind, volume, vwap))          # 91 ⚠️ 中性化
    alpha_list.append(alpha92(volume, high, low, close, Open))    # 92
    alpha_list.append(alpha93(vwap, ind, volume, close))          # 93 ⚠️ 中性化
    alpha_list.append(alpha94(volume, vwap))                      # 94
    alpha_list.append(alpha95(volume, high, low, Open))           # 95
    alpha_list.append(alpha96(volume, vwap, close))               # 96
    alpha_list.append(alpha97(volume, low, vwap, ind))            # 97 ⚠️ 中性化
    alpha_list.append(alpha98(volume, Open, vwap))                # 98
    alpha_list.append(alpha99(volume, high, low))                 # 99
    alpha_list.append(alpha100(volume, close, low, high, ind))    # 100 ⚠️ 中性化
    alpha_list.append(alpha101(close, Open, high, low))           # 101
        
    # 截断前 245 行（约一年数据）以消除计算窗口影响
    for i in range(len(alpha_list)):
        alpha_list[i] = alpha_list[i].iloc[245:]
    
    # 合并并保存
    df_all = pd.concat(alpha_list)
    df_all.to_csv(os.path.join(config.FACTOR_DATA_DIR, 'alpha.csv'))
    print(f"因子计算完成，耗时 {datetime.now() - start_time}")

if __name__ == '__main__':
    main()