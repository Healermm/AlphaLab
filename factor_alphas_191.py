# alpha191.py
import numpy as np
import pandas as pd
from factor_utils import (
    ts_sum, sma, stddev, correlation, covariance, ts_rank, product,ts_min, ts_max, 
    delta, delay,rank, scale, ts_argmax, ts_argmin,decay_linear, IndNeutralize, 
    ensure_dataframe, rolling_apply, ewm_mean, rolling_slope, high_day, low_day, accumulation, rank_ts
)


# ---------- 191 个因子 ----------
#经典量价关系因子，它衡量的是成交量变化与日内涨跌幅之间相关性的负向指标。因子值越大，代表量价同步性越差，可视为看多信号（因为-1乘了原相关系数）。
#典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
#公式：(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
def Alpha1(close, Open, volume):
    result = -1 * correlation(rank(delta(np.log(volume + 1e-10), 1)), rank((close - Open) / Open), 6)
    return pd.DataFrame(result, index=close.index, columns=close.columns)
# 短期反转因子：今日收盘价在日内区间的相对位置较昨日下降，视为超卖，预期短期反弹。因子值越大，表示日内价格走弱程度越高，可视为看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * DELTA((CLOSE - LOW - (HIGH - CLOSE)) / (HIGH - LOW), 1)
def Alpha2(close, high, low):
    tmp = (close - low - (high - close)) / (high - low)
    result = -1 * delta(tmp, 1)
    return result

# 短期动量因子：近6日上涨动能累积，动能越强代表趋势越健康，看多。因子值越大，表示上涨动量越强。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
def Alpha3(close, high, low):
    c1 = close > delay(close, 1)
    tmp = pd.DataFrame(np.where(c1, np.minimum(low, delay(close, 1)), np.maximum(high, delay(close, 1))),
                       index=close.index, columns=close.columns)
    cond = (close == delay(close, 1))
    part = close - tmp
    part = part.where(~cond, 0)
    result = ts_sum(part, 6)
    return result

# 布林带择时：2日均线上穿/下穿8日布林带，结合成交量确认信号。返回离散信号1（看多）或-1（看空）。
# 典型用法：横截面选股，做多因子值等于1的股票，做空因子值等于-1的股票。
# 公式：( (SUM(CLOSE,8)/8+STD(CLOSE,8)) < SUM(CLOSE,2)/2 ? -1 : (SUM(CLOSE,2)/2 < SUM(CLOSE,8)/8-STD(CLOSE,8) ? 1 : (VOL/MA20>=1 ? 1 : -1) ) )
def Alpha4(close, volume):
    cond1 = (sma(close, 8) + stddev(close, 8)) < sma(close, 2)
    cond2 = sma(close, 2) < (sma(close, 8) - stddev(close, 8))
    iffalse2 = np.where((1 < (volume / sma(volume, 20))) | (volume / sma(volume, 20) == 1), 1, -1)
    iffalse1 = np.where(cond2, 1, iffalse2)
    result = np.where(cond1, -1, iffalse1)
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 量价背离反转：捕捉近期量价排名相关性出现极高值后的反转机会。因子值越大，代表量价关系越不稳定，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3)
def Alpha5(high, volume):
    result = -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    return result

# 加权价格变动反转：加权价格近4日下跌的股票排名高，短期反转看多。因子值越大，表示近期价格下跌幅度越显著。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：RANK(SIGN(DELTA((OPEN*0.85 + HIGH*0.15), 4))) * -1
def Alpha6(Open, high):
    x = (Open * 0.85 + high * 0.15)
    result = -1 * rank(np.sign(delta(x, 4)))
    return result

# 均价乖离与量能共振：均价持续高于收盘且成交量放大，捕捉超卖放量后的反弹。因子值越大，代表超卖程度越高且放量。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3))
def Alpha7(close, volume, vwap):
    result = (rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3))
    return result

# 加权典型价格短期反转：价格重心近4日下移的股票排名高，短期看多。因子值越大，表示近期价格重心下跌越明显。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：RANK(DELTA((((HIGH+LOW)/2)*0.2 + VWAP*0.8), 4) * -1)
def Alpha8(high, low, vwap):
    tmp = (high + low) / 2 * 0.2 + vwap * 0.8
    result = rank(-1 * delta(tmp, 4))
    return result

# 缩量上涨识别：价格重心上移但成交量萎缩，视为上涨不可持续，看空信号。因子值越大，代表缩量上涨程度越高。
# 典型用法：横截面选股，做多因子值低的股票，做空因子值高的股票（或直接作为负向因子使用）。
# 公式：SMA(((HIGH+LOW)/2 - (DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 7, 2)
def Alpha9(high, low, volume):
    A = ((high + low) / 2 - (delay(high, 1) + delay(low, 1)) / 2) * (high - low) / volume
    result = ewm_mean(A, alpha=2/7)
    return result

# 尾部风险/波动异象：近期出现较大负收益波动或极端高价，后续可能反弹。因子值越大，代表极端行情特征越明显，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：RANK(MAX(((RET<0?STD(RET,20):CLOSE)^2), 5))
def Alpha10(close):
    ret = close / delay(close, 1) - 1
    part1 = stddev(ret, 20)
    part2 = close
    x = pd.DataFrame(np.where(ret < 0, part1, part2),
                     index=close.index, columns=close.columns) ** 2
    result = rank(ts_max(x, 5))
    return result

# 资金流因子：成交加权收盘价相对日内区间的位置，反映短期资金流入流出。因子值越大，表示资金流入越显著，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME, 6)
def Alpha11(close, high, low, volume):
    result = ts_sum((close - low - (high - close)) / (high - low) * volume, 6)
    return result

# 开盘强势与收盘弱势的背离：开盘价偏离均价的程度，乘以收盘价相对均价的偏差取负，捕捉开盘冲高后回落的反转机会。因子值越大，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
def Alpha12(Open, close, vwap):
    result = rank(Open - ts_sum(vwap, 10) / 10) * (-1 * rank(abs(close - vwap)))
    return result

# 几何平均与均价之差：衡量价格几何重心与成交均价的偏离，正值表示价格相对均价强势。因子值越大，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(((HIGH * LOW)^0.5) - VWAP)
def Alpha13(high, low, vwap):
    result = np.sqrt(high * low) - vwap
    # np.sqrt 可能返回数组，包装
    return pd.DataFrame(result, index=high.index, columns=high.columns)

# 5日价格动量：收盘价减去5日前收盘价，直接衡量短期价格趋势。因子值越大，动量越强，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：CLOSE - DELAY(CLOSE, 5)
def Alpha14(close):
    result = close - delay(close, 5)
    return result

# 开盘涨幅：开盘价相对于昨日收盘的涨跌幅，反映隔夜信息与开盘情绪。因子值越大，开盘越强势，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：OPEN / DELAY(CLOSE, 1) - 1
def Alpha15(Open, close):
    result = Open / delay(close, 1) - 1
    return result

# 量价排名相关性极值反转：取VWAP与成交量排名相关性5日内的最大值再取负，捕捉量价同步性极高后的反转。因子值越大，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)
def Alpha16(volume, vwap):
    result = -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
    return result

# VWAP偏离高点的复合幂：VWAP相对15日高点的偏离排名，幂以5日价格变化，放大近期动量效应。因子值越大，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
def Alpha17(close, vwap):
    result = rank(vwap - ts_max(vwap, 15)) ** delta(close, 5)
    return result

# 5日价格比：收盘价除以5日前收盘价，衡量累计收益。因子值越大，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：CLOSE / DELAY(CLOSE, 5)
def Alpha18(close):
    result = close / delay(close, 5)
    return result

# 非对称5日涨跌幅：上涨时使用相对昨日的变化率，下跌时使用相对今日的变化率，放大上涨收益、缩小下跌影响。因子值越大，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
def Alpha19(close):
    result = np.where(close < delay(close, 5), (close - delay(close, 5)) / delay(close, 5),
                      np.where(close == delay(close, 5), 0, (close - delay(close, 5)) / close))
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 6日收益率：收盘价相对6日前的涨跌幅（百分数），直观反映中期动量。因子值越大，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100
def Alpha20(close):
    result = (close - delay(close, 6)) / delay(close, 6) * 100
    return result

# 短期趋势强度：6日均价对时间序列的线性回归斜率，衡量短期上升/下降趋势的强度和方向。因子值越大，表示上升趋势越稳健，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：REGBETA(MEAN(CLOSE,6), SEQUENCE(6))
def Alpha21(close):
    result = rolling_slope(close, 6)
    return result

# 慢速随机指标变种：价格偏离6日均值的幅度再经3日滞后调整，捕捉价格动量的二次变化。因子值越大，短期反弹可能性越高，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)), 12, 1)
def Alpha22(close):
    A = (close - sma(close, 6)) / sma(close, 6) - delay((close - sma(close, 6)) / sma(close, 6), 3)
    result = ewm_mean(A, alpha=1/12)
    return result

# 波动率加权情绪指标：将上涨日和下跌日的波动率分别指数平滑，计算上涨波动占比，反映市场情绪偏向。因子值越大，上涨波动权重越高，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / (SMA(...) + SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)) * 100
def Alpha23(close):
    A = np.where(close > delay(close, 1), stddev(close, 20), 0)
    B = np.where(close <= delay(close, 1), stddev(close, 20), 0)
    # 先将 A,B 转为 DataFrame，再计算 ewm_mean
    A_df = pd.DataFrame(A, index=close.index, columns=close.columns)
    B_df = pd.DataFrame(B, index=close.index, columns=close.columns)
    ewm_A = ewm_mean(A_df, alpha=1/20)
    ewm_B = ewm_mean(B_df, alpha=1/20)
    result = ewm_A / (ewm_A + ewm_B) * 100
    return result

# 5日动量的指数平滑：将5日价格变化进行指数平滑，得到平滑后的短期动量指标。因子值越大，上升动量越稳固，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(CLOSE - DELAY(CLOSE, 5), 5, 1)
def Alpha24(close):
    result = ewm_mean(close - delay(close, 5), alpha=1/5)
    return result

# 反转量比复合因子：7日价格反转与成交量偏离均值的情况相结合，并叠加长期收益排名，捕捉量能异动的反转机会。因子值越大，反转潜力越高，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(-1 * RANK((DELTA(CLOSE,7) * (1 - RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9)))))) * (1 + RANK(SUM(RET,250)))
def Alpha25(close, volume):
    adv20 = sma(volume, 20)
    result = -1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9)))) * (1 + rank(ts_sum(close / delay(close, 1) - 1, 250)))
    return result

# 短期均值回复与长期相关性叠加：7日均线偏离与VWAP和长期延迟收盘价的相关性结合，综合捕捉短期超买超卖与长期关联的异常。因子值越大，超卖或背离程度越高，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(((SUM(CLOSE,7)/7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE,5), 230)))
def Alpha26(close, vwap):
    result = (ts_sum(close, 7) / 7 - close) + correlation(vwap, delay(close, 5), 230)
    return result

# 复合涨跌幅加权：3日和6日涨跌幅的加权移动平均，综合短中期动量。因子值越大，动量越强，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100 + (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100, 12)
def Alpha27(close):
    A = (close - delay(close, 3)) / delay(close, 3) * 100 + (close - delay(close, 6)) / delay(close, 6) * 100
    # 原代码中 A.fillna(0) 是为了避免 decay_linear 遇到 NaN 报错，这里保留，
    # 但注意 fillna(0) 会引入非原始值。也可以改用其他填充，但为保持原意，保留。
    result = decay_linear(A.fillna(0), 12)
    # decay_linear 可能返回数组，包装
    return pd.DataFrame(result, index=close.index, columns=close.columns).fillna(0)  # 此处保留原逻辑

# 改良慢速KD：对9日价格区间相对位置进行双重平滑，构造类似随机指标的信号。因子值越大，价格在低位回升的可能性越高，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1) - 2*SMA(SMA(...,3,1),3,1)
def Alpha28(close, high, low):
    A = (close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100
    B = ewm_mean(A.fillna(0), alpha=1/3)
    result = 3 * B - 2 * ewm_mean(B, alpha=1/3)
    return result

# 量价齐驱：6日收益率乘以当日成交量，放大有量能配合的价格变动。因子值越大，放量上涨动能越强，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME
def Alpha29(close, volume):
    result = (close - delay(close, 6)) / delay(close, 6) * volume
    return result

# Fama-French三因子残差波动：将个股收益对市场、规模、价值三因子回归的残差平方进行加权移动平均，衡量无法被因子解释的特异波动。因子值越大，特质波动越高，可能代表投机性更强或套利限制更大，信号方向取决于市场环境，通常高特质波动预期低收益。
# 典型用法：可考虑做多因子值低的股票，做空因子值高的股票（低波异象），具体需验证。
# 公式：WMA((RESIDUAL(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)
def Alpha30(close, index_close, MKT, SMB, HML):
    # 需计算 MKT, SMB, HML三因子的滚动回归残差平方的 WMA，此处略写,占位
    return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

# 12日价格偏离均值百分比：衡量当前收盘价相对12日简单均线的偏离度，正值表示价格高于均线（短期动量偏强）。
# 因子值越大 → 看多信号（顺势）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE - MEAN(CLOSE,12)) / MEAN(CLOSE,12) * 100
def Alpha31(close):
    return (close - sma(close, 12)) / sma(close, 12) * 100

# 量价背离累积：高价与成交量排名相关性3日求和后取负，捕捉量价同步性减弱后的反转机会。
# 因子值越大 → 看多信号（量价背离后预期反转）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)
def Alpha32(high, volume):
    return -1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)

# 长期收益反转与量能：结合长期收益差、最低价变化与成交量时序排名，捕捉长期偏弱但近期有支撑的反转。
# 因子值越大 → 看多信号（反转潜力）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((-1 * TSMIN(LOW,5)) + DELAY(TSMIN(LOW,5),5)) * RANK(((SUM(RET,240) - SUM(RET,20)) / 220)) * TSRANK(VOLUME,5)
def Alpha33(close, low, volume):
    ret = close / delay(close, 1) - 1
    return (-1 * ts_min(low, 5) + delay(ts_min(low, 5), 5)) * rank((ts_sum(ret, 240) - ts_sum(ret, 20)) / 220) * ts_rank(volume, 5)

# 12日均线与价格比值：均线除以当前价，值高代表价格低于均线（超卖特征），预期均值回复。
# 因子值越大 → 看多信号（超卖反弹）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MEAN(CLOSE,12) / CLOSE
def Alpha34(close):
    return sma(close, 12) / close

# 开盘变化与量价相关性背离：取开盘变化衰减排名与量价相关性衰减排名的较小值后取负，捕捉两者同时偏弱后的反转。
# 因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN*0.65)+(OPEN*0.35)),17),7))) * -1)
def Alpha35(Open, volume):
    result = -1 * np.minimum(rank(decay_linear(delta(Open, 1), 15)),
                             rank(decay_linear(correlation(volume, Open, 17), 7)))
    # np.minimum 返回数组，包装
    return pd.DataFrame(result, index=Open.index, columns=Open.columns)

# 量价相关性累积：VWAP与成交量排名相关系数2日求和后排名，值大表示近期量价配合度高，趋势健康。
# 因子值越大 → 看多信号（顺势）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))
def Alpha36(volume, vwap):
    return rank(ts_sum(correlation(rank(volume), rank(vwap), 6), 2))

# 开盘收益动量反转：5日开盘与收益乘积的10日变化取负排名，捕捉资金流入动能衰减后的反转。
# 因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * RANK(((SUM(OPEN,5) * SUM(RET,5)) - DELAY((SUM(OPEN,5) * SUM(RET,5)), 10)))
def Alpha37(Open, close):
    ret = close / delay(close, 1) - 1
    return -1 * rank(ts_sum(Open, 5) * ts_sum(ret, 5) - delay(ts_sum(Open, 5) * ts_sum(ret, 5), 10))

# 高位动能衰减：当20日均高低于当日最高价（处于高位）时，返回两日最高价下跌的幅度（-DELTA(HIGH,2)），否则为0。
# 因子值越大 → 看空信号（高位回落风险）。
# 典型用法：横截面选股，做多因子值低的股票，做空因子值高的股票。
# 公式：(((SUM(HIGH,20)/20) < HIGH) ? (-1 * DELTA(HIGH,2)) : 0)
def Alpha38(high):
    result = np.where(sma(high, 20) < high, -1 * delta(high, 2), 0)
    return pd.DataFrame(result, index=high.index, columns=high.columns)

# 价格变化与长期量价关系差值取负：短期价格动量衰减排名与长期量价相关性衰减排名之差取负，捕捉短期相对走弱后的反转。
# 因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(DECAYLINEAR(DELTA((CLOSE),2),8)) - RANK(DECAYLINEAR(CORR(((VWAP*0.3)+(OPEN*0.7)), SUM(MEAN(VOLUME,180),37),14),12))) * -1
def Alpha39(Open, close, volume, vwap):
    return (rank(decay_linear(delta(close, 2), 8)) - rank(decay_linear(correlation(vwap * 0.3 + Open * 0.7, ts_sum(sma(volume, 180), 37), 14), 12))) * -1

# 资金流量比：26日上涨日成交量之和与下跌日成交量之和的比率百分数，值大表示资金持续流入。
# 因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26) / SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26) * 100
def Alpha40(close, volume):
    up_vol = ts_sum(np.where(close > delay(close, 1), volume, 0), 26)
    dn_vol = ts_sum(np.where(close <= delay(close, 1), volume, 0), 26)
    return up_vol / dn_vol * 100
# VWAP加速反转：VWAP近3日变化的最大值排名取负，捕捉VWAP快速上升后的均值回复。因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(MAX(DELTA((VWAP),3),5)) * -1)
def Alpha41(vwap):
    return -1 * rank(ts_max(delta(vwap, 3), 5))

# 波动率与量价背离复合：高波动排名取负，结合高成交量与高价的正相关，整体捕捉高波动但量价关系不稳定的情况。因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10)
def Alpha42(high, volume):
    return -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)

# 净成交量（6日）：涨日量计正，跌日量计负，6日求和，反映短期资金净流入。因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
def Alpha43(close, volume):
    return ts_sum(np.where(close > delay(close, 1), volume,
                           np.where(close < delay(close, 1), -volume, 0)), 6)

# 低价与成交量均值相关性及VWAP变化时序排名组合：捕捉低价位伴随量能变化以及均价趋势的共振。因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：TSRANK(DECAYLINEAR(CORR((LOW), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15)
def Alpha44(low, volume, vwap):
    return rank_ts(decay_linear(correlation(low, sma(volume, 10), 7), 6), 4) + \
           rank_ts(decay_linear(delta(vwap, 3), 10), 15)

# 加权价格变化与VWAP长期相关性乘积：衡量价格动量与长期量价关系强度的联合效应。因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15))
def Alpha45(Open, close, volume, vwap):
    return rank(delta(close * 0.6 + Open * 0.4, 1)) * rank(correlation(vwap, sma(volume, 150), 15))

# 多周期均线与现价比值：四个周期均价的平均值除以现价，比值高表示价格处于多周期低位，存在均值回升潜力。因子值越大 → 看多信号（超卖）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
def Alpha46(close):
    return (sma(close, 3) + sma(close, 6) + sma(close, 12) + sma(close, 24)) / (4 * close)

# 改良威廉指标（9日平滑）：收盘价距离6日高点的距离占整个区间的百分比，经指数平滑，反映超买超卖状态。因子值越大 → 超卖程度越高，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 9, 1)
def Alpha47(close, high, low):
    A = (ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100
    return ewm_mean(A, alpha=1/9)

# 趋势强度信号：近3日收盘涨跌符号的和的排名乘以5日与20日成交量比率，捕捉持续单向运动强度。因子值越大 → 看多信号（上涨趋势强）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1))) + SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2)))) + SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3)))))) * SUM(VOLUME,5)) / SUM(VOLUME,20)
def Alpha48(close, volume):
    sign_sum = np.sign(close - delay(close, 1)) + np.sign(delay(close, 1) - delay(close, 2)) + np.sign(delay(close, 2) - delay(close, 3))
    return -1 * rank(sign_sum) * ts_sum(volume, 5) / ts_sum(volume, 20)

# 高低点均线回归波动（上升比例）：高点+低点较昨日下降时，取价格跳动的较大值，12日累积中上升的比例。因子值越大 → 下跌波动占比高，可能超卖反弹。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(...上升...)+SUM(...下跌...))
def Alpha49(high, low):
    diff = np.maximum(np.abs(high - delay(high, 1)), np.abs(low - delay(low, 1)))
    cond1 = (high + low) >= (delay(high, 1) + delay(low, 1))
    sum1 = ts_sum(np.where(cond1, 0, diff), 12)
    sum2 = ts_sum(np.where(~cond1, 0, diff), 12)
    return sum1 / (sum1 + sum2)

# 高低点均线回归波动（下跌减上升差）：下跌波动占比与上升波动占比之差。因子值越大 → 下跌波动主导，越可能超卖，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
def Alpha50(high, low):
    diff = np.maximum(np.abs(high - delay(high, 1)), np.abs(low - delay(low, 1)))
    cond1 = (high + low) <= (delay(high, 1) + delay(low, 1))
    sum1 = ts_sum(np.where(cond1, 0, diff), 12)
    sum2 = ts_sum(np.where(~cond1, 0, diff), 12)
    return sum1 / (sum1 + sum2) - sum2 / (sum1 + sum2)

# 下跌波动占比：与Alpha51仅保留下跌波动占总变动的比例。因子值越大 → 下跌波动占比高，超卖，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(下跌)+SUM(上升))
def Alpha51(high, low):
    diff = np.maximum(np.abs(high - delay(high, 1)), np.abs(low - delay(low, 1)))
    cond = (high + low) <= (delay(high, 1) + delay(low, 1))
    sum1 = ts_sum(np.where(cond, 0, diff), 12)
    sum2 = ts_sum(np.where(~cond, 0, diff), 12)
    return sum1 / (sum1 + sum2)

# 26日上涨/下跌潜能比：基于典型价格延迟值计算的上涨潜在幅度与下跌潜在幅度之比，是趋势强度指标。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY(...,1)-L),26)*100
def Alpha52(close, high, low):
    ref = delay((high + low + close) / 3, 1)
    return ts_sum(np.maximum(0, high - ref), 26) / ts_sum(np.maximum(0, ref - low), 26) * 100

# 上涨天数占比（12日）：近12个交易日中收盘上涨的天数比例，直接反映短期动量强弱。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
def Alpha53(close):
    return ts_sum((close > delay(close, 1)).astype(int), 12) / 12 * 100

# 波动率与开盘收盘价差负向复合：高波动和正向价差与相关性的组合取负排名，偏离正常关系时看多。因子值越大 → 看多信号。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * RANK((STD(ABS(CLOSE-OPEN)) + (CLOSE-OPEN)) + CORR(CLOSE, OPEN,10))
def Alpha54(Open, close):
    return -1 * rank(stddev(np.abs(close - Open), 10) + (close - Open) + correlation(close, Open, 10))

# 价格跳跃与波动的复杂分段条件（占位）：公式极复杂，根据价格跳跃、开盘价变动、真实波幅等条件计算加权值。实际使用时需参照原版DolphinDB实现。
# 典型用法：待完整实现后确定。
# 公式：SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2...))...
def Alpha55(Open, close, high, low):
    # 返回与 close 同形状的全 NaN DataFrame
    return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

# 开盘价格偏离与高低点相关性条件：开盘价距12日低点的排名小于高低中点与成交量均值的相关性排名的五次方时取1，否则0。视为超卖信号。
# 典型用法：横截面选股，做多信号值为1的股票。
# 公式：(RANK((OPEN - TSMIN(OPEN,12))) < RANK((RANK(CORR(SUM(((HIGH+LOW)/2),19), SUM(MEAN(VOLUME,40),19),13))^5))) * -1? 原版缺失-1? 实际实现为(A<B).astype(float)
def Alpha56(Open, high, low, volume):
    A = rank(Open - ts_min(Open, 12))
    B = rank(rank(correlation(ts_sum((high + low) / 2, 19),
                              ts_sum(sma(volume, 40), 19), 13)) ** 5)
    result = (A < B).astype(float)
    # 返回 DataFrame
    return pd.DataFrame(result, index=Open.index, columns=Open.columns)

# 威廉指标平滑版（9日，3日平滑）：价格在9日高低区间的相对位置，经3日指数平滑，捕捉短期超卖。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
def Alpha57(close, high, low):
    A = (close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100
    return ewm_mean(A, alpha=1/3)

# 20日上涨天数占比：近20日上涨天数比例，中期动量指标。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
def Alpha58(close):
    return ts_sum((close > delay(close, 1)).astype(int), 20) / 20 * 100

# 20日累积价格相对昨日关键位的偏离和：上涨时取收盘与min(昨低,昨收)的差，下跌时取收盘与max(昨高,昨收)的差，反映持续性超买超卖。因子值越大 → 超买，看空? 根据原公式可能偏空，但原注释信号不明确，暂定为看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票（谨慎）。
# 公式：SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
def Alpha59(close, high, low):
    cond_eq = (close == delay(close, 1))
    cond_gt = (close > delay(close, 1))
    return ts_sum(np.where(cond_eq, 0,
                           close - np.where(cond_gt, np.minimum(low, delay(close, 1)),
                                            np.maximum(high, delay(close, 1)))), 20)

# 20日量价资金流：成交加权的收盘价相对位置，反映中期资金净流向。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)
def Alpha60(close, high, low, volume):
    return ts_sum((close - low - (high - close)) / (high - low) * volume, 20)

# VWAP变化与低价/均量相关性背离取最大负值：两种趋势衰减排名的最大值取负，捕捉其中任一趋势过强后的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)), RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80),8)),17))) * -1)
def Alpha61(low, volume, vwap):
    val = np.maximum(rank(decay_linear(delta(vwap, 1), 12)),
                     rank(decay_linear(rank(correlation(low, sma(volume, 80), 8)), 17)))
    result = -1 * val
    # np.maximum 返回数组，包装
    return pd.DataFrame(result, index=low.index, columns=low.columns)

# 高价与成交量排名负相关：高价时成交量排名低（量价背离）的特征取负值，实际信号为量价背离程度。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * CORR(HIGH, RANK(VOLUME), 5)
def Alpha62(high, volume):
    return -1 * correlation(high, rank(volume), 5)

# RSI平滑版（6日）：上涨幅度与总变动幅度的指数加权比，经典摆荡指标。因子值越大 → 看多（超买则偏空？RSI高通常超买，但此处原逻辑是作为正向因子，需结合具体回测，通常RSI>70超买看空，但序列排名可能趋向反转，暂按因子值越大看多说明）
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
def Alpha63(close):
    A = np.maximum(close - delay(close, 1), 0)
    B = np.abs(close - delay(close, 1))
    # ewm_mean 需要 DataFrame，先转换
    A_df = pd.DataFrame(A, index=close.index, columns=close.columns)
    B_df = pd.DataFrame(B, index=close.index, columns=close.columns)
    result = ewm_mean(A_df, alpha=1/6) / ewm_mean(B_df, alpha=1/6) * 100
    return result

# 量价相关性极值的衰减比较：VWAP/量相关性排名衰减与收盘/均量相关性最大值衰减，取最大后负向。捕捉相关性异常。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME),4),4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)),4),13),14))) * -1)
def Alpha64(close, volume, vwap):
    val = np.maximum(rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4)),
                     rank(decay_linear(ts_max(correlation(rank(close), rank(sma(volume, 60)), 4), 13), 14)))
    result = -1 * val
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 6日均价比现价：均线高于现价表示超卖，比值越大价格越低，均值回复潜力越大。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MEAN(CLOSE,6)/CLOSE
def Alpha65(close):
    return sma(close, 6) / close

# 6日价格偏离百分比：现价相对6日均线的偏离百分数，正值表示价格高于均线（近期偏强）。因子值越大 → 看多（顺势）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
def Alpha66(close):
    return (close - sma(close, 6)) / sma(close, 6) * 100

# RSI平滑版（24日）：长周期的相对强弱指数，捕捉中期超买超卖。因子值越大 → 看多（视为动量延续）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
def Alpha67(close):
    A = np.maximum(close - delay(close, 1), 0)
    B = np.abs(close - delay(close, 1))
    # ewm_mean 需要 DataFrame，先转换
    A_df = pd.DataFrame(A, index=close.index, columns=close.columns)
    B_df = pd.DataFrame(B, index=close.index, columns=close.columns)
    result = ewm_mean(A_df, alpha=1/24) / ewm_mean(B_df, alpha=1/24) * 100
    return result

# 量价波动（15日平滑）：基于高低点移动和成交量计算的价格变动强度，反映剧烈波动下的量能配合。因子值越大 → 波动放大且缩量时可能见顶，看空（需结合具体逻辑，此处暂按负向）。但原多因子通常作为alpha信号，根据历史回测决定方向，这里描述为看空信号。
# 典型用法：横截面选股，做多因子值低的股票，做空因子值高的股票。
# 公式：SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
def Alpha68(high, low, volume):
    A = ((high + low) / 2 - (delay(high, 1) + delay(low, 1)) / 2) * (high - low) / volume
    # ewm_mean 需要 DataFrame，先转换
    A_df = pd.DataFrame(A, index=high.index, columns=high.columns)
    result = ewm_mean(A_df, alpha=2/15)
    return result

# 趋向系统多空力量对比：DTM与DBM的20日累积之差的比例，正值表示多头力量主导。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(SUM(DTM,20)>SUM(DBM,20) ? (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20) : ... )
def Alpha69(Open, high, low):
    DTM = np.where(Open <= delay(Open, 1), 0, np.maximum(high - Open, Open - delay(Open, 1)))
    DBM = np.where(Open >= delay(Open, 1), 0, np.maximum(Open - low, Open - delay(Open, 1)))
    sum_dtm = ts_sum(DTM, 20)
    sum_dbm = ts_sum(DBM, 20)
    # 使用 numpy 条件，最后包装
    result = np.where(sum_dtm > sum_dbm, (sum_dtm - sum_dbm) / sum_dtm,
                      np.where(sum_dtm == sum_dbm, 0, (sum_dtm - sum_dbm) / sum_dbm))
    return pd.DataFrame(result, index=Open.index, columns=Open.columns)

# 成交额波动（6日）：6个交易日内成交额的标准差，衡量近期交易活跃度的变化。因子值越大 → 交易活跃度剧变，可能伴随转折，方向不定，暂按反转看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票（需验证）。
# 公式：STD(AMOUNT,6)
def Alpha70(volume, vwap):
    return stddev(volume * vwap, 6)

# 24日价格偏离百分比：收盘价相对24日均线的偏离，中期动量。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
def Alpha71(close):
    return (close - sma(close, 24)) / sma(close, 24) * 100

# 威廉指标（6日，15日平滑）：超买超卖程度，平滑后反转信号更稳健。因子值越大 → 超卖，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
def Alpha72(close, high, low):
    A = (ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100
    return ewm_mean(A, alpha=1/15)

# 双重平滑量价相关性衰减与VWAP相关性衰减之差取负：深度平滑的价量关系与VWAP关系背离时看多。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
def Alpha73(close, volume, vwap):
    return -1 * (rank_ts(decay_linear(decay_linear(correlation(close, volume, 10), 16), 4), 5) -
                 rank(decay_linear(correlation(vwap, sma(volume, 30), 4), 3)))

# 加权低价与均量相关性总和：两个相关性排名相加，综合捕捉量价与价格结构的信息。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(CORR(SUM(((LOW*0.35)+(VWAP*0.65)),20), SUM(MEAN(VOLUME,40),20),7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME),6)))
def Alpha74(low, volume, vwap):
    return rank(correlation(ts_sum(low * 0.35 + vwap * 0.65, 20), ts_sum(sma(volume, 40), 20), 7)) + \
           rank(correlation(rank(vwap), rank(volume), 6))

# 个股强势异象（相对指数）：个股上涨而指数下跌的天数占比，衡量个股独立于市场的强势。因子值越大 → 看多（个股Alpha强）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)
def Alpha75(Open, close, index_Open, index_close):
    return ts_sum((close > Open) & (index_close < index_Open), 50) / ts_sum(index_close < index_Open, 50)

# 收益波动与成交量的异变系数：收益绝对值/成交量的标准差除以均值，类似于变异系数，值大表示收益-量关系不稳定，可能预警。因子值越大 → 看空（不确定性高）。
# 典型用法：横截面选股，做多因子值低的股票，做空因子值高的股票。
# 公式：STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
def Alpha76(close, volume):
    return stddev(np.abs(close / delay(close, 1) - 1) / volume, 20) / sma(np.abs(close / delay(close, 1) - 1) / volume, 20)

# 高低中点与高价组合与均量相关性的最小值：取两个衰减排名的较小值，捕捉价格结构异常。因子值越大 → 看多（异常后反转）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MIN(RANK(DECAYLINEAR(((((HIGH+LOW)/2)+HIGH)-(VWAP+HIGH)),20)), RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2), MEAN(VOLUME,40),3),6)))
def Alpha77(high, low, volume, vwap):
    val = np.minimum(rank(decay_linear(((high + low) / 2 + high) - (vwap + high), 20)),
                     rank(decay_linear(correlation((high + low) / 2, sma(volume, 40), 3), 6)))
    result = val
    return pd.DataFrame(result, index=high.index, columns=high.columns)

# 典型价格偏离均线的归一化波动：类似布林带宽度，用绝对偏离均值标准化。因子值越大 → 价格波动异常放大，可能反转，看多（超卖）或看空需结合方向，暂定为看多（低吸机会）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((HIGH+LOW+CLOSE)/3-MA(...,12))/(0.015*MEAN(ABS(CLOSE-MEAN(...,12)),12))
def Alpha78(close, high, low):
    return ((high + low + close) / 3 - sma((high + low + close) / 3, 12)) / \
           (0.015 * sma(np.abs(close - sma((high + low + close) / 3, 12)), 12))

# RSI平滑版（12日）：经典12日RSI，动量指标。因子值越大 → 看多（强势延续）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
def Alpha79(close):
    A = np.maximum(close - delay(close, 1), 0)
    B = np.abs(close - delay(close, 1))
    A_df = pd.DataFrame(A, index=close.index, columns=close.columns)
    B_df = pd.DataFrame(B, index=close.index, columns=close.columns)
    result = ewm_mean(A_df, alpha=1/12) / ewm_mean(B_df, alpha=1/12) * 100
    return result

# 成交量5日变化率：量增可能代表关注度提升，量缩可能预示变盘。因子值越大 → 成交量放大，短期可能延续趋势或反转，暂定为看多（有量配合）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
def Alpha80(volume):
    return (volume - delay(volume, 5)) / delay(volume, 5) * 100

# 成交量指数平滑（21日）：成交量趋势，平滑后反映近期量能中枢。因子值越大 → 成交活跃度上升，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(VOLUME,21,2)
def Alpha81(volume):
    return ewm_mean(volume, alpha=1/21)

# 威廉指标（6日，20日平滑）：更长平滑的威廉指标，捕捉中期超卖状态。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
def Alpha82(close, high, low):
    A = (ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100
    return ewm_mean(A, alpha=1/20)

# 高价与成交量排名协方差负向：高排名协方差取负，捕捉量价同向运动减弱后的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5))
def Alpha83(high, volume):
    return -1 * rank(covariance(rank(high), rank(volume), 5))

# 净成交量（20日）：涨日量计正，跌日量计负，20日累积，中期资金流。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
def Alpha84(close, volume):
    return ts_sum(np.where(close > delay(close, 1), volume,
                           np.where(close < delay(close, 1), -volume, 0)), 20)

# 量比排名与价格反转排名乘积：成交量相对均量的时序排名乘以负的7日价格变化的时序排名，捕捉放量下跌后的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：TSRANK((VOLUME/MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE,7)), 8)
def Alpha85(close, volume):
    return rank_ts(volume / sma(volume, 20), 20) * rank_ts(-1 * delta(close, 7), 8)

# 价格加速度条件信号：根据短期和中期速度差返回离散值，-1、1或价格变化。加速度正向时看空，负向时看多。
# 典型用法：横截面选股，做多信号为1的股票。
# 公式：((0.25 < (((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10))) ? -1 : (((...)<0) ? 1 : -1*(CLOSE-DELAY(CLOSE,1))))
def Alpha86(close):
    A = ((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10)
    result = np.where(A > 0.25, -1, np.where(A < 0, 1, -1 * (close - delay(close, 1))))
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# VWAP变化与低价VWAP偏离的衰减排名和：捕捉VWAP趋势与低价异常偏离的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((RANK(DECAYLINEAR(DELTA(VWAP,4),7)) + TSRANK(DECAYLINEAR(((((LOW*0.9)+(LOW*0.1))-VWAP)/(OPEN-((HIGH+LOW)/2))),11),7)) * -1)
def Alpha87(Open, high, low, vwap):
    return -1 * (rank(decay_linear(delta(vwap, 4), 7)) + 
                 rank_ts(decay_linear((low * 0.9 + low * 0.1 - vwap) / (Open - (high + low) / 2), 11), 7))

# 20日收益率（百分数）：长期动量。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
def Alpha88(close):
    return (close - delay(close, 20)) / delay(close, 20) * 100

# 三重指数平滑MACD：类似MACD，反映趋势转折。DIF上穿DEA时为正。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
def Alpha89(close):
    A = ewm_mean(close, alpha=2/13)
    B = ewm_mean(A, alpha=2/27)
    return 2 * (A - B - ewm_mean(A - B, alpha=2/10))

# VWAP与成交量排名相关性负向：量价同步性高时取负，捕捉转折。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
def Alpha90(volume, vwap):
    return -1 * rank(correlation(rank(vwap), rank(volume), 5))

# 收盘偏离高点与均量/低价相关性的负向：价格远离高点且成交量与低价相关性强时，视为弱势，取负后看多。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((RANK((CLOSE - MAX(CLOSE,5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
def Alpha91(close, low, volume):
    return -1 * rank(close - ts_max(close, 5)) * rank(correlation(sma(volume, 40), low, 5))

# 加权价格变化衰减与长期量价相关性衰减的最大负值：捕捉短期与长期结构异动后的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)), TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE,13)),5),15)) * -1)
def Alpha92(close, volume, vwap):
    val = np.maximum(rank(decay_linear(delta(close * 0.35 + vwap * 0.65, 2), 3)),
                     rank_ts(decay_linear(np.abs(correlation(sma(volume, 180), close, 13)), 5), 15))
    result = -1 * val
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 20日开盘潜在下跌幅度累积：开盘低于前开时，取开盘与最低价或开盘与前开的较大差值的和，衡量开盘弱势累积。因子值越大 → 开盘弱势严重，可能超卖反弹，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
def Alpha93(Open, low):
    return ts_sum(np.where(Open >= delay(Open, 1), 0, np.maximum(Open - low, Open - delay(Open, 1))), 20)

# 30日净成交量：涨日量正，跌日量负，长周期资金流。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
def Alpha94(close, volume):
    return ts_sum(np.where(close > delay(close, 1), volume,
                           np.where(close < delay(close, 1), -volume, 0)), 30)

# 成交额波动（20日）：中期成交额标准差，波动放大可能预示变盘。因子值越大 → 活跃度突变，暂定为看多（流动性改善）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：STD(AMOUNT,20)
def Alpha95(volume, vwap):
    return stddev(volume * vwap, 20)

# 双重平滑威廉指标（9日）：二次平滑的随机指标，进一步滤噪，捕捉稳定超卖。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
def Alpha96(close, high, low):
    A = (close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100
    B = ewm_mean(A, alpha=1/3)
    return ewm_mean(B, alpha=1/3)

# 成交量10日标准差：短期成交量波动率。因子值越大 → 成交量异常放大，可能伴随趋势转折，反转看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：STD(VOLUME,10)
def Alpha97(volume):
    return stddev(volume, 10)

# 百日趋势条件翻转：如果长期趋势（100日均线变化率）小于等于5%，则取反转值（100日低点距离），否则取3日价格负变化。捕捉长期盘整后的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)||(==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3)))
def Alpha98(close):
    cond = (delta(sma(close, 100), 100) / delay(close, 100)) <= 0.05
    result = np.where(cond, -1 * (close - ts_min(close, 100)), -1 * delta(close, 3))
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 收盘价与成交量排名协方差负向：收盘排名与量排名协方差取负，捕捉量价同向后的背离。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))
def Alpha99(close, volume):
    return -1 * rank(covariance(rank(close), rank(volume), 5))

# 成交量20日标准差：中期成交量波动率，高波动预示关注度异常。因子值越大 → 看多（异动后可能趋势延续或反转，通常作为反转信号）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：STD(VOLUME,20)
def Alpha100(volume):
    return stddev(volume, 20)

# 相关性比较条件：比较收盘价与长期成交量均值的相关性和高价与成交量排名的相关性，前者小于后者时取-1，否则0。取负后成为正向信号，表示价量关系更有利于反转。
# 典型用法：横截面选股，做多因子值为 1 的股票，做空因子值为 0 的股票。
# 公式：((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30),37),15)) < RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))), RANK(VOLUME),11))) * -1)
def Alpha101(close, high, volume, vwap):
    cond = (rank(correlation(close, ts_sum(sma(volume, 30), 37), 15)) <
            rank(correlation(rank(high * 0.1 + vwap * 0.9), rank(volume), 11)))
    # 布尔条件，转换为整数 -1 或 0
    result = -1 * cond.astype(int)
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 成交量RSI（6日）：类似于相对强弱指数但用于成交量，上涨量占比。因子值越大，成交量放大趋势越强，视为活跃度提升，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
def Alpha102(volume):
    A = np.maximum(volume - delay(volume, 1), 0)
    B = np.abs(volume - delay(volume, 1))
    A_df = pd.DataFrame(A, index=volume.index, columns=volume.columns)
    B_df = pd.DataFrame(B, index=volume.index, columns=volume.columns)
    result = ewm_mean(A_df, alpha=1/6) / ewm_mean(B_df, alpha=1/6) * 100
    return result

# 距离20日低点天数比例：反映当前价格在近20日中的相对位置（越低则数值越小? 实际为 (20- low_day)/20 *100，值越大表示距离低点越近，超卖特征）。因子值越大 → 超卖程度越高，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((20-LOWDAY(LOW,20))/20)*100
def Alpha103(low):
    return (20 - low_day(low, 20)) / 20 * 100

# 量价相关性变化与波动率乘积取负：5日量价相关性变化乘以收盘价波动率排名后取负，捕捉量价关系转变与高波动带来的反转机会。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20)))
def Alpha104(close, high, volume):
    return -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))

# 开盘与成交量排名负相关：开盘价排名与成交量排名相关系数取负，偏离正相关时（即开盘价高但量排名低，或反之）看多。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * CORR(RANK(OPEN), RANK(VOLUME), 10)
def Alpha105(Open, volume):
    return -1 * correlation(rank(Open), rank(volume), 10)

# 20日价格动量（简单差值）：直接衡量20个交易日价格上涨幅度。因子值越大 → 看多（强势）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：CLOSE - DELAY(CLOSE,20)
def Alpha106(close):
    return close - delay(close, 20)

# 开盘价与前日高低收多维偏离负向乘积：开盘价高于前高、前收、前低的程度越弱（或者负向偏离越大），因子值越大，捕捉开盘弱势后的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((-1 * RANK((OPEN - DELAY(HIGH,1)))) * RANK((OPEN - DELAY(CLOSE,1)))) * RANK((OPEN - DELAY(LOW,1)))
def Alpha107(Open, close, high, low):
    return -1 * rank(Open - delay(high, 1)) * rank(Open - delay(close, 1)) * rank(Open - delay(low, 1))

# 短期最高价变化与VWAP长期量价相关性幂取负：高价变化排名与VWAP/均量相关性排名幂次方后取负，捕捉价格冲高动能的衰退。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((RANK((HIGH - MIN(HIGH,2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)
def Alpha108(high, volume, vwap):
    return -1 * (rank(high - ts_min(high, 2)) ** rank(correlation(vwap, sma(volume, 120), 6)))

# 振幅的SMA相对强度：10日指数平滑振幅与二次平滑振幅的比值，反映振幅扩张或收缩的状态。因子值大于1表示振幅扩张，可能伴随趋势转折。因子值越大 → 波动放大后的反转可能，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(HIGH-LOW,10,2) / SMA(SMA(HIGH-LOW,10,2),10,2)
def Alpha109(high, low):
    A = ewm_mean(high - low, alpha=2/10)
    return A / ewm_mean(A, alpha=2/10)

# 20日上涨潜能比（基于前收）：收盘价上方潜在涨幅与下方潜在跌幅的累积比，表示买方力量相对卖方更强。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20) / SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
def Alpha110(close, high, low):
    return ts_sum(np.maximum(0, high - delay(close, 1)), 20) / ts_sum(np.maximum(0, delay(close, 1) - low), 20) * 100

# 量价资金流双均线差：11日与4日指数平滑的成交加权价格位置之差，类似MACD，反映资金流加速度。因子值越大 → 资金流入加速，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2) - SMA(...,4,2)
def Alpha111(close, high, low, volume):
    A = volume * ((close - low) - (high - close)) / (high - low)
    return ewm_mean(A, alpha=2/11) - ewm_mean(A, alpha=2/4)

# 钱德动量摆动指标（CMO）：12日上涨幅度与下跌幅度的差除以总和，衡量纯动量。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((...<0?ABS(...):0),12)) / (SUM(上涨)+SUM(下跌))*100
def Alpha112(close):
    up = ts_sum(np.where(close - delay(close, 1) > 0, close - delay(close, 1), 0), 12)
    dn = ts_sum(np.where(close - delay(close, 1) < 0, np.abs(close - delay(close, 1)), 0), 12)
    return (up - dn) / (up + dn) * 100

# 延迟收盘均值排名与量价相关性乘积取负：利用5日延迟收盘的20日均线排名乘以短期量价相关性和自相关，取负捕捉均值回复。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(-1 * ((RANK((SUM(DELAY(CLOSE,5),20)/20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE,5), SUM(CLOSE,20), 2))))
def Alpha113(close, volume):
    return -1 * rank(ts_sum(delay(close, 5), 20) / 20) * correlation(close, volume, 2) * \
           rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))

# 价格区间与成交量、VWAP的相对比值：使用振幅、均收盘和VWAP的结构，综合反映价格极端特性和成交量分布。因子值越大 → 可能存在定价异常，视为反转机会，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,5)/5)),2)) * RANK(RANK(VOLUME))) / (((HIGH-LOW)/(SUM(CLOSE,5)/5)) / (VWAP-CLOSE)))
def Alpha114(close, high, low, volume, vwap):
    return rank(delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rank(rank(volume)) / \
           (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))

# 量价相关性排名幂：高加权价格与均量相关性的排名，幂以高低中点时序排名与量排名相关性的排名，放大正反馈。因子值越大 → 趋势强度高，但可能反转，按原意看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)), MEAN(VOLUME,30),10)) ^ RANK(CORR(TSRANK(((HIGH+LOW)/2),4), TSRANK(VOLUME,10),7)))
def Alpha115(close, high, low, volume):
    return rank(correlation(high * 0.9 + close * 0.1, sma(volume, 30), 10)) ** \
           rank(correlation(rank_ts((high + low) / 2, 4), rank_ts(volume, 10), 7))

# 20日线性回归斜率：用近20日收盘价拟合时间序列的贝塔系数，代表趋势方向与强度。因子值越大 → 上升趋势稳健，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：REGBETA(CLOSE, SEQUENCE, 20)
def Alpha116(close):
    return rolling_slope(close, 20)

# 成交量时序排名、价格区间排名与收益排名综合：三个时序排名相乘，捕捉量、价、收益的协同或背离。因子值越大 → 结合后给出正向预期。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((TSRANK(VOLUME,32) * (1-TSRANK(((CLOSE+HIGH)-LOW),16))) * (1-TSRANK(RET,32)))
def Alpha117(close, high, low, volume):
    return rank_ts(volume, 32) * (1 - rank_ts(close + high - low, 16)) * (1 - rank_ts(close / delay(close, 1) - 1, 32))

# 上涨/下跌实体比率（20日）：收盘高于开盘与开盘高于低点的累积比，衡量买盘相对卖盘的持续性。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(HIGH-OPEN,20) / SUM(OPEN-LOW,20) * 100
def Alpha118(Open, high, low):
    return ts_sum(high - Open, 20) / ts_sum(Open - low, 20) * 100

# VWAP相关性和开盘相关性衰减排名差：VWAP的短量相关性衰减排名减去开盘量相关性复杂衰减排名，差值为正则VWAP因素走强，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5),26),5),7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)),21),9),7),8)))
def Alpha119(Open, volume, vwap):
    return rank(decay_linear(correlation(vwap, ts_sum(sma(volume, 5), 26), 5), 7)) - \
           rank(decay_linear(rank_ts(np.minimum(correlation(rank(Open), rank(sma(volume, 15)), 21), 9), 7), 8))

# VWAP与收盘价差排名比：衡量均价与收盘价相对位置的秩比，正值表示均价低于收盘价，超买可能回调；原公式无负号，故因子值大偏多? 实际为 rank(VWAP-CLOSE)/rank(VWAP+CLOSE)，分子为正时偏多? 需结合回测，但通常VWAP<CLOSE表示强势，因子值大看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
def Alpha120(close, vwap):
    return rank(vwap - close) / rank(vwap + close)

# VWAP相对12日低点的偏离幂次取负：VWAP偏离低点排名以VWAP时序排名与均量排名相关性的幂次方，最终取负，表示VWAP低位回升乏力时的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((RANK((VWAP - MIN(VWAP,12)))^TSRANK(CORR(TSRANK(VWAP,20), TSRANK(MEAN(VOLUME,60),2),18),3)) * -1)
def Alpha121(volume, vwap):
    return -1 * rank(vwap - ts_min(vwap, 12)) ** rank_ts(correlation(rank_ts(vwap, 20), rank_ts(sma(volume, 60), 2), 18), 3)

# 三重指数平滑的对数价格变化率：对数价格的三重平滑后计算日变化率，类似于TEMA的变化速度，捕捉趋势加速。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(...,1))/DELAY(...,1)
def Alpha122(close):
    A = ewm_mean(ewm_mean(ewm_mean(np.log(close), alpha=2/13), alpha=2/13), alpha=2/13)
    return (A - delay(A, 1)) / delay(A, 1)

# 高低中点与均量相关性和低价量相关性条件：前者排名小于后者时取-1，否则0，再取负变为1信号。因子值为1时看多。
# 典型用法：横截面选股，做多因子值为1的股票，做空因子值为0的股票。
# 公式：((RANK(CORR(SUM(((HIGH+LOW)/2),20), SUM(MEAN(VOLUME,60),20),9)) < RANK(CORR(LOW, VOLUME,6))) * -1)
def Alpha123(high, low, volume):
    cond = (rank(correlation(ts_sum((high + low) / 2, 20), ts_sum(sma(volume, 60), 20), 9)) <
            rank(correlation(low, volume, 6)))
    result = -1 * cond.astype(int)
    return pd.DataFrame(result, index=high.index, columns=high.columns)

# 收盘价与VWAP乖离除以衰减排名：乖离相对于30日最高收盘价排名衰减，标准化乖离率。因子值越大 → 收盘价高于VWAP程度大，短期强势，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)), 2)
def Alpha124(close, vwap):
    return (close - vwap) / decay_linear(rank(ts_max(close, 30)), 2)

# 长期VWAP均量相关性与加权收盘价变化比值：长期相关性强与短期价格变化缓慢的比值。因子值越大 → 长期关系稳定而短期未变，可能酝酿突破或反转，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17),20)) / RANK(DECAYLINEAR(DELTA(((CLOSE*0.5)+(VWAP*0.5)),3),16)))
def Alpha125(close, volume, vwap):
    return rank(decay_linear(correlation(vwap, sma(volume, 80), 17), 20)) / \
           rank(decay_linear(delta(close * 0.5 + vwap * 0.5, 3), 16))

# 典型价格：简单的收盘、最高、最低的平均值，代表当日价格重心。
# 典型用法：通常不作为独立因子，但可用于构建其他指标。
# 公式：(CLOSE + HIGH + LOW) / 3
def Alpha126(close, high, low):
    return (close + high + low) / 3

# 12日价格波动率（相对最大值的均方根）：衡量近期价格从高点回撤的幅度，类似回撤波动。因子值越大 → 回撤大，超卖反转机会，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
def Alpha127(close):
    return np.sqrt(sma((100 * (close - ts_max(close, 12)) / ts_max(close, 12)) ** 2, 12))

# 典型价格资金流强度：基于典型价格涨跌和成交量的资金流指标，类似MFI（资金流量指数）的变体。因子值越大 → 资金流入强，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：100-(100/(1+SUM((典型价格涨)*量,14)/SUM((典型价格跌)*量,14)))
def Alpha128(close, high, low, volume):
    A = (high + low + close) / 3
    up = ts_sum(np.where(A > delay(A, 1), A * volume, 0), 14)
    dn = ts_sum(np.where(A < delay(A, 1), A * volume, 0), 14)
    return 100 - (100 / (1 + up / dn))

# 12日下跌幅度累积：仅统计下跌日的绝对跌幅累积，反映近期空方力量。因子值越大 → 空方宣泄充分，可能超卖反弹，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
def Alpha129(close):
    return ts_sum(np.where(close - delay(close, 1) < 0, np.abs(close - delay(close, 1)), 0), 12)

# 高低中点均量相关性与VWAP排名相关性之比：形容价格结构相对量价关系的强弱。因子值大看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2), MEAN(VOLUME,40),9),10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME),7),3)))
def Alpha130(high, low, volume, vwap):
    return rank(decay_linear(correlation((high + low) / 2, sma(volume, 40), 9), 10)) / \
           rank(decay_linear(correlation(rank(vwap), rank(volume), 7), 3))

# VWAP变化排名幂乘收盘与均量相关性排名：VWAP突变与收盘量能稳定性的混合效应，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
def Alpha131(close, volume, vwap):
    return rank(delta(vwap, 1)) ** rank_ts(correlation(close, sma(volume, 50), 18), 18)

# 20日平均成交额：直接衡量中期流动性水平。因子值越大 → 流动性越好，通常与未来收益正相关。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MEAN(AMOUNT, 20)
def Alpha132(volume, vwap):
    return sma(volume * vwap, 20)

# 距20日高低点天数差：近期高点天数与低点天数之差，正值表示更靠近高点（强势），负值靠近低点。因子值越大 → 远离低点，趋势偏强，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((20-HIGHDAY(HIGH,20))/20)*100 - ((20-LOWDAY(LOW,20))/20)*100
def Alpha133(high, low):
    return (20 - high_day(high, 20)) / 20 * 100 - (20 - low_day(low, 20)) / 20 * 100

# 12日收益乘以成交量：量增价涨的乘数效应。因子值越大 → 放量上涨，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12) * VOLUME
def Alpha134(close, volume):
    return (close - delay(close, 12)) / delay(close, 12) * volume

# 20日涨幅的延迟指数平滑：20日涨幅的1日延迟再用20日平滑，捕捉动量的持续性。因子值越大 → 动量惯性延续，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
def Alpha135(close):
    A = delay(close / delay(close, 20), 1)
    return ewm_mean(A, alpha=1/20)

# 收益变化与开盘量相关性取负：3日收益变化排名乘以开盘量与成交量的相关性，取负值捕捉收益动能衰退。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((-1 * RANK(DELTA(RET,3))) * CORR(OPEN, VOLUME, 10))
def Alpha136(Open, close, volume):
    ret = close / delay(close, 1) - 1
    return -1 * rank(delta(ret, 3)) * correlation(Open, volume, 10)

# 复杂跳跃波动因子（占位）：实现过于复杂，涉及价格跳跃条件加权，当前返回0。
# 典型用法：待完整实现后确定。
# 公式：16*(CLOSE-DELAY(CLOSE,1)+...)/...
def Alpha137(Open, close, high, low):
    result = np.zeros_like(close)
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 低价VWAP混合变化与量价深层次时序排名差取负：衡量低价成分变化与复杂量价结构的背离。背离时看多。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((RANK(DECAYLINEAR(DELTA((((LOW*0.7)+(VWAP*0.3))),3),20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8), TSRANK(MEAN(VOLUME,60),17),5),19),16),7)) * -1)
def Alpha138(low, volume, vwap):
    return -1 * (rank(decay_linear(delta(low * 0.7 + vwap * 0.3, 3), 20)) -
                 rank_ts(decay_linear(rank_ts(correlation(rank_ts(low, 8), rank_ts(sma(volume, 60), 17), 5), 19), 16), 7))

# 开盘与成交量负相关：量价关系异常（开盘价与成交量反向）时看多。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * CORR(OPEN, VOLUME, 10)
def Alpha139(Open, volume):
    return -1 * correlation(Open, volume, 10)

# 排名差分的衰减最小化：比较价格排名组合的衰减与收盘量能时序相关性的衰减，取较小者。捕捉结构性弱势后的修复。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MIN(RANK(DECAYLINEAR(((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE))),8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8), TSRANK(MEAN(VOLUME,60),20),8),7),3))
def Alpha140(Open, close, high, low, volume):
    val = np.minimum(rank(decay_linear(rank(Open) + rank(low) - (rank(high) + rank(close)), 8)),
                     rank_ts(decay_linear(correlation(rank_ts(close, 8), rank_ts(sma(volume, 60), 20), 8), 7), 3))
    result = val
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 高价排名与均量排名的负相关：量能无法支撑高价时反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
def Alpha141(high, volume):
    return -1 * rank(correlation(rank(high), rank(sma(volume, 15)), 9))

# 收盘时序排名、价格二阶差分、量比排名三者乘积取负：捕捉多重动量与反转信号的综合。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((-1 * RANK(TSRANK(CLOSE,10))) * RANK(DELTA(DELTA(CLOSE,1),1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)),5))
def Alpha142(close, volume):
    return -1 * rank(rank_ts(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(rank_ts(volume / sma(volume, 20), 5))

# 自累积复利因子：当日收益率大于0时，按复利累积；否则保持不变。类似连续上涨的累计收益。因子值越大 → 近期连续上涨能力强，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：CLOSE>DELAY(CLOSE,1)?(CLOSE/DELAY(CLOSE,1)-1)*SELF:SELF
def Alpha143(close):
    ratio = close / delay(close, 1)   # DataFrame
    def acc_func(prev, curr):
        # curr > 1 表示上涨（因为 ratio = 1 + 收益率）
        return (curr - 1) * prev if curr > 1 else prev
    return accumulation(ratio, acc_func)

# 下跌日收益绝对值/成交额的均值：衡量下跌时单位成交额的亏损程度，因子越大表示下跌时亏损大但可能超卖。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1)) / COUNT(CLOSE<DELAY(CLOSE,1),20)
def Alpha144(close, volume, vwap):
    cond = close < delay(close, 1)
    return ts_sum(np.where(cond, np.abs(close / delay(close, 1) - 1) / (volume * vwap), 0), 20) / \
           ts_sum(cond.astype(int), 20)

# 成交量均线交叉（9,26,12）：类似MACD的成交量版，快线减慢线除以12日均线，标准化量能变化。因子值越大 → 量能趋势增强，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
def Alpha145(volume):
    return (sma(volume, 9) - sma(volume, 26)) / sma(volume, 12) * 100

# 收益率的长期均值回归分数：类似于Z-score与信号强度乘积，衡量当前收益率偏离长期指数均线的程度及方向。因子值越大 → 正向偏离（上涨拉力强），看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA(...,61,2),20) * ((...)/SMA(...))
def Alpha146(close):
    A = (close - delay(close, 1)) / delay(close, 1)
    B = ewm_mean(A, alpha=2/61)
    return sma(A - B, 20) * (A - B) / sma((A - B) ** 2, 60)

# 12日均线的12期线性趋势斜率：均线本身的趋势强度，二次确认。因子值越大 → 趋势强化，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：REGBETA(MEAN(CLOSE,12), SEQUENCE(12))
def Alpha147(close):
    return rolling_slope(sma(close, 12), 12)

# 开盘价相关性与低点距离条件：开盘与均量相关性排名小于开盘与14日低点距离排名时取1（看多），否则0。捕捉开盘稳定且接近低点的反弹。
# 典型用法：横截面选股，做多信号值为1的股票，做空信号值为0的股票。
# 公式：((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60),9),6)) < RANK((OPEN - TSMIN(OPEN,14)))) * -1)
def Alpha148(Open, volume):
    cond = (rank(correlation(Open, ts_sum(sma(volume, 60), 9), 6)) < rank(Open - ts_min(Open, 14)))
    result = -1 * cond.astype(int)
    return pd.DataFrame(result, index=Open.index, columns=Open.columns)

# 下跌市场Beta（条件Beta）：仅在大盘下跌时计算个股相对于大盘的252日Beta，衡量系统性风险。Beta较低表示抗跌。因子值越大 → 系统风险高，通常看空，但此处可能作为反转？常规说明：高Beta在下跌市场不利，故因子值大视为负面，但原公式无负号，暂按风险因子描述，具体方向需验证。
# 典型用法：可作为风险控制因子，通常做多低Beta（因子值小）的股票。
# 公式：REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1, INDEX_CLOSE<DELAY(INDEX_CLOSE,1)), ...)
def Alpha149(close, index_close):
    cond = index_close < delay(index_close, 1)
    x = (close / delay(close, 1) - 1) * cond
    y = (index_close / delay(index_close, 1) - 1) * cond
    # x 和 y 中 cond 为 False 的位置为 0，然后计算相关性
    result = correlation(x.fillna(0), y.fillna(0), 252)
    return result

# 典型价格成交额：价格重心乘以成交量，综合价格和量的规模。因子值越大 → 市场参与度高且价格重心高，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE+HIGH+LOW)/3 * VOLUME
def Alpha150(close, high, low, volume):
    return (close + high + low) / 3 * volume
# 20日动量的指数平滑：对20日价格变化进行20日指数平滑，得到稳定的中期动量指标。因子值越大 → 中期上升趋势稳固，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(CLOSE - DELAY(CLOSE,20), 20, 1)
def Alpha151(close):
    return ewm_mean(close - delay(close, 20), alpha=1/20)

# 价格比例的双重MACD：基于收盘价/延迟收盘价比率构建的慢速MACD变体，捕捉长期趋势转变。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(...),1),26),9,1)
def Alpha152(close):
    A = ewm_mean(delay(close / delay(close, 9), 1), alpha=1/9)
    B = sma(delay(A, 1), 12) - sma(delay(A, 1), 26)
    return ewm_mean(B, alpha=1/9)

# 多周期均线均值：3/6/12/24日均线的平均值，反映价格的综合均值水平。因子值本身不直接代表方向，但常与其他因子结合使用。
# 典型用法：可作为基础参考，或与收盘价比较判断超买超卖。若用于反转，值大意味着价格低于多周期均价，看多。
# 公式：(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
def Alpha153(close):
    return (sma(close, 3) + sma(close, 6) + sma(close, 12) + sma(close, 24)) / 4

# VWAP偏离最低值与量价相关性的条件：VWAP距16日最低值的距离小于VWAP与180日均量的相关性时，信号为真（1），否则0。视为VWAP相对低估且量价关系稳定，看多。
# 典型用法：横截面选股，做多信号为1的股票。
# 公式：(((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))
def Alpha154(volume, vwap):
    cond = (vwap - ts_min(vwap, 16)) < correlation(vwap, sma(volume, 180), 18)
    result = cond.astype(int)
    return pd.DataFrame(result, index=vwap.index, columns=vwap.columns)

# 成交量MACD：类似价格的MACD指标，衡量成交量短期与长期趋势的差异。因子值越大 → 成交量扩张加速，看多（有量能配合）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
def Alpha155(volume):
    A = ewm_mean(volume, alpha=2/13)
    B = ewm_mean(A, alpha=2/27)
    return A - B - ewm_mean(A - B, alpha=2/10)

# VWAP变化和开盘/低价加权变化衰减的最大负值：捕捉VWAP趋势和开盘低价混合变化背离后的反转。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)), RANK(DECAYLINEAR(((DELTA((OPEN*0.15)+(LOW*0.85)),2)/(...))*-1),3))) * -1)
def Alpha156(Open, low, vwap):
    val = np.maximum(rank(decay_linear(delta(vwap, 5), 3)),
                     rank(decay_linear(-1 * (delta(Open * 0.15 + low * 0.85, 2) / (Open * 0.15 + low * 0.85)), 3)))
    result = -1 * val
    return pd.DataFrame(result, index=Open.index, columns=Open.columns)

# 复杂嵌套的多重排名与累积（占位）：因子结构极复杂，涉及多次排名、对数、求和、时序最小值和延迟收益。
# 典型用法：待完整实现后确定。
# 公式：(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5) + TSRANK(DELAY((-1*RET),6),5))
def Alpha157(close, returns):
    # 内层：DELTA((CLOSE-1), 5)
    delta_part = delta(close - 1, 5)
    # -1 * RANK(DELTA(...))
    step1 = -1 * rank(delta_part)
    # RANK(RANK(...))
    step2 = rank(rank(step1))
    # TSMIN(... , 2)   -- 滚动2天取最小值
    step3 = ts_min(step2, 2)
    # SUM(TSMIN(...), 1)   -- 窗口1的滚动和，等于自身，但保留以对应原公式
    step4 = ts_sum(step3, 1)
    # LOG(...)   -- factor_utils 没有直接提供 log，此处用 numpy
    step5 = np.log(step4)
    # RANK(RANK(LOG(...)))
    step6 = rank(rank(step5))
    # PROD(... , 1)   -- 窗口1的滚动乘积，等于自身
    step7 = product(step6, 1)
    # MIN(PROD(...), 5)   -- 滚动5天取最小值
    left_part = ts_min(step7, 5)
    # 右半部分：TSRANK(DELAY((-1 * RET), 6), 5)
    right_part = ts_rank(delay(-1 * returns, 6), 5)
    result = left_part + right_part
    # 最终结果保证为 DataFrame
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 价格通道相对宽度：(最高价-指数均线)与(最低价-指数均线)的差值除以收盘价，反映价格偏离均线的非对称程度。因子值越大 → 价格相对于均线偏向上方，强势，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
def Alpha158(close, high, low):
    A = ewm_mean(close, alpha=2/15)
    return ((high - A) - (low - A)) / close

# 多周期价格相对最小值的加权幅度：综合6/12/24日价格与历史低点的偏离幅度，衡量中期超卖程度。因子值越大 → 离低点越远（回升），看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24 + ...)*100/(6*12+6*24+12*24)
def Alpha159(close, high, low):
    m = delay(close, 1)
    tmp1 = (close - ts_sum(np.minimum(low, m), 6)) / ts_sum(np.maximum(high, m) - np.minimum(low, m), 6) * 12 * 24
    tmp2 = (close - ts_sum(np.minimum(low, m), 12)) / ts_sum(np.maximum(high, m) - np.minimum(low, m), 12) * 6 * 24
    tmp3 = (close - ts_sum(np.minimum(low, m), 24)) / ts_sum(np.maximum(high, m) - np.minimum(low, m), 24) * 6 * 24
    return (tmp1 + tmp2 + tmp3) * 100 / (6*12 + 6*24 + 12*24)

# 下跌波动率平滑：仅考虑下跌日的波动率，经指数平滑，衡量下跌过程中的风险累积。因子值越大 → 下跌波动剧烈，可能超卖反弹，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
def Alpha160(close):
    A = np.where(close <= delay(close, 1), stddev(close, 20), 0)
    return ewm_mean(pd.DataFrame(A, index=close.index, columns=close.columns), alpha=1/20)

# 12日真实波幅均值：衡量平均日内波动与隔夜跳空风险。因子值越大 → 波动性高，可能伴随超卖或趋势转折机会，视组合方向而定，一般作为风险指标。此处暂定为看多（波动后的反转）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票（需谨慎验证）。
# 公式：MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
def Alpha161(close, high, low):
    val = np.maximum(np.maximum(high - low, np.abs(delay(close, 1) - high)), np.abs(delay(close, 1) - low))
    # sma 要求 DataFrame，先将 val 转为 DataFrame
    val_df = pd.DataFrame(val, index=close.index, columns=close.columns)
    result = sma(val_df, 12)
    return result

# RSI随机化处理：12日RSI值减去其12日最小值，再除以其12日范围，得到相对位置（Stochastic RSI）。因子值越大 → RSI处于近期高位，强势看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(...),12,1)*100 - MIN(...)) / (MAX(...)-MIN(...))
def Alpha162(close):
    A = np.maximum(close - delay(close, 1), 0)
    B = np.abs(close - delay(close, 1))
    C = ewm_mean(A, alpha=1/12)
    D = ewm_mean(B, alpha=1/12)
    ratio = C / D * 100
    return (ratio - ts_min(ratio, 12)) / (ts_max(ratio, 12) - ts_min(ratio, 12))

# 负收益与量价综合反转：负的日收益率、20日均量、VWAP和高收盘差四者乘积排名，捕捉大幅下跌且量能异常后的反弹。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
def Alpha163(close, high, volume, vwap):
    return rank(-1 * (close / delay(close, 1) - 1) * sma(volume, 20) * vwap * (high - close))

# 价格变动速度距其低点的振幅比：取上涨日价格变动的倒数与1比较后减去12日最低，再除以振幅，衡量变动速率从低位回升的程度。因子值越大 → 变动速率加速，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(...,12))/(HIGH-LOW)*100,13,2)
def Alpha164(close, high, low):
    A = np.where(close > delay(close, 1), 1 / (close - delay(close, 1)), 1)
    A_df = pd.DataFrame(A, index=close.index, columns=close.columns)
    B = (A_df - ts_min(A_df, 12)) / (high - low) * 100
    result = ewm_mean(B.fillna(0), alpha=2/13)
    return result

# 48日累积离差极值除以标准差：价格累积偏离均值的范围与标准差的比值，类似长期通道宽度。因子值越大 → 价格处于极端状态（可能超卖），看多反转。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
def Alpha165(close):
    cumsum = ts_sum(close - sma(close, 48), 48)
    # cumsum 是 DataFrame，max/min 返回 Series，需转为 DataFrame
    series_result = (cumsum.max(axis=1) - cumsum.min(axis=1)) / stddev(close, 48)
    result = pd.DataFrame(series_result, index=close.index, columns=close.columns)
    return result

# 收益偏度系数（修正）：复杂公式，衡量收益率的三阶矩特征，负值表示收益左偏。因子值越大（通常负得少或正） → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-20 * (19^1.5) * SUM(...) / ((19*18)*(SUM(...))^1.5)
def Alpha166(close):
    return -20 * (19**1.5) * ts_sum(close / delay(close, 1) - 1 - sma(close / delay(close, 1) - 1, 20), 20) / \
           (19 * 18 * (ts_sum(sma(close / delay(close, 1), 20)**2, 20)**1.5))

# 12日上涨幅度累积：只统计上涨日的涨幅累积，代表近期多头力量。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
def Alpha167(close):
    return ts_sum(np.where(close > delay(close, 1), close - delay(close, 1), 0), 12)

# 负量比：成交量除以20日均量再取负，放量时因子值小（负值更负），缩量时大。捕捉缩量后的反弹。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：-1 * VOLUME / MEAN(VOLUME,20)
def Alpha168(volume):
    return -1 * volume / sma(volume, 20)

# 价格变化MACD的指数平滑：对日价格变化的9日指数平滑值，计算其12日与26日均线差，再10日平滑，类似MACD。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(...,1),26),10,1)
def Alpha169(close):
    A = ewm_mean(close - delay(close, 1), alpha=1/9)
    B = sma(delay(A, 1), 12) - sma(delay(A, 1), 26)
    return ewm_mean(B, alpha=1/10)

# 倒数收盘价与高价动量的复合：结合低价股效应（1/收盘价）、成交量、高价距收盘价差的累计。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(((RANK((1/CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH-CLOSE))) / (SUM(HIGH,5)/5))) - RANK((VWAP - DELAY(VWAP,5)))
def Alpha170(close, high, volume, vwap):
    return rank(1 / close) * volume / sma(volume, 20) * high * rank(high - close) / (ts_sum(high, 5)/5) - rank(vwap - delay(vwap, 5))

# 非线性结构：低点、收盘、开盘、高点的五次方组合，捕捉极端价格与非线性关系。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
def Alpha171(Open, close, high, low):
    return -1 * (low - close) * (Open**5) / ((close - high) * (close**5))

# 趋向系统平均差异（6日平滑）：正向运动和负向运动14日累计值之差除以和，再6日平滑，衡量趋向强度。因子值越大 → 多头趋向明确，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM(上涨)+SUM(下跌))*100,6)
def Alpha172(close, high, low):
    HD = high - delay(high, 1)
    LD = delay(low, 1) - low
    TR = np.maximum(np.maximum(high - low, np.abs(high - delay(close, 1))), np.abs(low - delay(close, 1)))
    sum1 = ts_sum(np.where((LD > 0) & (LD > HD), LD, 0), 14) * 100 / ts_sum(TR, 14)
    sum2 = ts_sum(np.where((HD > 0) & (HD > LD), HD, 0), 14) * 100 / ts_sum(TR, 14)
    return sma(np.abs(sum1 - sum2) / (sum1 + sum2) * 100, 6)

# TEMA变体：三重指数平滑的组合，3*EMA1 - 2*EMA2 + EMA3。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
def Alpha173(close):
    A = ewm_mean(close, alpha=2/13)
    B = ewm_mean(A, alpha=2/13)
    return 3 * A - 2 * B + ewm_mean(B, alpha=2/13)

# 上涨日波动率平滑：仅统计上涨日的波动率指数平滑，代表上行风险与机会。因子值越大 → 上涨波动大，强势，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
def Alpha174(close):
    A = np.where(close > delay(close, 1), stddev(close, 20), 0)
    result = ewm_mean(pd.DataFrame(A, index=close.index, columns=close.columns), alpha=1/20)
    return result

# 6日真实波幅均值：短周期的平均真实波幅，反映近期价格振动幅度。因子值越大 → 波动剧烈，可能超卖反弹，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
def Alpha175(close, high, low):
    val = np.maximum(np.maximum(high - low, np.abs(delay(close, 1) - high)), np.abs(delay(close, 1) - low))
    val_df = pd.DataFrame(val, index=close.index, columns=close.columns)
    result = sma(val_df, 6)
    return result

# 价格位置与成交量排名相关性：收盘价在12日高低区间的相对位置与成交量排名的相关系数。因子值大表示高位放量或低位缩量，捕捉量价配合的健康趋势。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：CORR(RANK(((CLOSE - TSMIN(LOW,12)) / (TSMAX(HIGH,12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
def Alpha176(close, high, low, volume):
    return correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)

# 距20日高点天数比例：反应价格距离近期高点的远近，数值越小表示越接近高点。因子值越大 → 离高点远（回调充分），看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((20-HIGHDAY(HIGH,20))/20)*100
def Alpha177(high):
    return (20 - high_day(high, 20)) / 20 * 100

# 量价齐驱（1日）：日收益率乘以当日成交量，放量上涨则大。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1) * VOLUME
def Alpha178(close, volume):
    return (close - delay(close, 1)) / delay(close, 1) * volume

# VWAP成交量相关性与低价均量相关性乘积：综合短期VWAP量价关系与长期低价量关系。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(CORR(VWAP, VOLUME, 4)) * RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
def Alpha179(low, volume, vwap):
    return rank(correlation(vwap, volume, 4)) * rank(correlation(rank(low), rank(sma(volume, 50)), 12))

# 放量反转信号：当日成交量大于20日均量时，取过去60日价格绝对变化的排名和方向的反转值；否则取负成交量。放量下捕捉反向。因子值越大（负得少） → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((MEAN(VOLUME,20) < VOLUME) ? ((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7))) : (-1*VOLUME))
def Alpha180(close, volume):
    result = np.where(sma(volume, 20) < volume,
                      -1 * rank_ts(np.abs(delta(close, 7)), 60) * np.sign(delta(close, 7)),
                      -1 * volume)
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 收益与指数偏度的复杂比率：结合个股收益偏离和指数收益的三阶矩，捕捉共振异常。因子值越大 → 个股相对指数偏强且波动结构有利，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN(...,20)) - (BANCHMARKINDEXCLOSE-MEAN(...))^2 ,20) / SUM(...^3)
def Alpha181(close, index_close):
    return ts_sum(((close / delay(close, 1) - 1) - sma(close / delay(close, 1) - 1, 20)) -
                  (index_close - sma(index_close, 20))**2, 20) / ts_sum((index_close - sma(index_close, 20))**3, 20)

# 个股与指数同向比例：近20日中个股涨跌与市场方向一致的天数占比，捕捉同步性。因子值越大 → 个股与市场同步性强，若市场上涨则有利，看多（当市场看涨时）。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票（需结合市场状态）。
# 公式：COUNT((CLOSE>OPEN & INDEX_CLOSE>INDEX_OPEN) OR (CLOSE<OPEN & INDEX_CLOSE<INDEX_OPEN),20)/20
def Alpha182(Open, close, index_Open, index_close):
    cond = ((close > Open) & (index_close > index_Open)) | ((close < Open) & (index_close < index_Open))
    return ts_sum(cond.astype(int), 20) / 20

# 24日累积离差极值比标准差：类似Alpha165但周期为24日，监控中期极端状态。因子值越大 → 极端状态蕴含反转机会，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
def Alpha183(close):
    cumsum = ts_sum(close - sma(close, 24), 24)
    series_result = (cumsum.max(axis=1) - cumsum.min(axis=1)) / stddev(close, 24)
    result = pd.DataFrame(series_result, index=close.index, columns=close.columns)
    return result

# 开盘收盘差值的长期相关性加当前差值：昨日开盘-收盘差值与收盘价的200日相关性加上今日差值排名。因子值越大 → 近期开盘弱于收盘或相关性转折，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
def Alpha184(Open, close):
    return rank(correlation(delay(Open - close, 1), close, 200)) + rank(Open - close)

# 开盘涨跌幅平方负向排名：公式为 RANK(-(1-OPEN/CLOSE)^2)，当开盘价接近收盘价（小涨跌）时负值小，排名高；大幅跳空时排名低。因子值越大 → 开盘价较平稳，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
def Alpha185(Open, close):
    return rank(-1 * (1 - Open / close)**2)

# 趋向系统双平滑差异：类似Alpha172但增加了6日延迟平滑的平均，减弱噪音。因子值越大 → 多头趋向稳定，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：(MEAN(ABS(SUM(LD...)-SUM(HD...))/(...)*100,6) + DELAY(MEAN(...),6)) / 2
def Alpha186(close, high, low):
    HD = high - delay(high, 1)
    LD = delay(low, 1) - low
    TR = np.maximum(np.maximum(high - low, np.abs(high - delay(close, 1))), np.abs(low - delay(close, 1)))
    sum1 = ts_sum(np.where((LD > 0) & (LD > HD), LD, 0), 14) * 100 / ts_sum(TR, 14)
    sum2 = ts_sum(np.where((HD > 0) & (HD > LD), HD, 0), 14) * 100 / ts_sum(TR, 14)
    series = np.abs(sum1 - sum2) / (sum1 + sum2) * 100
    return (sma(series, 6) + delay(sma(series, 6), 6)) / 2

# 开盘向上跳空累积幅度：20日内开盘价高于昨日开盘价时的潜在上涨幅度累积，反映开盘强势动能。因子值越大 → 看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
def Alpha187(Open, high):
    return ts_sum(np.where(Open <= delay(Open, 1), 0, np.maximum(high - Open, Open - delay(Open, 1))), 20)

# 振幅相对其均线的偏离百分比：(今日振幅 - 11日指数均振幅) / 11日均振幅 *100。正值表示振幅扩张，可能预示变盘或动能。因子值越大 → 振幅扩张（看作活跃度上升），暂看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((HIGH-LOW - SMA(HIGH-LOW,11,2)) / SMA(HIGH-LOW,11,2)) * 100
def Alpha188(high, low):
    A = ewm_mean(high - low, alpha=2/11)
    return (high - low - A) / A * 100

# 6日收盘价绝对偏离均值的平均值：衡量短期价格偏离均值的平均幅度，类似于短期波动率。因子值越大 → 价格不稳定，可能超卖反弹，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
def Alpha189(close):
    return sma(np.abs(close - sma(close, 6)), 6)

# 复杂对数比率（占位）：公式极复杂，涉及多个条件求和与对数，当前无法直接实现，返回0。
# 典型用法：待完整实现后确定。
# 公式：LOG((COUNT(CLOSE/DELAY(CLOSE)-1>...)...)
def Alpha190(close):
    result = np.zeros_like(close)
    return pd.DataFrame(result, index=close.index, columns=close.columns)

# 均量-低价相关性加上典型价格减收盘价：综合了成交量与低价的关系以及日内价格重心与收盘价的差异。因子值越大 → 重心上移或量价关系健康，看多。
# 典型用法：横截面选股，做多因子值高的股票，做空因子值低的股票。
# 公式：((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
def Alpha191(close, high, low, volume):
    return correlation(sma(volume, 20), low, 5) + (high + low) / 2 - close