# factor_alphas.py
import numpy as np
import pandas as pd
from factor_utils import (
    ts_sum, sma, stddev, correlation, covariance, ts_rank, product,
    ts_min, ts_max, delta, delay, rank, scale, ts_argmax, ts_argmin,
    decay_linear, IndNeutralize
)
"""
因子命名规则：alpha1-alpha101，每个因子包含以下注释维度：
1. 逻辑：因子核心设计思路
2. 类型：量价因子/情绪因子/波动因子/行业中性因子等
3. 公式：数学表达式（基于代码逻辑还原）
4. 用法：
   - 用处：该因子捕捉的市场信号类型
   - 适用场景：适合的市场环境/时间周期
   - 数值含义：因子值大小/正负代表的市场意义
"""

# ------------------------------ Alpha1 ------------------------------
def alpha1(close, returns):
    """
    逻辑：利用收盘价平方的5日最大值排名，结合收益为负时的20日收益标准差，捕捉价格波动的极端信号
    类型：量价波动因子
    公式：
        (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    用法：
        - 用处：捕捉价格平方的短期极值对应的多空信号
        - 适用场景：震荡市中识别价格异动的个股
        - 数值含义：值>0代表价格平方短期极值偏多，值<0代表偏空，绝对值越大信号越强
    """
    x = close.copy()
    x[returns < 0] = stddev(returns, 20)
    alpha = rank(ts_argmax(x ** 2, 5)) - 0.5
    return alpha.fillna(0)

# ------------------------------ Alpha2 ------------------------------
def alpha2(Open, close, volume):
    """
    逻辑：通过成交量对数差分的排名与开盘收盘价差率排名的6日相关性，捕捉量价背离信号
    类型：量价相关因子
    公式：
        (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    用法：
        - 用处：识别成交量与价格变动方向的背离程度
        - 适用场景：趋势市中判断量价是否同步，背离时可能反转
        - 数值含义：值>0代表量价正向背离（多头信号），值<0代表负向背离（空头信号）
    """
    r1 = rank(delta(np.log(volume), 2))
    r2 = rank((close - Open) / Open)
    alpha = -1 * correlation(r1, r2, 6)
    return alpha.fillna(0)

# ------------------------------ Alpha3 ------------------------------
def alpha3(Open,volume):
    """
    逻辑：开盘价排名与成交量排名的10日相关性，捕捉开盘价与成交量的联动关系
    类型：量价联动因子
    公式：
         (-1 * correlation(rank(open), rank(volume), 10))
    用法：
        - 用处：判断开盘价与成交量是否同步变化，识别资金对开盘价的认可度
        - 适用场景：开盘阶段资金异动的捕捉，适合短线交易
        - 数值含义：值>0代表开盘价与成交量正向联动（资金认可开盘价），值<0代表反向联动
    """
    r1 = rank(Open)
    r2 = rank(volume)
    alpha = -1 * correlation(r1,r2,10)
    return alpha.replace([-np.inf, np.inf], 0).fillna(value = 0)

# ------------------------------ Alpha4 ------------------------------
def alpha4(low):
    """
    逻辑：最低价排名的9日时序排名，捕捉低价股的相对强弱
    类型：价格趋势因子
    公式：
        (-1 * Ts_Rank(rank(low), 9))
    用法：
        - 用处：识别最低价的短期相对强弱，捕捉低价股的超跌/超涨信号
        - 适用场景：底部震荡或低价股轮动行情
        - 数值含义：值>0代表最低价相对偏强（超涨），值<0代表相对偏弱（超跌）
    """
    r = rank(low)
    alpha = -1 * ts_rank(r,9)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha5 ------------------------------
def alpha5(Open,vwap,close):
    """
    逻辑：开盘价与10日VWAP均值的偏离排名，结合收盘价与VWAP偏离的绝对值排名，捕捉价格与平均成交价格的背离
    类型：成交均价偏离因子
    公式：
        (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    用法：
        - 用处：识别价格与平均成交价格的偏离程度，判断资金成交成本与价格的关系
        - 适用场景：主力资金建仓/出货阶段的识别
        - 数值含义：值>0代表价格高于成交均价且开盘价偏离均值（多头），值<0代表价格低于成交均价（空头）
    """
    alpha = (rank((Open - (ts_sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha6 ------------------------------
def alpha6(Open, volume):
    """
    逻辑：开盘价与成交量的10日相关性，捕捉开盘价与成交量的联动信号
    类型：量价相关因子
    公式：
         (-1 * correlation(open, volume, 10))
    用法：
        - 用处：判断开盘价变动与成交量的关联度，识别开盘阶段的资金动向
        - 适用场景：早盘异动股的筛选，适合日内交易
        - 数值含义：值>0代表开盘价与成交量正向相关（资金推升/打压开盘价），值<0代表反向相关
    """
    alpha = -1 * correlation(Open, volume, 10)
    return alpha.replace([-np.inf, np.inf], 0).fillna(value = 0)

# ------------------------------ Alpha7 ------------------------------
def alpha7(volume,close):
    """
    逻辑：结合20日均量筛选，通过收盘价7日差分的绝对值排名和方向，捕捉量能不足时的价格异动
    类型：量价趋势因子
    公式：
        ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    用法：
        - 用处：识别量能萎缩时的价格涨跌信号，量能不足时看空
        - 适用场景：趋势行情末端（量能跟不上价格）
        - 数值含义：值>0代表量能充足且价格上涨（多头），值=-1代表量能不足（空头）
    """
    adv20 = sma(volume, 20)
    alpha = -1 * ts_rank(abs(delta(close, 7)), 60) * np.sign(delta(close, 7))
    alpha[adv20 >= volume] = -1
    return alpha.fillna(value = 0)

# ------------------------------ Alpha8 ------------------------------
def alpha8(Open,returns):
    """
    逻辑：5日开盘价和收益求和的乘积，与10日延迟值的差排名，捕捉短期开盘价与收益的联动变化
    类型：量价动量因子
    公式：
        (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    用法：
        - 用处：识别开盘价与收益的短期联动相对于历史的变化
        - 适用场景：中期趋势中（10-15日）的动量延续/反转
        - 数值含义：值>0代表当前联动强于历史（多头），值<0代表弱于历史（空头）
    """
    x1 = (ts_sum(Open, 5) * ts_sum(returns, 5))
    x2 = delay((ts_sum(Open, 5) * ts_sum(returns, 5)), 10)
    alpha = -1 * rank(x1-x2)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha9 ------------------------------
def alpha9(close):
    """
    逻辑：收盘价1日差分的5日极值筛选，反转极端涨跌的信号
    类型：价格反转因子
    公式：
        ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    用法：
        - 用处：识别短期价格涨跌的极端值，反转信号
        - 适用场景：震荡市中的超买超卖修复
        - 数值含义：值>0代表极端下跌后反转（多头），值<0代表极端上涨后反转（空头）
    """
    delta_close = delta(close, 1)
    x1 = ts_min(delta_close, 5) > 0
    x2 = ts_max(delta_close, 5) < 0
    alpha = -1 * delta_close
    alpha[x1 | x2] = delta_close
    return alpha.fillna(value = 0)

# ------------------------------ Alpha10 ------------------------------
def alpha10(close):
    """
    逻辑：收盘价1日差分的4日极值筛选，排名后捕捉反转信号
    类型：价格反转因子
    公式：
        rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    用法：
        - 用处：更短周期（4日）的价格极端值反转，信号更灵敏
        - 适用场景：短期（1周内）超买超卖的修复行情
        - 数值含义：值越高代表反转多头信号越强，值越低代表反转空头信号越强
    """
    delta_close = delta(close, 1)
    x1 = ts_min(delta_close, 4) > 0
    x2 = ts_max(delta_close, 4) < 0
    x = -1 * delta_close
    x[x1 | x2] = delta_close
    alpha = rank(x)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha11 ------------------------------
def alpha11(vwap,close,volume):
    """
    逻辑：VWAP与收盘价差值的3日极值排名，结合成交量3日差分排名，捕捉量价与成交均价的联动
    类型：成交均价联动因子
    公式：
       # Alpha#11	 ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
    用法：
        - 用处：识别成交均价与价格、成交量的联动程度
        - 适用场景：主力资金成交密集区的价格判断
        - 数值含义：值>0代表成交均价偏离+成交量变动正向（多头），值<0代表反向（空头）
    """
    x1 = rank(ts_max((vwap - close), 3))
    x2 = rank(ts_min((vwap - close), 3))
    x3 = rank(delta(volume, 3))
    alpha = (x1 + x2) * x3
    return alpha.fillna(value = 0)

# ------------------------------ Alpha12 ------------------------------
def alpha12(volume,close):
    """
    逻辑：成交量1日差分的方向，乘以收盘价1日差分的反向，捕捉量价反向信号
    类型：量价反转因子
    公式：
        alpha = sign(delta(volume,1)) * (-1 * delta(close,1))
    用法：
        - 用处：识别量价反向变动（量涨价跌/量跌价涨）的信号
        - 适用场景：短期资金博弈激烈的行情，反转概率高
        - 数值含义：值>0代表量涨价跌（多头反转），值<0代表量跌价涨（空头反转）
    """
    alpha = np.sign(delta(volume, 1)) * (-1 * delta(close, 1))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha13 ------------------------------
def alpha13(volume,close):
    """
    逻辑：收盘价排名与成交量排名的5日协方差排名，捕捉量价的协变关系
    类型：量价协变因子
    公式：
        alpha = -1 * rank(covariance(rank(close), rank(volume),5))
    用法：
        - 用处：判断价格与成交量的协变程度，识别资金与价格的同步性
        - 适用场景：趋势行情中量价是否同步验证
        - 数值含义：值>0代表协方差为负（量价反向），值<0代表协方差为正（量价同向）
    """
    alpha = -1 * rank(covariance(rank(close), rank(volume), 5))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha14 ------------------------------
def alpha14(Open,volume,returns):
    """
    逻辑：开盘价与成交量的10日相关性，结合收益3日差分排名，捕捉开盘量价与收益的联动
    类型：开盘量价因子
    公式：
       ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    用法：
        - 用处：识别开盘阶段量价关系对收益的影响
        - 适用场景：早盘资金异动对当日收益的预判
        - 数值含义：值>0代表开盘量价正相关+收益差分偏多（多头），值<0代表反向（空头）
    """
    x1 = correlation(Open, volume, 10).replace([-np.inf, np.inf], 0).fillna(value=0)
    x2 = -1 * rank(delta(returns, 3))
    alpha = x1 * x2
    return alpha.fillna(value = 0)

# ------------------------------ Alpha15 ------------------------------
def alpha15(high,volume):
    """
    逻辑：最高价排名与成交量排名的3日相关性，求和3日排名后取反，捕捉高价与量能的联动趋势
    类型：高价量能因子
    公式：
        (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    用法：
        - 用处：判断最高价与成交量的短期联动趋势
        - 适用场景：高价股的量能验证（量价是否同步）
        - 数值含义：值>0代表联动趋势向下（空头），值<0代表联动趋势向上（多头）
    """
    x1 = correlation(rank(high), rank(volume), 3).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * ts_sum(rank(x1), 3)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha16 ------------------------------
def alpha16(high,volume):
    """
    逻辑：最高价排名与成交量排名的5日协方差排名，捕捉高价与量能的协变关系
    类型：高价量能因子
    公式：
        -1 * rank(covariance(rank(high), rank(volume),5))
    用法：
        - 用处：识别高价股成交量与价格的协变程度
        - 适用场景：高价股趋势延续性判断
        - 数值含义：值>0代表协方差负（量价反向），值<0代表协方差正（量价同向）
    """
    alpha = -1 * rank(covariance(rank(high), rank(volume), 5))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha17 ------------------------------
def alpha17(volume,close):
    """
    逻辑：结合20日均量，通过收盘价排名、收盘价二阶差分排名、量能相对均量排名的乘积，捕捉量价的多层级联动
    类型：多维度量价因子
    公式：
        (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    用法：
        - 用处：从价格趋势、价格加速度、量能相对水平三个维度捕捉信号
        - 适用场景：中期趋势（20日）中量价结构健康度判断
        - 数值含义：值>0代表多维度共振多头，值<0代表共振空头
    """
    adv20 = sma(volume, 20)
    x1 = rank(ts_rank(close, 10))
    x2 = rank(delta(delta(close, 1), 1))
    x3 = rank(ts_rank((volume / adv20), 5))
    alpha = -1 * (x1 * x2 * x3)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha18 ------------------------------
def alpha18(close,Open):
    """
    逻辑：收盘价与开盘价的10日相关性，结合开盘收盘价差的波动率，捕捉价格日内波动与相关性的联动
    类型：日内波动因子
    公式：
        (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    用法：
        - 用处：识别日内价格波动与日间相关性的综合信号
        - 适用场景：日内交易为主的行情，判断价格波动的持续性
        - 数值含义：值>0代表综合波动信号偏空，值<0代表偏多
    """
    x = correlation(close, Open, 10).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * (rank((stddev(abs((close - Open)), 5) + (close - Open)) + x))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha19 ------------------------------
def alpha19(close,returns):
    """
    逻辑：7日收盘价变动的方向，结合250日收益求和的排名，捕捉长期收益对短期价格方向的修正
    类型：长短期动量因子
    公式：
        ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    用法：
        - 用处：结合长期收益趋势修正短期价格方向信号
        - 适用场景：跨年/长期趋势中的短期波段操作
        - 数值含义：值>0代表长期多头+短期反转（多头），值<0代表长期空头+短期反转（空头）
    """
    x1 = (-1 * np.sign((close - delay(close, 7)) + delta(close, 7)))
    x2 = (1 + rank(1 + ts_sum(returns, 250)))
    alpha = x1 * x2
    return alpha.fillna(value = 0)

# ------------------------------ Alpha20 ------------------------------
def alpha20(Open,high,close,low):
    """
    逻辑：开盘价与前一日高低收盘价的差值排名乘积，捕捉开盘价相对于昨日价格区间的偏离
    类型：开盘价偏离因子
    公式：
        alpha = -1 * (rank(Open - delay(high,1)) * rank(Open - delay(close,1)) * rank(Open - delay(low,1)))
    用法：
        - 用处：识别开盘价突破昨日价格区间的程度
        - 适用场景：跳空开盘后的趋势判断
        - 数值含义：值>0代表开盘价偏离昨日区间偏多，值<0代表偏空
    """
    alpha = -1 * (rank(Open - delay(high, 1)) * rank(Open - delay(close, 1)) * rank(Open - delay(low, 1)))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha21 ------------------------------
def alpha21(volume,close):
    """
    逻辑：通过8日收盘价均线±波动率与2日均线的关系，结合量能相对20日均量的水平，生成多空信号
    类型：量价均线因子
    公式：
        ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    用法：
        - 用处：均线系统突破+量能验证的综合信号
        - 适用场景：均线趋势反转的确认
        - 数值含义：值=1代表均线+量能健康（多头），值=-1代表均线+量能走弱（空头）
    """
    x1 = sma(close, 8) + stddev(close, 8) < sma(close, 2)
    x2 = sma(close, 8) - stddev(close, 8) > sma(close, 2)
    x3 = sma(volume, 20) / volume < 1
    alpha = pd.DataFrame(np.ones_like(close), index = close.index,columns = close.columns)
    alpha[x1 | x3] = -1 * alpha
    return alpha

# ------------------------------ Alpha22 ------------------------------
def alpha22(high,volume,close):
    """
    逻辑：最高价与成交量的5日相关性差分，结合收盘价20日波动率排名，捕捉高价量能相关性的变化
    类型：高价量能趋势因子
    公式：
        (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    用法：
        - 用处：识别高价与量能相关性的变化幅度，结合价格波动放大信号
        - 适用场景：高价股趋势加速/减速的判断
        - 数值含义：值>0代表相关性下降+波动大（空头），值<0代表相关性上升+波动大（多头）
    """
    x = correlation(high, volume, 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * delta(x, 5) * rank(stddev(close, 20))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha23 ------------------------------
def alpha23(high,close):
    """
    逻辑：20日最高价均线低于当前最高价时，取最高价2日差分的反向，捕捉高价突破后的回调信号
    类型：高价突破因子
    公式：
        (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    用法：
        - 用处：识别高价突破长期均线后的短期回调风险
        - 适用场景：突破型行情的回踩确认
        - 数值含义：值<0代表突破后回调（空头），值=0代表未突破（无信号）
    """
    x = sma(high, 20) < high
    alpha = pd.DataFrame(np.zeros_like(close),index = close.index,columns = close.columns)
    alpha[x] = -1 * delta(high, 2).fillna(value = 0)
    return alpha

# ------------------------------ Alpha24 ------------------------------
def alpha24(close):
    """
    逻辑：100日收盘价均线的100日变化率筛选，区分短期价格变动与长期底部的关系
    类型：长短期价格因子
    公式：
        ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    用法：
        - 用处：长期横盘（变化率<5%）时捕捉价格相对底部的偏离，否则看短期3日变动
        - 适用场景：长期横盘股的突破/破位判断
        - 数值含义：值>0代表长期横盘+价格离底部远（多头），值<0代表短期下跌（空头）
    """
    x = delta(sma(close, 100), 100) / delay(close, 100) <= 0.05
    alpha = -1 * delta(close, 3)
    alpha[x] = -1 * (close - ts_min(close, 100))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha25 ------------------------------
def alpha25(volume,returns,vwap,high,close):
    """
    逻辑：结合20日均量，通过收益、均量、VWAP、高低收盘价差的乘积排名，捕捉量价与成交均价的综合信号
    类型：综合量价因子
    公式：
        rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    用法：
        - 用处：多维度量价指标的共振信号，识别资金成交与价格的关系
        - 适用场景：主力资金成交活跃的行情
        - 数值含义：值越高代表综合信号偏多，值越低代表偏空
    """
    adv20 = sma(volume, 20)
    alpha = rank((((-1 * returns) * adv20) * vwap) * (high - close))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha26 ------------------------------
def alpha26(volume,high):
    """
    逻辑：成交量与最高价5日时序排名的5日相关性，取3日最大值反向，捕捉量价排名相关性的极值
    类型：量价排名因子
    公式：
        (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    用法：
        - 用处：识别量价排名相关性的短期最大值，判断联动的极值
        - 适用场景：短期量价联动的反转点判断
        - 数值含义：值>0代表相关性极值偏空，值<0代表相关性极值偏多
    """
    x = correlation(ts_rank(volume, 5), ts_rank(high, 5), 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * ts_max(x, 3)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha27 ------------------------------
def alpha27(volume,vwap):
    """
    逻辑：成交量与VWAP排名的6日相关性2日均值排名，阈值划分多空信号
    类型：成交均价相关因子
    公式：
        ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    用法：
        - 用处：通过相关性均值的排名阈值生成明确的多空信号
        - 适用场景：量化策略的多空持仓判断
        - 数值含义：值=1代表相关性低（多头），值=-1代表相关性高（空头）
    """
    alpha = rank((sma(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))
    alpha[alpha > 0.5] = -1
    alpha[alpha <= 0.5] = 1
    return alpha.fillna(value = 0)  

# ------------------------------ Alpha28 ------------------------------
def alpha28(volume,high,low,close):
    """
    逻辑：20日均量与最低价的5日相关性，结合平均价格与收盘价的差值标准化，捕捉量能与低价的关联
    类型：量价标准化因子
    公式：
        scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    用法：
        - 用处：标准化后的量能-低价相关性与价格偏离信号
        - 适用场景：不同市值/价格个股的横向比较
        - 数值含义：值>0代表标准化后信号偏多，值<0代表偏空，绝对值越大信号越强
    """
    adv20 = sma(volume, 20)
    x = correlation(adv20, low, 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = scale(((x + ((high + low) / 2)) - close))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha29 ------------------------------
def alpha29(close,returns):
    """
    逻辑：多层级排名/标准化后的收盘价差分求和的5日最小值，结合收益延迟排名，捕捉极端信号
    类型：多层级量价因子
    公式：
        (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    用法：
        - 用处：通过多层级变换放大极端价格信号，结合延迟收益排名
        - 适用场景：极端行情中的超买超卖识别
        - 数值含义：值>0代表多层级信号共振多头，值<0代表共振空头
    """
    x1 = ts_min(rank(rank(scale(np.log(ts_sum(rank(rank(-1 * rank(delta((close - 1), 5)))), 2))))), 5)
    x2 = ts_rank(delay((-1 * returns), 6), 5)
    alpha = x1 + x2
    return alpha.fillna(value = 0) 

# ------------------------------ Alpha30 ------------------------------
def alpha30(close,volume):
    """
    逻辑：收盘价3日差分方向的求和，结合成交量5/20日求和的比值，捕捉价格方向与量能趋势的联动
    类型：量价趋势因子
    公式：
        (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    用法：
        - 用处：识别价格短期方向的一致性（3日）与量能相对水平的关系
        - 适用场景：短期趋势（3-5日）的量能验证
        - 数值含义：值>0代表价格方向一致+量能占比高（多头），值<0代表方向分歧+量能占比低（空头）
    """
    delta_close = delta(close, 1)
    x = np.sign(delta_close) + np.sign(delay(delta_close, 1)) + np.sign(delay(delta_close, 2))
    alpha = ((1.0 - rank(x)) * ts_sum(volume, 5)) / ts_sum(volume, 20)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha31 ------------------------------
def alpha31(close,low,volume):
    """
    逻辑：结合20日均量，多层级排名后的收盘价差分衰减，结合收盘价短期差分、量能-低价相关性，捕捉多维度信号
    类型：多维度量价因子
    公式：
        ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    用法：
        - 用处：从长期价格差分、短期价格差分、量能-低价关联三个维度捕捉信号
        - 适用场景：中期（10-20日）趋势中的多维度共振
        - 数值含义：值>0代表多维度共振多头，值<0代表共振空头
    """
    adv20 = sma(volume,20)
    x1 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10))))
    x2 = rank((-1 * delta(close, 3)))
    x3 = np.sign(scale(correlation(adv20, low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)))
    alpha = x1 + x2 + x3
    return alpha.fillna(value = 0)

# ------------------------------ Alpha32 ------------------------------
def alpha32(close,vwap):
    """
    逻辑：VWAP与5日延迟收盘价的230日相关性，结合7日收盘价均线偏离，捕捉长期成交均价与价格的关联
    类型：长期成交均价因子
    公式：
        (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    用法：
        - 用处：长期（230日）成交均价关联+短期均线偏离的综合信号
        - 适用场景：中长期（季度级）趋势中的短期波段
        - 数值含义：值>0代表长期关联+短期均线偏离偏多，值<0代表偏空
    """
    x = correlation(vwap, delay(close, 5),230).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = scale(((sma(close, 7)) - close)) + 20 * scale(x)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha33 ------------------------------
def alpha33(Open,close):
    """
    逻辑：开盘价/收盘价减1的排名，捕捉开盘价相对于收盘价的偏离程度
    类型：开盘收盘偏离因子
    公式：
        alpha = rank(-1 + Open/close)
    用法：
        - 用处：识别开盘价相对于收盘价的溢价/折价程度
        - 适用场景：日内开盘定价合理性判断
        - 数值含义：值>0代表开盘价溢价（多头），值<0代表开盘价折价（空头）
    """
    alpha = rank(-1 + (Open / close))
    return alpha

# ------------------------------ Alpha34 ------------------------------
def alpha34(close,returns):
    """
    逻辑：收益2/5日波动率比值，结合收盘价1日差分排名，捕捉收益波动结构的变化
    类型：收益波动因子
    公式：
        rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    用法：
        - 用处：识别短期收益波动率的相对变化，结合价格变动
        - 适用场景：波动市中的收益结构判断
        - 数值含义：值>0代表波动结构+价格变动偏多，值<0代表偏空
    """
    x = (stddev(returns, 2) / stddev(returns, 5)).fillna(value = 0)
    alpha = rank(2 - rank(x) - rank(delta(close, 1)))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha35 ------------------------------
def alpha35(volume,close,high,low,returns):
    """
    逻辑：成交量32日排名、价格区间16日排名、收益32日排名的乘积，捕捉量价收益的长期相对位置
    类型：长周期量价收益因子
    公式：
        ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    用法：
        - 用处：从量能、价格区间、收益三个维度捕捉长期（32日）相对位置
        - 适用场景：月度级趋势中的多空判断
        - 数值含义：值>0代表长周期多维度共振多头，值<0代表共振空头
    """
    x1 = ts_rank(volume, 32)
    x2 = 1 - ts_rank(close + high - low, 16)
    x3 = 1 - ts_rank(returns, 32)
    alpha = (x1 * x2 * x3).fillna(value = 0)
    return alpha

# ------------------------------ Alpha36 ------------------------------
def alpha36(Open,close,volume,returns,vwap):
    """
    逻辑：多维度加权的量价相关因子，包含开盘收盘价差与量能相关性、开盘收盘价差、延迟收益排名、VWAP与均量相关性、长期均线偏离
    类型：加权综合量价因子
    公式：
        (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    用法：
        - 用处：多维度量价指标加权求和，突出核心信号（开盘价相关）
        - 适用场景：综合型量化策略的核心因子
        - 数值含义：值越高代表综合加权信号偏多，值越低代表偏空
    """
    adv20 = sma(volume, 20)
    x1 = 2.21 * rank(correlation((close - Open), delay(volume, 1), 15))
    x2 = 0.7 * rank((Open- close))
    x3 = 0.73 * rank(ts_rank(delay((-1 * returns), 6), 5))
    x4 = rank(abs(correlation(vwap,adv20, 6)))
    x5 = 0.6 * rank((sma(close, 200) - Open) * (close - Open))
    alpha = x1 + x2 + x3 + x4 + x5
    return alpha.fillna(value = 0)

# ------------------------------ Alpha37 ------------------------------
def alpha37(Open,close):
    """
    逻辑：延迟1日的开盘收盘价差与收盘价的200日相关性排名，结合开盘收盘价差排名，捕捉长期价差相关性
    类型：长期价差相关因子
    公式：
        alpha = rank(correlation(delay(Open-close,1), close,200)) + rank(Open-close)
    用法：
        - 用处：识别长期（200日）开盘收盘价差与收盘价的关联，结合短期价差
        - 适用场景：长期趋势中日内价差的持续性判断
        - 数值含义：值>0代表长期+短期价差信号偏多，值<0代表偏空
    """
    alpha = rank(correlation(delay(Open - close, 1), close, 200)) + rank(Open - close)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha38 ------------------------------
def alpha38(close,Open):
    """
    逻辑：开盘价10日时序排名，结合收盘价/开盘价的排名，捕捉开盘价排名与价格比值的联动
    类型：开盘价排名因子
    公式：
        x = close/Open (填充极值/空值)
        alpha = -1 * rank(ts_rank(Open,10)) * rank(x)
    用法：
        - 用处：识别开盘价相对排名与价格比值的反向联动
        - 适用场景：开盘价排名异动对价格的影响
        - 数值含义：值>0代表开盘价排名高+价格比值低（空头），值<0代表开盘价排名低+价格比值高（多头）
    """
    x = (close / Open).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * rank(ts_rank(Open, 10)) * rank(x)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha39 ------------------------------
def alpha39(volume,close,returns):
    """
    逻辑：结合20日均量，收盘价7日差分排名反向，结合量能相对均量的线性衰减排名，再结合长期收益求和排名
    类型：量价衰减因子
    公式：
        ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    用法：
        - 用处：量能衰减+价格变动+长期收益的综合信号
        - 适用场景：量能衰减阶段的长期收益修正
        - 数值含义：值>0代表量能衰减+价格上涨+长期收益高（多头），值<0代表反向（空头）
    """
    adv20 = sma(volume, 20)
    x = -1 * rank(delta(close, 7)) * (1 - rank(decay_linear((volume / adv20), 9)))
    alpha = x *(1 + rank(ts_sum(returns, 250)))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha40 ------------------------------
def alpha40(high,volume):
    """
    逻辑：最高价10日波动率排名，结合最高价与成交量的10日相关性，捕捉高价波动与量能的联动
    类型：高价波动因子
    公式：
        alpha = -1 * rank(stddev(high,10)) * correlation(high,volume,10)
    用法：
        - 用处：识别高价波动率与量能相关性的反向联动
        - 适用场景：高价股波动与量能的验证
        - 数值含义：值>0代表高价波动大+量能负相关（空头），值<0代表高价波动大+量能正相关（多头）
    """
    alpha = -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha41 ------------------------------
def alpha41(high,low,vwap):
    """
    逻辑：高低价平方根均值与VWAP的差值，捕捉价格中枢与成交均价的偏离
    类型：价格中枢因子
    公式：
        alpha = sqrt(high*low) - vwap
    用法：
        - 用处：识别价格中枢（高低价均值）与成交均价的偏离程度
        - 适用场景：价格中枢与资金成交成本的背离判断
        - 数值含义：值>0代表价格中枢高于成交均价（多头），值<0代表低于成交均价（空头）
    """
    alpha = pow((high * low),0.5) - vwap
    return alpha

# ------------------------------ Alpha42 ------------------------------
def alpha42(vwap,close):
    """
    逻辑：VWAP与收盘价差值的排名，除以VWAP与收盘价和的排名，捕捉成交均价与价格的相对偏离
    类型：成交均价相对偏离因子
    公式：
        alpha = rank(vwap-close) / rank(vwap+close)
    用法：
        - 用处：相对比值形式的成交均价偏离，消除绝对价格的影响
        - 适用场景：不同价格水平个股的横向比较
        - 数值含义：值>0代表VWAP相对收盘价偏高（多头），值<0代表偏低（空头）
    """
    alpha = rank((vwap - close)) / rank((vwap + close))
    return alpha

# ------------------------------ Alpha43 ------------------------------
def alpha43(volume,close):
    """
    逻辑：结合20日均量，量能相对均量的20日排名，结合收盘价7日差分反向的8日排名，捕捉量能与价格的联动
    类型：量价相对排名因子
    公式：
        (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    用法：
        - 用处：量能相对水平+价格短期变动的排名乘积
        - 适用场景：量能相对均量异动+价格异动的共振
        - 数值含义：值>0代表量能高+价格下跌（空头），值<0代表量能低+价格上涨（多头）
    """
    adv20 = sma(volume, 20)
    alpha = ts_rank(volume / adv20, 20) * ts_rank((-1 * delta(close, 7)), 8)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha44 ------------------------------
def alpha44(high,volume):
    """
    逻辑：最高价与成交量排名的5日相关性反向，捕捉高价与量能排名的关联
    类型：高价量能排名因子
    公式：
        alpha = -1 * correlation(high, rank(volume),5) (填充极值/空值)
    用法：
        - 用处：识别最高价与成交量排名的相关性，消除量能绝对大小的影响
        - 适用场景：不同市值个股的高价量能关联比较
        - 数值含义：值>0代表相关性负（空头），值<0代表相关性正（多头）
    """
    alpha = -1 *correlation(high, rank(volume), 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    return alpha

# ------------------------------ Alpha45 ------------------------------
def alpha45(close,volume):
    """
    逻辑：延迟5日收盘价的20日均线排名，结合收盘价与成交量的2日相关性，再结合收盘价5/20日求和的2日相关性
    类型：多周期价格相关因子
    公式：
        (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
    用法：
        - 用处：多周期收盘价相关性+量价相关性的综合信号
        - 适用场景：不同周期价格趋势的一致性判断
        - 数值含义：值>0代表多周期信号共振空头，值<0代表共振多头
    """
    x = correlation(close, volume, 2).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * (rank(sma(delay(close, 5), 20)) * x * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha46 ------------------------------
def alpha46(close):
    """
    逻辑：收盘价10/20日差分的斜率比较，筛选不同斜率区间的价格1日差分信号
    类型：价格斜率因子
    公式：
        ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
    用法：
        - 用处：通过价格斜率判断趋势加速度，生成反转/延续信号
        - 适用场景：趋势加速/减速的拐点判断
        - 数值含义：值=1代表斜率负（多头反转），值=-1代表斜率>0.25（空头），其他为价格差分反向
    """
    x = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    alpha = (-1 * (close - delay(close, 1)))
    alpha[x < 0] = 1
    alpha[x > 0.25] = -1
    return alpha.fillna(value = 0)

# ------------------------------ Alpha47 ------------------------------
def alpha47(volume,close,high,vwap):
    """
    逻辑：结合20日均量，收盘价倒数*成交量/均量，结合高价偏离*高价排名/高价均线，再减去VWAP5日差分排名
    类型：复杂量价综合因子
    公式：
        ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    用法：
        - 用处：多维度量价指标的非线性组合，捕捉资金成交与价格结构的信号
        - 适用场景：主力资金操盘的复杂行情
        - 数值含义：值>0代表综合信号偏多，值<0代表偏空
    """
    adv20 = sma(volume, 20)
    alpha = (((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / sma(high, 5)) - rank((vwap - delay(vwap, 5))))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha48 ------------------------------
def alpha48(close,ind):
    """
    逻辑：收盘价差分的250日相关性*收盘价差分/收盘价，除以收盘价差分平方和，行业中性化后捕捉长期价格相关的信号
    类型：行业中性价格相关因子
    公式：
        (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    用法：
        - 用处：消除行业影响后，捕捉长期价格自相关的信号
        - 适用场景：跨行业的价格自相关比较
        - 数值含义：值>0代表行业中性后价格自相关偏多，值<0代表偏空
    """
    r1 = (correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close
    r2 = ts_sum((pow((delta(close, 1) / delay(close, 1)),2)), 250)
    alpha = IndNeutralize(r1, ind) / r2
    return alpha.fillna(value = 0)

# ------------------------------ Alpha49 ------------------------------
def alpha49(close):
    """
    逻辑：收盘价10/20日差分的斜率筛选，斜率<-0.1时反转价格1日差分信号
    类型：价格斜率反转因子
    公式：
        (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    用法：
        - 用处：更严格的斜率阈值（<-0.1）触发反转信号
        - 适用场景：趋势快速下跌后的反转判断
        - 数值含义：值=1代表斜率<-0.1（多头反转），其他为价格差分反向
    """
    x = (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))
    alpha = (-1 * delta(close,1))
    alpha[x < -0.1] = 1
    return alpha.fillna(value = 0)

# ------------------------------ Alpha50 ------------------------------
def alpha50(volume,vwap):
    """
    逻辑：成交量与VWAP排名的5日相关性排名的5日最大值反向，捕捉成交均价与量能相关性的极值
    类型：成交均价量能相关因子
    公式：
        alpha = -1 * ts_max(rank(correlation(rank(volume), rank(vwap),5)),5)
    用法：
        - 用处：识别成交均价与量能排名相关性的短期最大值
        - 适用场景：量价相关性极值后的反转判断
        - 数值含义：值>0代表相关性极值偏空，值<0代表相关性极值偏多
    """
    alpha = -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha51 ------------------------------
def alpha51(close):
    """
    逻辑：收盘价10/20日差分的斜率筛选，斜率<-0.05时反转价格1日差分信号
    类型：价格斜率反转因子
    公式：
        (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    用法：
        - 用处：中等斜率阈值（<-0.05）触发反转，比alpha49更宽松
        - 适用场景：趋势温和下跌后的反转判断
        - 数值含义：值=1代表斜率<-0.05（多头反转），其他为价格差分反向
    """
    inner = (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))
    alpha = (-1 * delta(close,1))
    alpha[inner < -0.05] = 1
    return alpha.fillna(value = 0)

# ------------------------------ Alpha52 ------------------------------
def alpha52(returns,volume,low):
    """
    逻辑：收益240/20日求和的差值排名，结合最低价5日最小值的5日差分反向，再结合成交量5日排名
    类型：长周期收益+低价因子
    公式：
        ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    用法：
        - 用处：长期收益趋势+短期低价变动+量能的综合信号
        - 适用场景：长期收益分化行情中的低价股机会
        - 数值含义：值>0代表长期收益高+低价上涨+量能高（多头），值<0代表反向（空头）
    """
    x = rank(((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220))
    alpha = -1 * delta(ts_min(low, 5), 5) * x * ts_rank(volume, 5)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha53 ------------------------------
def alpha53(close,high,low):
    """
    逻辑：价格在高低价区间的相对位置9日差分反向，捕捉价格相对位置的变化
    类型：价格位置因子
    公式：
        (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    用法：
        - 用处：识别价格在高低价区间的相对位置变化，判断趋势强弱
        - 适用场景：区间震荡/突破行情中的价格位置判断
        - 数值含义：值>0代表位置变化偏空，值<0代表位置变化偏多
    """
    alpha = -1 * delta((((close - low) - (high - close)) / (close - low).replace(0, 0.0001)), 9)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha54 ------------------------------
def alpha54(Open,close,high,low):
    """
    逻辑：开盘价5次方*（最低价-收盘价），除以（最低价-最高价）*收盘价5次方，捕捉日内价格结构的非线性信号
    类型：日内非线性价格因子
    公式：
        ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    用法：
        - 用处：非线性放大开盘价/收盘价对日内价格结构的影响
        - 适用场景：日内价格结构极端的行情（如跳空、宽幅震荡）
        - 数值含义：值>0代表日内价格结构偏多，值<0代表偏空，绝对值越大信号越强
    """
    x = (low - high).replace(0, -0.0001)
    alpha = -1 * (low - close) * (Open ** 5) / (x * (close ** 5))
    return alpha

# ------------------------------ Alpha55 ------------------------------
def alpha55(high,low,close,volume):
    """
    逻辑：价格在12日高低价区间的相对位置排名，与成交量排名的6日相关性反向，捕捉价格位置与量能的关联
    类型：价格位置量能因子
    公式：
        (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    用法：
        - 用处：识别价格在中期（12日）区间的位置与量能的相关性
        - 适用场景：中期区间震荡行情中的量价验证
        - 数值含义：值>0代表相关性负（空头），值<0代表相关性正（多头）
    """
    x = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)).replace(0, 0.0001)
    alpha = -1 * correlation(rank(x), rank(volume), 6).replace([-np.inf, np.inf], 0).fillna(value=0)
    return alpha

# ------------------------------ Alpha56 ------------------------------
def alpha56(returns,cap):
    """
    逻辑：收益10日均值/收益2日均值的3日均值排名，结合收益*市值排名，捕捉收益与市值的联动
    类型：收益市值因子
    公式：
        alpha = -1 * rank(sma(returns,10)/sma(sma(returns,2),3)) * rank(returns*cap)
    用法：
        - 用处：识别收益趋势与市值的联动关系，区分大小市值股的收益表现
        - 适用场景：大小市值风格切换的行情
        - 数值含义：值>0代表收益趋势+市值联动偏空，值<0代表偏多
    """
    alpha = 0 - (1 * (rank((sma(returns, 10) / sma(sma(returns, 2), 3))) * rank((returns * cap))))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha57 ------------------------------
def alpha57(close,vwap):
    """
    逻辑：收盘价与VWAP的差值，除以收盘价30日最大值排名的线性衰减，捕捉成交均价偏离的衰减信号
    类型：成交均价衰减因子
    公式：
        alpha = -1 * (close - vwap) / decay_linear(rank(ts_argmax(close,30)),2)
    用法：
        - 用处：成交均价偏离随价格极值排名衰减的信号
        - 适用场景：价格极值后的成交均价偏离修复
        - 数值含义：值>0代表偏离修复偏多，值<0代表偏离修复偏空
    """
    alpha = 0 - 1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha58 ------------------------------
def alpha58(vwap,volume,ind):
    """
    逻辑：VWAP行业中性化后，与成交量4日相关性的线性衰减，8日时序排名6日排名反向，捕捉行业中性的成交均价量能信号
    类型：行业中性成交均价因子
    公式：
        (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))
    用法：
        - 用处：消除行业影响后，捕捉成交均价与量能的衰减相关性
        - 适用场景：跨行业的成交均价量能比较
        - 数值含义：值>0代表行业中性后信号偏空，值<0代表偏多
    """
    x = IndNeutralize(vwap, ind)
    alpha = -1 * ts_rank(decay_linear(correlation(x, volume, 4), 8), 6)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha59 ------------------------------
def alpha59(vwap,volume,ind):
    """
    逻辑：VWAP加权自组合后行业中性化，与成交量4日相关性的线性衰减，16日时序排名8日排名反向
    类型：行业中性加权成交均价因子
    公式：
        (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    用法：
        - 用处：加权VWAP消除行业影响后，捕捉更长周期（16日）的衰减相关性
        - 适用场景：中期跨行业的成交均价量能趋势
        - 数值含义：值>0代表行业中性后信号偏空，值<0代表偏多
    """
    x = IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), ind)
    alpha = -1 * ts_rank(decay_linear(correlation(x, volume, 4), 16), 8)
    return alpha.fillna(value = 0)

# ------------------------------ Alpha60 ------------------------------
def alpha60(close,high,low,volume):
    """
    逻辑：价格在高低价区间的相对位置*成交量，标准化后与收盘价10日最大值排名标准化的差值，捕捉量价位置的综合信号
    类型：量价位置综合因子
    公式：
        (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
    用法：
        - 用处：量价位置的标准化信号，消除量价绝对大小的影响
        - 适用场景：不同量价水平个股的横向比较
        - 数值含义：值>0代表量价位置信号偏多，值<0代表偏空
    """
    x = ((close - low) - (high - close)) * volume / (high - low).replace(0, 0.0001)
    alpha = - ((2 * scale(rank(x))) - scale(rank(ts_argmax(close, 10))))
    return alpha.fillna(value = 0)

# ------------------------------ Alpha61 ------------------------------
def alpha61(volume,vwap):
    """
    逻辑：VWAP与16日最小值的排名，小于VWAP与180日均量18日相关性排名，生成布尔型信号
    类型：成交均价均量相关因子
    公式：
        (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    用法：
        - 用处：成交均价低位与均量相关性的比较信号
        - 适用场景：成交均价低位的量能验证
        - 数值含义：True代表VWAP低位+相关性高（多头），False代表反之（空头）
    """
    adv180 = sma(volume, 180)
    alpha = rank((vwap - ts_min(vwap, 16))) < rank(correlation(vwap, adv180, 18))
    return alpha

# ------------------------------ Alpha62 ------------------------------
def alpha62(volume,high,low,Open,vwap):
    """
    逻辑：VWAP与20日均量22日求和的10日相关性排名，小于开盘价排名和与价格中枢排名和的比较排名，生成反向布尔信号
    类型：成交均价量能比较因子
    公式：
        ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    用法：
        - 用处：成交均价量能相关性与开盘价-价格中枢结构的比较
        - 适用场景：开盘价结构与量能的联动验证
        - 数值含义：-1代表比较成立（多头），0代表不成立（空头）
    """
    adv20 = sma(volume, 20)
    x1 = rank(correlation(vwap, ts_sum(adv20, 22), 10))
    x2 =  rank(((rank(Open) + rank(Open)) < (rank(((high + low) / 2)) + rank(high))))
    alpha = x1 < x2
    return alpha*-1

def alpha63(volume, close, vwap, Open, ind):
    """
    因子名称：Alpha63
    逻辑：对收盘价做行业中性后取2日差分，做线性衰减并排名；
          再对加权VWAP-开盘价与180日均量总和做相关性，线性衰减并排名；
          两者做差后取反，得到行业中性的量价异动反转信号。
    类型：行业中性量价因子 | 短期异动反转因子
    公式：
        ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)
    用法：
        用处：剔除行业β后，捕捉个股独立的短期价量异动，用于行业内选股、对冲组合；
        适用场景：震荡市/结构市，行业中性策略，剔除行业整体涨跌影响；
        数值含义：
            因子值越大 → 看多信号越强（短期价量偏弱，预期反转向上）
            因子值越小 → 看空信号越强（短期价量偏强，预期反转向下）
    """
    # 180日均量，填充空值避免计算出错
    adv180 = sma(volume, 180).fillna(value=0)
    
    # 第一个信号：收盘价行业中性 → 2日差分 → 8期线性衰减 → 排名
    r1 = rank(decay_linear(delta(IndNeutralize(close, ind), 2), 8))
    
    # 加权价格：vwap与Open固定权重组合
    weight = 0.318108
    weighted_price = vwap * weight + Open * (1 - weight)
    
    # 第二个信号：加权价与180日均量37日和的14日相关 → 12期线性衰减 → 排名
    corr_part = correlation(weighted_price, ts_sum(adv180, 37), 14).replace([-np.inf, np.inf], 0).fillna(0)
    r2 = rank(decay_linear(corr_part, 12))
    
    # 最终因子：差值取反
    alpha = -1 * (r1 - r2)
    
    return alpha.fillna(value=0)

# ------------------------------ Alpha64 ------------------------------
def alpha64(high,low,Open,volume,vwap):
    """
    逻辑：加权开盘低价组合13日和与120日均量13日和的17日相关性排名，小于加权均价3日差分排名，布尔信号
    类型：行业无关量价比较因子
    公式：
        ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    用法：
        - 用处：加权价格与长期均量的相关性强弱比较
        - 适用场景：中长期量价趋势判断
        - 数值含义：True=信号偏多，False=偏空；返回-1/0
    """
    adv120 = sma(volume, 120)
    w = 0.178404
    part1 = ts_sum(Open * w + low * (1 - w), 13)
    part2 = ts_sum(adv120, 13)
    x1 = rank(correlation(part1, part2, 17))
    x2 = rank(delta(((high + low) / 2 * w) + vwap * (1 - w), 3))
    alpha = x1 < x2
    return alpha * -1

# ------------------------------ Alpha65 ------------------------------
def alpha65(volume,vwap,Open):
    """
    逻辑：加权开盘VWAP与60日均量9日和的6日相关性排名，小于开盘价14日最小值偏离排名，布尔信号
    类型：量能与开盘价偏离因子
    公式：
        ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    用法：
        - 用处：开盘价低位偏离 vs 量能相关性
        - 适用场景：开盘跳空、低位启动
        - 数值含义：True=看多，False=看空；返回-1/0
    """
    adv60 = sma(volume, 60)
    w = 0.00817205
    x1 = rank(correlation(Open * w + vwap * (1 - w), ts_sum(adv60, 9), 6))
    x2 = rank((Open - ts_min(Open, 14)))
    alpha = x1 < x2
    return alpha * -1

# ------------------------------ Alpha66 ------------------------------
def alpha66(vwap,low,Open,high):
    """
    逻辑：VWAP4日差分线性衰减排名，结合加权低价与VWAP偏离的线性衰减时序排名，组合后取反
    类型：成交均价衰减趋势因子
    公式：
        ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    用法：
        - 用处：捕捉衰减型趋势，适合震荡回落/反弹
        - 适用场景：短期趋势衰减行情
        - 数值含义：越大越看空，越小越看多
    """
    x1 = rank(decay_linear(delta(vwap, 4), 7))
    w = 0.96633
    x2 = ((low * w + low * (1 - w)) - vwap) / (Open - (high + low) / 2)
    x3 = ts_rank(decay_linear(x2, 11), 7)
    alpha = (x1 + x3) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha67 ------------------------------
def alpha67(volume,vwap,high,ind):
    """
    逻辑：行业中性VWAP与20日均量的6日相关性排名，以其为指数对高价偏离排名做幂，再取反
    类型：行业中性量价幂指因子
    公式：
        ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    用法：
        - 用处：行业中性后放大相关性强弱信号
        - 适用场景：行业内选股、行业中性组合
        - 数值含义：负向越强越看空
    """
    adv20 = sma(volume, 20)
    r = rank(correlation(IndNeutralize(vwap, ind), IndNeutralize(adv20, ind), 6))
    alpha = pow(rank(high - ts_min(high, 2)), r) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha68 ------------------------------
def alpha68(high, low, vwap):
    """
    逻辑：直接复用alpha41，高低价几何平均减VWAP
    类型：价格中枢因子（复用）
    公式：alpha = sqrt(high*low) - vwap
    用法：
        - 用处：快速判断价格中枢与成交成本偏离
        - 适用场景：日内/短线成本判断
        - 数值含义：正=中枢高于成本看多，负=看空
    """
    return alpha41(high, low, vwap)

# ------------------------------ Alpha69 ------------------------------
def alpha69(volume,vwap,ind,close):
    """
    逻辑：行业中性VWAP3日差分5日最大值排名，为指数对加权收盘价相关性时序排名做幂，取反
    类型：行业中性衰减幂指因子
    公式：
        ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)
    用法：
        - 用处：行业内非线性放大强弱信号
        - 适用场景：行业轮动、行业中性策略
        - 数值含义：负向越强看空越强
    """
    adv20 = sma(volume, 20)
    w = 0.490655
    r1 = rank(ts_max(delta(IndNeutralize(vwap, ind), 3), 5))
    r2 = ts_rank(correlation(close * w + vwap * (1 - w), adv20, 5), 9)
    alpha = pow(r1, r2) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha70 ------------------------------
def alpha70(close,ind,vwap,volume):
    """
    逻辑：行业中性收盘价与50日均量18日相关性时序排名为指数，对VWAP1日差分排名做幂，取反
    类型：行业中性指数化因子
    公式：
        ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
    用法：
        - 用处：行业内非线性放大短期动量
        - 适用场景：行业内强势股筛选
        - 数值含义：负向越强越看空
    """
    adv50 = sma(volume, 50).fillna(0)
    r = ts_rank(correlation(IndNeutralize(close, ind), adv50, 18), 18)
    alpha = pow(rank(delta(vwap, 1)), r) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha71 ------------------------------
def alpha71(volume,close,low,Open,vwap):
    """
    逻辑：双重衰减时序排名，取两者较大值，保留强势信号
    类型：量价衰减择强因子
    公式：
        max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
    用法：
        - 用处：取更强信号，过滤弱信号
        - 适用场景：震荡市、趋势不明时
        - 数值含义：越大信号越强，偏多
    """
    adv180 = sma(volume, 180)
    x1 = ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16)
    x2 = ts_rank(decay_linear(pow(rank((low + Open) - (vwap + vwap)), 2), 16), 4)
    alpha = x1
    alpha[x1 < x2] = x2
    return alpha.fillna(0)

# ------------------------------ Alpha72 ------------------------------
def alpha72(volume,high,low,vwap):
    """
    逻辑：双重衰减排名做比值，放大信号差异
    类型：量价比值因子
    公式：
        (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
    用法：
        - 用处：放大强弱差异，适合排序选股
        - 适用场景：横截面选股
        - 数值含义：越大越看多
    """
    adv40 = sma(volume, 40)
    x1 = rank(decay_linear(correlation((high + low) / 2, adv40, 9), 10))
    x2 = rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))
    alpha = x1 / x2.replace(0, 0.0001)
    return alpha.fillna(0)

# ------------------------------ Alpha73 ------------------------------
def alpha73(vwap,Open,low):
    """
    逻辑：VWAP衰减排名与加权价格变动衰减排名取大，再取反
    类型：衰减趋势反转因子
    公式：
        (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    用法：
        - 用处：捕捉衰减趋势末端反转
        - 适用场景：超跌反弹、超买回落
        - 数值含义：负向越强越看多
    """
    x1 = rank(decay_linear(delta(vwap, 5), 3))
    w = 0.147155
    px = Open * w + low * (1 - w)
    x2 = delta(px, 2) / px
    x3 = ts_rank(decay_linear(-x2, 3), 17)
    alpha = x1
    alpha[x1 < x3] = x3
    return -1 * alpha.fillna(0)

# ------------------------------ Alpha74 ------------------------------
def alpha74(volume,close,high,vwap):
    """
    逻辑：收盘价与30日均量和相关性排名，小于加权高价VWAP与成交量相关性排名，布尔信号
    类型：量能相关性比较因子
    公式：
        ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    用法：
        - 用处：判断哪类相关性更强，决定方向
        - 适用场景：趋势确认/背离
        - 数值含义：True看多，False看空；返回-1/0
    """
    adv30 = sma(volume, 30)
    x1 = rank(correlation(close, ts_sum(adv30, 37), 15))
    w = 0.0261661
    px = high * w + vwap * (1 - w)
    x2 = rank(correlation(rank(px), rank(volume), 11))
    alpha = x1 < x2
    return alpha * -1

# ------------------------------ Alpha75 ------------------------------
def alpha75(volume,vwap,low):
    """
    逻辑：VWAP与成交量4日相关性排名，小于最低价与50日均量12日相关性排名
    类型：低价量能相关性因子
    公式：
        (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
    用法：
        - 用处：低价量能更相关则看多
        - 适用场景：低位放量、底部确认
        - 数值含义：True看多，False看空
    """
    adv50 = sma(volume, 50)
    x1 = rank(correlation(vwap, volume, 4))
    x2 = rank(correlation(rank(low), rank(adv50), 12))
    alpha = x1 < x2
    return alpha

# ------------------------------ Alpha76 ------------------------------
def alpha76(volume,vwap,low,ind):
    """
    逻辑：行业中性低价与81日均量相关性衰减排名，与VWAP衰减排名取大
    类型：行业中性低价衰减因子
    公式：
        (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)
    用法：
        - 用处：行业内低位强势股筛选
        - 适用场景：行业内部轮动
        - 数值含义：越大信号越强看多
    """
    adv81 = sma(volume, 81).fillna(0)
    r1 = rank(decay_linear(delta(vwap, 1), 12))
    cor = correlation(IndNeutralize(low, ind), adv81, 8)
    r2 = ts_rank(decay_linear(ts_rank(cor, 20), 17), 19)
    alpha = r1
    alpha[r1 < r2] = r2
    return alpha.fillna(0)

# ------------------------------ Alpha77 ------------------------------
def alpha77(volume,high,low,vwap):
    """
    逻辑：双重衰减排名取较小值，保留弱趋势信号
    类型：量价择弱因子
    公式：
        min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    用法：
        - 用处：抓弱趋势、潜伏反转
        - 适用场景：趋势末期、磨底阶段
        - 数值含义：越小越偏多
    """
    adv40 = sma(volume, 40)
    x1 = rank(decay_linear((((high + low) / 2) + high) - (vwap + high), 20))
    x2 = rank(decay_linear(correlation((high + low) / 2, adv40, 3), 6))
    alpha = x1
    alpha[x1 > x2] = x2
    return alpha.fillna(0)

# ------------------------------ Alpha78 ------------------------------
def alpha78(volume,low,vwap):
    """
    逻辑：加权低价与均量相关性排名为底，量价相关性排名为指数，做幂运算
    类型：量能幂指非线性因子
    公式：
        (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    用法：
        - 用处：极度放大强相关信号
        - 适用场景：趋势极强行情
        - 数值含义：越大看多越强
    """
    adv40 = sma(volume, 40)
    w = 0.352233
    part1 = ts_sum(low * w + vwap * (1 - w), 20)
    part2 = ts_sum(adv40, 20)
    x1 = rank(correlation(part1, part2, 7))
    x2 = rank(correlation(rank(vwap), rank(volume), 6))
    alpha = pow(x1, x2)
    return alpha.fillna(0)

# ------------------------------ Alpha79 ------------------------------
def alpha79(volume,close,Open,ind,vwap):
    """
    逻辑：行业中性加权收盘价1日差分排名，小于VWAP与150日均量相关性排名，布尔信号
    类型：行业中性量能比较因子
    公式：
        (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
    用法：
        - 用处：行业中性方向判断
        - 适用场景：行业中性组合、对冲
        - 数值含义：-1看多，0看空
    """
    adv150 = sma(volume, 150).fillna(0)
    w = 0.60733
    px = IndNeutralize(close * w + Open * (1 - w), ind)
    r1 = rank(delta(px, 1))
    r2 = rank(correlation(ts_rank(vwap, 4), ts_rank(adv150, 9), 15))
    alpha = (r1 < r2) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha80 ------------------------------
def alpha80(Open,high,ind,volume):
    """
    逻辑：行业中性加权开盘高价4日差分符号排名，为指数对高价量能相关性时序排名做幂
    类型：行业中性开盘动量因子
    公式：
        ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
    用法：
        - 用处：行业内开盘动量强弱
        - 适用场景：早盘选股、日内策略
        - 数值含义：负向越强越看多
    """
    adv10 = sma(volume, 10)
    w = 0.868128
    px = IndNeutralize(Open * w + high * (1 - w), ind)
    r1 = rank(np.sign(delta(px, 4)))
    r2 = ts_rank(correlation(high, adv10, 5), 6)
    alpha = pow(r1, r2) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha81 ------------------------------
def alpha81(volume,vwap):
    """
    逻辑：对数乘积排名与量价相关性排名比较，布尔信号
    类型：量能对数趋势因子
    公式：
        ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    用法：
        - 用处：对数压缩后判断趋势
        - 适用场景：高波动、极值多的行情
        - 数值含义：True看多，False看空；返回-1/0
    """
    adv10 = sma(volume, 10)
    cor = correlation(vwap, ts_sum(adv10, 50), 8)
    part = rank(pow(cor, 4))
    x1 = rank(np.log(product(part, 15)))
    x2 = rank(correlation(rank(vwap), rank(volume), 5))
    alpha = x1 < x2
    return alpha * -1

# ------------------------------ Alpha82 ------------------------------
def alpha82(Open,volume,ind):
    """
    逻辑：行业中性成交量与加权开盘价相关性衰减排名，与开盘差分衰减排名取小再取反
    类型：行业中性开盘量能因子
    公式：
        (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    用法：
        - 用处：行业内开盘量能背离/共振
        - 适用场景：行业开盘异动捕捉
        - 数值含义：负向越强越看多
    """
    r1 = rank(decay_linear(delta(Open, 1), 15))
    w = 0.634196
    px = Open * w + Open * (1 - w)
    cor = correlation(IndNeutralize(volume, ind), px, 17)
    r2 = ts_rank(decay_linear(cor, 7), 13)
    alpha = r1
    alpha[r1 > r2] = r2
    return -1 * alpha.fillna(0)

# ------------------------------ Alpha83 ------------------------------
def alpha83(high,low,close,volume,vwap):
    """
    逻辑：高低价差与5日均价比值的延迟排名，乘以成交量排名，再除以价差与VWAP偏离比值
    类型：日内价差与成交均价因子
    公式：
        ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
    用法：
        - 用处：波动与成交成本背离信号
        - 适用场景：宽幅震荡、变盘前夕
        - 数值含义：正强看多，负强看空
    """
    ratio = (high - low) / (ts_sum(close, 5) / 5)
    x = rank(delay(ratio, 2)) * rank(rank(volume))
    alpha = x / (ratio / (vwap - close))
    return alpha.fillna(0)

# ------------------------------ Alpha84 ------------------------------
def alpha84(vwap,close):
    """
    逻辑：VWAP15日最大值偏离的21日时序排名，以收盘价5日差分为指数做幂
    类型：成交均价动量幂指因子
    公式：
        SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
    用法：
        - 用处：VWAP趋势强弱随价格变动放大
        - 适用场景：趋势延续/反转
        - 数值含义：越大趋势越强看多
    """
    tr = ts_rank((vwap - ts_max(vwap, 15)), 21)
    dc = delta(close, 5)
    alpha = pow(tr, dc)
    return alpha.fillna(0)

# ------------------------------ Alpha85 ------------------------------
def alpha85(volume,high,close,low):
    """
    逻辑：加权高价收盘价与30日均量相关性排名为底，价格中枢与量能相关性排名为指数做幂
    类型：加权价格量能幂指因子
    公式：
        (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
    def alpha085(self):
    用法：
        - 用处：极度放大量价共振
        - 适用场景：主升/主跌浪
        - 数值含义：越大趋势越强
    """
    adv30 = sma(volume, 30)
    w = 0.876703
    px = high * w + close * (1 - w)
    x1 = rank(correlation(px, adv30, 10))
    x2 = rank(correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7))
    alpha = pow(x1, x2)
    return alpha.fillna(0)

# ------------------------------ Alpha86 ------------------------------
def alpha86(volume,close,Open,vwap):
    """
    逻辑：收盘价与15日均量相关性时序排名，小于开盘收盘与VWAP偏离排名
    类型：量能与开盘收盘偏离因子
    公式：
        ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)
    用法：
        - 用处：量能趋势 vs 日内成本偏离
        - 适用场景：变盘、突破确认
        - 数值含义：True看多，False看空；返回-1/0
    """
    adv20 = sma(volume, 20)
    x1 = ts_rank(correlation(close, sma(adv20, 15), 6), 20)
    x2 = rank(((Open + close) - (vwap + Open)))
    alpha = x1 < x2
    return alpha * -1

# ------------------------------ Alpha87 ------------------------------
def alpha87(volume,close,vwap,ind):
    """
    逻辑：行业中性均量与收盘价相关性绝对值衰减排名，与加权收盘价差分衰减排名取大
    类型：行业中性量能衰减因子
    公式：
        (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    用法：
        - 用处：行业内量能趋势强弱
        - 适用场景：行业轮动、行业中性
        - 数值含义：负向越强越看多
    """
    adv81 = sma(volume, 81).fillna(0)
    w = 0.369701
    px = close * w + vwap * (1 - w)
    r1 = rank(decay_linear(delta(px, 2), 3))
    cor = abs(correlation(IndNeutralize(adv81, ind), close, 13))
    r2 = ts_rank(decay_linear(cor, 5), 14)
    alpha = r1
    alpha[r1 < r2] = r2
    return -1 * alpha.fillna(0)

# ------------------------------ Alpha88 ------------------------------
def alpha88(volume,Open,low,high,close):
    """
    逻辑：开盘高低收盘排名差衰减排名，与收盘价60日均量相关性衰减排名取小
    类型：价格结构衰减因子
    公式：
        min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
    用法：
        - 用处：价格结构弱信号潜伏
        - 适用场景：震荡磨底、趋势末期
        - 数值含义：越小越偏多
    """
    adv60 = sma(volume, 60)
    x1 = rank(decay_linear((rank(Open) + rank(low)) - (rank(high) + rank(close)), 8))
    x2 = ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3)
    alpha = x1
    alpha[x1 > x2] = x2
    return alpha.fillna(0)

# ------------------------------ Alpha89 ------------------------------
def alpha89(low,vwap,ind,volume):
    """
    逻辑：加权低价与10日均量相关性衰减排名，减去行业中性VWAP衰减排名
    类型：行业中性低价动量因子
    公式：
        (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
    用法：
        - 用处：行业内低价相对强弱
        - 适用场景：低位反转、超跌股
        - 数值含义：正强看多，负强看空
    """
    adv10 = sma(volume, 10)
    w = 0.967285
    px = low * w + low * (1 - w)
    r1 = ts_rank(decay_linear(correlation(px, adv10, 7), 6), 4)
    r2 = ts_rank(decay_linear(delta(IndNeutralize(vwap, ind), 3), 10), 15)
    alpha = r1 - r2
    return alpha.fillna(0)

# ------------------------------ Alpha90 ------------------------------
def alpha90(volume,close,ind,low):
    """
    逻辑：收盘价5日最大值偏离排名，为指数对行业中性均量低价相关性时序排名做幂，取反
    类型：行业中性价格位置幂指因子
    公式：
        ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
    用法：
        - 用处：行业内价格高位风险放大
        - 适用场景：高位回落、顶部确认
        - 数值含义：负向越强越看空
    """
    adv40 = sma(volume, 40).fillna(0)
    r1 = rank((close - ts_max(close, 5)))
    r2 = ts_rank(correlation(IndNeutralize(adv40, ind), low, 5), 3)
    alpha = pow(r1, r2) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha91 ------------------------------
def alpha91(close,ind,volume,vwap):
    """
    逻辑：行业中性收盘价成交量相关性双重衰减排名，减去VWAP与30日均量相关性衰减排名，再取反
    类型：行业中性综合量价因子
    公式：
        ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    用法：
        - 用处：行业内量价趋势综合打分
        - 适用场景：行业内排序选股
        - 数值含义：正强看多，负强看空
    """
    adv30 = sma(volume, 30)
    cor = correlation(IndNeutralize(close, ind), volume, 10)
    r1 = ts_rank(decay_linear(decay_linear(cor, 16), 4), 5)
    r2 = rank(decay_linear(correlation(vwap, adv30, 4), 3))
    alpha = (r1 - r2) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha92 ------------------------------
def alpha92(volume,high,low,close,Open):
    """
    逻辑：价格中枢与开盘价比较衰减排名，与低价30日均量相关性衰减排名取小
    类型：价格中枢量能择弱因子
    公式：
        min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    用法：
        - 用处：潜伏弱趋势反转
        - 适用场景：磨底、震荡市
        - 数值含义：越小越偏多
    """
    adv30 = sma(volume, 30)
    cond = ((((high + low) / 2) + close) < (low + Open))
    x1 = ts_rank(decay_linear(cond, 15), 19)
    x2 = ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7)
    alpha = x1
    alpha[x1 > x2] = x2
    return alpha.fillna(0)

# ------------------------------ Alpha93 ------------------------------
def alpha93(vwap,ind,volume,close):
    """
    逻辑：行业中性VWAP与81日均量相关性衰减排名，除以加权收盘价差分衰减排名
    类型：行业中性成交均价比值因子
    公式：
        (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
    用法：
        - 用处：行业内VWAP趋势强弱打分
        - 适用场景：行业内趋势跟踪
        - 数值含义：越大越看多
    """
    adv81 = sma(volume, 81).fillna(0)
    r1 = ts_rank(decay_linear(correlation(IndNeutralize(vwap, ind), adv81, 17), 20), 8)
    w = 0.524434
    px = close * w + vwap * (1 - w)
    r2 = rank(decay_linear(delta(px, 3), 16))
    alpha = r1 / r2.replace(0, 0.0001)
    return alpha.fillna(0)

# ------------------------------ Alpha94 ------------------------------
def alpha94(volume,vwap):
    """
    逻辑：VWAP12日最小值偏离排名，为指数对VWAP与60日均量相关性时序排名做幂，再取反
    类型：成交均价低位量能幂指因子
    公式：
        ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    用法：
        - 用处：低位放量信号放大
        - 适用场景：底部启动、V型反转
        - 数值含义：负向越强越看多
    """
    adv60 = sma(volume, 60)
    x = rank((vwap - ts_min(vwap, 12)))
    r = ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)
    alpha = pow(x, r) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha95 ------------------------------
def alpha95(volume,high,low,Open):
    """
    逻辑：价格中枢与40日均量相关性5次幂时序排名，与开盘价12日最小值偏离排名比较
    类型：价格中枢量能比较因子
    公式：
        (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    用法：
        - 用处：开盘低位 vs 量能中枢强弱
        - 适用场景：开盘突破/回踩
        - 数值含义：True看多，False看空
    """
    adv40 = sma(volume, 40)
    cor = correlation(sma((high + low) / 2, 19), sma(adv40, 19), 13)
    x = ts_rank(pow(rank(cor), 5), 12)
    o = rank(Open - ts_min(Open, 12))
    alpha = o < x
    return alpha.fillna(0)

# ------------------------------ Alpha96 ------------------------------
def alpha96(volume,vwap,close):
    """
    逻辑：VWAP成交量相关性衰减排名，与收盘价60日均量相关性极值衰减排名取大
    类型：量价极值择强因子
    公式：
        (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    用法：
        - 用处：取最强信号，过滤噪音
        - 适用场景：趋势明确行情
        - 数值含义：越大越看多
    """
    adv60 = sma(volume, 60)
    x1 = ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8)
    cor = correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4)
    x2 = ts_rank(decay_linear(ts_argmax(cor, 13), 14), 13)
    alpha = x1
    alpha[x1 < x2] = x2
    return alpha.fillna(0)

# ------------------------------ Alpha97 ------------------------------
def alpha97(volume,low,vwap,ind):
    """
    逻辑：行业中性加权低价VWAP3日差分衰减排名，减去低价60日均量相关性衰减排名，再取反
    类型：行业中性低价VWAP因子
    公式：
        ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    用法：
        - 用处：行业内低位成本偏离信号
        - 适用场景：行业低位反转
        - 数值含义：正强看多，负强看空
    """
    adv60 = sma(volume, 60).fillna(0)
    w = 0.721001
    px = IndNeutralize(low * w + vwap * (1 - w), ind)
    r1 = rank(decay_linear(delta(px, 3), 20))
    cor = correlation(ts_rank(low, 8), ts_rank(adv60, 17), 5)
    r2 = ts_rank(decay_linear(ts_rank(cor, 19), 16), 7)
    alpha = (r1 - r2) * -1
    return alpha.fillna(0)

# ------------------------------ Alpha98 ------------------------------
def alpha98(volume,Open,vwap):
    """
    逻辑：VWAP与5日均量相关性衰减排名，减去开盘价与15日均量相关性极值时序排名衰减
    类型：双重量能衰减因子
    公式：
        (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    用法：
        - 用处：捕捉量能趋势差
        - 适用场景：量能切换、风格切换
        - 数值含义：正强看多，负强看空
    """
    adv5 = sma(volume, 5)
    adv15 = sma(volume, 15)
    x1 = rank(decay_linear(correlation(vwap, sma(adv5, 26), 5), 7))
    cor = correlation(rank(Open), rank(adv15), 21)
    x2 = rank(decay_linear(ts_rank(ts_argmin(cor, 9), 7), 8))
    alpha = x1 - x2
    return alpha.fillna(0)

# ------------------------------ Alpha99 ------------------------------
def alpha99(volume,high,low):
    """
    逻辑：价格中枢与60日均量20日相关性排名，小于低价与成交量6日相关性排名
    类型：价格中枢量能比较因子
    公式：
        ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)
    用法：
        - 用处：低价量能更相关则看多
        - 适用场景：低位放量、底部确认
        - 数值含义：True看多，False看空；返回-1/0
    """
    adv60 = sma(volume, 60)
    x1 = rank(correlation(ts_sum(((high + low) / 2), 20), ts_sum(adv60, 20), 9))
    x2 = rank(correlation(low, volume, 6))
    alpha = x1 < x2
    return alpha * -1

# ------------------------------ Alpha100 ------------------------------
def alpha100(volume,close,low,high,ind):
    """
    逻辑：价格位置成交量行业中性标准化，减去收盘价均量相关性行业中性标准化，再乘以量能比
    类型：行业中性终极量价因子
    公式：
        (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),IndClass.subindustry))) * (volume / adv20))))
    用法：
        - 用处：全维度量价+行业中性，最综合因子
        - 适用场景：全市场行业中性选股、对冲组合
        - 数值含义：负向越强越看多，正向越强越看空
    """
    adv20 = sma(volume, 20)
    pos = (((close - low) - (high - close)) / (high - low).replace(0, 0.0001)) * volume
    r1 = IndNeutralize(rank(pos), ind)
    r2 = 1.5 * scale(IndNeutralize(r1, ind))
    cor = correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))
    r3 = scale(IndNeutralize(cor, ind))
    alpha = -1 * (r2 - r3) * (volume / adv20)
    return alpha.fillna(0)

# ------------------------------ Alpha101 ------------------------------
def alpha101(close,Open,high,low):
    """
    逻辑：经典日内收益因子，（收盘价-开盘价）/（最高价-最低价+小常数）
    类型：日内动量/反转因子（经典）
    公式：alpha = (close - Open) / (high - low + 0.001)
    用法：
        - 用处：衡量日内趋势强度，标准化日内收益
        - 适用场景：日内交易、短线反转、全市场通用
        - 数值含义：正=日内强势看多，负=日内弱势看空；绝对值越大越强
    """
    alpha = (close - Open) / ((high - low) + 0.001)
    return alpha