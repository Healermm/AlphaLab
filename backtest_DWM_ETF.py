# backtest_DWM.py
'''
修复说明：
  1. order_percent 改为 rebalance_to_target：计算持仓差值，避免重复叠加买入
  2. rebalance_dates 改用 set，查找从 O(n) → O(1)，并统一 normalize 避免时区错位
  3. 新增最小持仓期检查（min_holding_days=2）
  4. 新增调仓阈值检查（weight_drift_threshold=0.05），权重偏离 >5% 才调仓
  5. 新增风控模块：
       - 单行业止损 >5% 强制平仓
       - 组合回撤 >8%  → 降仓至50%
       - 组合回撤 >12% → 全部平仓
  6. 主流程中补充 etf_vol_dict 计算（供 risk_parity 使用）
'''

from datetime import date

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import config
from utils import strip_stock_suffix
from industry_rotation import build_industry_scores
from portfolio_construction import construct_portfolio
from ic_analysis_DWM import get_rebalance_dates

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════
def load_price_data(start_date, end_date, stock_list, price_dir=config.STOCK_DAILY_DIR):
    dates = pd.date_range(start_date, end_date, freq='B')
    price_dict = {}
    money_dict = {}
    open_dict = {}
    for code in stock_list:
        file_path = os.path.join(price_dir, f'{code}.csv')
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        
        # 兼容 'date' 和 'datetime' 两种列名
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'date'})
        
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = df[~df['date'].duplicated(keep='last')] 
        df.set_index('date', inplace=True)
        df['close'] = df['close'].replace(0, np.nan).ffill()
        df['open'] = df['open'].replace(0, np.nan).ffill()
        price_dict[code] = df['close'].reindex(dates).ffill()
        money_dict[code] = df['money'].reindex(dates).ffill()
        open_dict[code] = df['open'].reindex(dates).ffill()
        
    price_df = pd.DataFrame(price_dict, index=dates)
    money_df = pd.DataFrame(money_dict, index=dates)
    open_df = pd.DataFrame(open_dict, index=dates)

    return price_df.dropna(axis=1, how='all'), money_df.dropna(axis=1, how='all'), open_df.dropna(axis=1, how='all') 

def calc_etf_volatility(etf_price_df: pd.DataFrame, window: int = 20) -> dict:
    """
    计算 ETF 滚动年化波动率（用于 risk_parity 权重）。
    返回最新一期的 {etf_code: annualized_vol}。
    """
    ret = etf_price_df.pct_change().dropna(how='all')
    vol = ret.rolling(window).std().iloc[-1] * np.sqrt(252)
    return vol.dropna().to_dict()

def calc_performance(nav_df, bm_ret=None, risk_free_rate=0.02):
    nav = nav_df.set_index('date')['nav']
    daily_ret = nav.pct_change().dropna()
    
    # 年化收益率
    n_days = len(daily_ret)
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    annual_ret = (1 + total_ret) ** (252 / n_days) - 1
    
    # 年化波动率
    annual_vol = daily_ret.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe = (annual_ret - risk_free_rate) / annual_vol if annual_vol > 0 else np.nan
    
    # 最大回撤
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()
    
    # Calmar比率
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    # 胜率和盈亏比
    win_rate = (daily_ret > 0).mean()
    avg_win = daily_ret[daily_ret > 0].mean()
    avg_loss = daily_ret[daily_ret < 0].mean()
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    
    # 信息比率（相对基准）
    ir = np.nan
    if bm_ret is not None:
        bm_aligned = bm_ret.reindex(daily_ret.index).fillna(0)
        excess_ret = daily_ret - bm_aligned
        ir = excess_ret.mean() / excess_ret.std() * np.sqrt(252) if excess_ret.std() > 0 else np.nan
    
    # 换手率（需要nav_df里有market_value，用市值变化估算）
    # 简化：每次调仓的换手用持仓变化金额 / 总nav 估算，这里先输出NaN待后续完善
    
    result = {
        '年化收益率': f'{annual_ret:.2%}',
        '年化波动率': f'{annual_vol:.2%}',
        '夏普比率':   f'{sharpe:.2f}',
        '最大回撤':   f'{max_dd:.2%}',
        'Calmar比率': f'{calmar:.2f}',
        '信息比率IR': f'{ir:.2f}' if not np.isnan(ir) else 'N/A',
        '胜率':       f'{win_rate:.2%}',
        '盈亏比':     f'{profit_loss_ratio:.2f}',
    }
    return result
# ══════════════════════════════════════════════
# 回测引擎
# ══════════════════════════════════════════════

class SimpleBacktest:
    def __init__(
        self,
        start_date,
        end_date,
        initial_capital: float = 10_000_000,
        commission: float = 0.0003,
        # ── 调仓规则 ──
        min_holding_days: int = 2,           # 最小持仓期（交易日）
        weight_drift_threshold: float = 0.05, # 权重偏离阈值
        # ── 风控参数 ──
        stop_loss_single: float = 0.05,       # 单行业止损线
        drawdown_half: float = 0.08,          # 回撤 >8%  → 降仓50%
        drawdown_clear: float = 0.12,         # 回撤 >12% → 全平
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date   = pd.to_datetime(end_date)
        self.capital    = initial_capital
        self.cash       = initial_capital
        self.commission = commission

        self.min_holding_days       = min_holding_days
        self.weight_drift_threshold = weight_drift_threshold
        self.stop_loss_single       = stop_loss_single
        self.drawdown_half          = drawdown_half
        self.drawdown_clear         = drawdown_clear

        self.positions    = {}   # {etf_code: shares}
        self.entry_price  = {}   # {etf_code: 买入均价}，用于单行业止损
        self.nav_history  = []
        self.peak_nav     = initial_capital
        self.last_rebalance_date = None   # 上次调仓日
        self.days_since_rebalance = 0     # 距上次调仓经过交易日数
        self.money_df = None
        self.bm_ret = None
        self.open_df = None
    # ──────────────────────────────────────────
    # 基础工具
    # ──────────────────────────────────────────
    def get_price(self, date, code) -> float:
            if code not in self.price_df.columns:
                return np.nan
            val = self.price_df.loc[date, code]
            return float(val) if not pd.isna(val) else np.nan
    def get_exec_price(self, date, code, use_open=False) -> float:
        if use_open and hasattr(self, 'open_df') and code in self.open_df.columns:
            val = self.open_df.loc[date, code]
            return float(val) if not pd.isna(val) else self.get_price(date, code)
        return self.get_price(date, code)

    def calc_current_nav(self, date) -> float:
        mv = sum(
            shares * self.get_price(date, code)
            for code, shares in self.positions.items()
            if not np.isnan(self.get_price(date, code))
        )
        return self.cash + mv

    def record_nav(self, date):
        mv = sum(
            shares * self.get_price(date, code)
            for code, shares in self.positions.items()
            if not np.isnan(self.get_price(date, code))
        )
        total = self.cash + mv
        self.nav_history.append({
            'date': date,
            'nav': total,
            'cash': self.cash,
            'market_value': mv
        })
        # 更新历史最高净值
        if total > self.peak_nav:
            self.peak_nav = total
        return total

    # ──────────────────────────────────────────
    # 交易执行（增量调仓）
    # ──────────────────────────────────────────

    def _sell_shares(self, code, shares, date, use_open=False):
        price = self.get_exec_price(date, code, use_open=use_open)
        price = max(price-0.001, 0.001) # 卖出滑点，但不能低于0
        if np.isnan(price) or shares <= 0:
            return
        # 冲击成本限制
        if self.money_df is not None and code in self.money_df.columns:
            avg_money = self.money_df[code].loc[:date].tail(20).mean()
            if not pd.isna(avg_money) and price > 0:
                max_trade_value = avg_money * 0.05
                max_shares = max_trade_value / price
                shares = min(shares, max_shares)
        actual_shares = min(shares, self.positions.get(code, 0))
        if actual_shares <= 0:
            return
        self.cash += actual_shares * price * (1 - self.commission)
        self.positions[code] = self.positions.get(code, 0) - actual_shares
        if self.positions[code] <= 0:
            self.positions.pop(code, None)
            self.entry_price.pop(code, None)

    def _buy_shares_by_value(self, code, target_value, date, use_open=False):
        """买入使仓位市值达到 target_value"""
        price = self.get_exec_price(date, code, use_open=use_open)
        price = price + 0.001  # 买入滑点
        if np.isnan(price) or price <= 0:
            return
        current_value = self.positions.get(code, 0) * price
        delta_value = target_value - current_value
        if delta_value <= 0:
            return
        if self.money_df is not None and code in self.money_df.columns:
            avg_money = self.money_df[code].loc[:date].tail(20).mean()
            if not pd.isna(avg_money):
                max_trade_value = avg_money * 0.05
                delta_value = min(delta_value, max_trade_value)
        cost_per_share = price * (1 + self.commission)
        shares_to_buy = delta_value / cost_per_share
        cost = shares_to_buy * cost_per_share
        if cost > self.cash:
            shares_to_buy = self.cash / cost_per_share
            cost = shares_to_buy * cost_per_share
        if shares_to_buy <= 0:
            return
        # 更新均价
        old_shares = self.positions.get(code, 0)
        old_price  = self.entry_price.get(code, price)
        new_shares = old_shares + shares_to_buy
        self.entry_price[code] = (old_shares * old_price + shares_to_buy * price) / new_shares
        self.positions[code] = new_shares
        self.cash -= cost

    def calc_nav_with_price(self, date, use_open=False) -> float:
        mv = sum(
            shares * self.get_exec_price(date, code, use_open=use_open)
            for code, shares in self.positions.items()
            if not np.isnan(self.get_exec_price(date, code, use_open=use_open))
        )
        return self.cash + mv

    def rebalance_to_target(self, target_portfolio, date, use_open=True):
        
        """
        增量调仓：计算每个标的当前市值与目标市值的差值，
        先卖出超仓/退出标的，再补仓/新建标的，避免重复叠加。
        """
        nav = self.calc_nav_with_price(date, use_open=use_open)
        if nav <= 0:
            return

        # ── 先卖出：不在目标组合，或需要减仓 ──
        all_codes = set(self.positions.keys()) | set(target_portfolio.keys())
        sell_orders = {}
        for code in all_codes:
            target_value   = target_portfolio.get(code, 0.0) * nav
            price          = self.get_exec_price(date, code, use_open=use_open)
            if np.isnan(price) or price <= 0:
                continue
            current_shares = self.positions.get(code, 0)
            current_value  = current_shares * price
            if current_value > target_value + 1e-6:
                # 需要减仓
                delta_value  = current_value - target_value
                shares_to_sell = delta_value / price
                sell_orders[code] = shares_to_sell

        for code, shares in sell_orders.items():
            self._sell_shares(code, shares, date, use_open=use_open)

        # ── 再买入：新建仓或补仓 ──
        for code, weight in target_portfolio.items():
            target_value = weight * nav
            self._buy_shares_by_value(code, target_value, date, use_open=use_open)

    # ──────────────────────────────────────────
    # 风控：单行业止损
    # ──────────────────────────────────────────

    def check_single_stop_loss(self, date):
        """
        单行业持仓亏损 > stop_loss_single → 强制平仓该标的
        以买入均价为基准计算亏损比例
        """
        to_clear = []
        for code, shares in list(self.positions.items()):
            price = self.get_price(date, code)
            if np.isnan(price):
                continue
            avg_cost = self.entry_price.get(code, price)
            pnl_pct  = (price - avg_cost) / avg_cost
            if pnl_pct < -self.stop_loss_single:
                to_clear.append(code)
        for code in to_clear:
            print(f'[止损] {date.date()} 平仓 {code}，亏损超过 {self.stop_loss_single:.0%}')
            self._sell_shares(code, self.positions.get(code, 0), date)

    # ──────────────────────────────────────────
    # 风控：组合回撤保护
    # ──────────────────────────────────────────

    def check_drawdown_protection(self, date) -> bool:
        """
        检查组合最大回撤并执行降仓/清仓。
        返回 True 表示触发了风控（本日不应再正常调仓）。
        """
        nav = self.calc_current_nav(date)
        drawdown = (self.peak_nav - nav) / self.peak_nav if self.peak_nav > 0 else 0

        if drawdown >= self.drawdown_clear:
            print(f'[极端回撤] {date.date()} 回撤={drawdown:.1%}，全部平仓')
            for code in list(self.positions.keys()):
                self._sell_shares(code, self.positions.get(code, 0), date)
            self.peak_nav = self.calc_current_nav(date)  # 重置峰值
            return True

        if drawdown >= self.drawdown_half:
            print(f'[回撤保护] {date.date()} 回撤={drawdown:.1%}，降仓至50%')
            for code in list(self.positions.keys()):
                half_shares = self.positions.get(code, 0) / 2
                self._sell_shares(code, half_shares, date)
            self.peak_nav = self.calc_current_nav(date)  # 更新峰值
            return True

        return False

    # ──────────────────────────────────────────
    # 调仓阈值检查
    # ──────────────────────────────────────────

    def need_rebalance_by_drift(self, target_portfolio: dict, date) -> bool:
        """
        检查当前持仓权重与目标权重的偏离是否超过阈值。
        只要有一个标的偏离 > weight_drift_threshold，则触发调仓。
        """
        nav = self.calc_current_nav(date)
        if nav <= 0:
            return True
        all_codes = set(self.positions.keys()) | set(target_portfolio.keys())
        for code in all_codes:
            price          = self.get_price(date, code)
            current_weight = (self.positions.get(code, 0) * price / nav
                              if not np.isnan(price) else 0.0)
            target_weight  = target_portfolio.get(code, 0.0)
            if abs(current_weight - target_weight) > self.weight_drift_threshold:
                return True
        return False

    # ──────────────────────────────────────────
    # 主回测循环
    # ──────────────────────────────────────────

    def run(
        self,
        industry_score_df: pd.DataFrame,
        price_df: pd.DataFrame,
        money_df: pd.DataFrame,
        open_df: pd.DataFrame,
        rebalance_dates: set,
        n_top: int = 5,
        weighting: str = 'score',
        etf_vol_dict: dict = None,
        bm_ret=None
    ) -> pd.DataFrame:
        """
        执行回测。

        Parameters
        ----------
        industry_score_df : date | industry | industry_score
        price_df          : 宽表，index=date，columns=etf_code
        money_df          : 日均成交额数据框
        rebalance_dates   : set of pd.Timestamp（已 normalize）
        n_top             : construct_portfolio 参数
        weighting         : 权重方式
        etf_vol_dict      : risk_parity 模式下的波动率字典
        """
        self.price_df  = price_df
        self.all_dates = price_df.index
        self.money_df = money_df
        self.bm_ret = bm_ret 
        self.open_df = open_df 

        # 缓存上一期目标组合，用于最小持仓期 + 阈值判断
        last_target_portfolio = {}
        for i, date in enumerate(self.all_dates):
           
            nav = self.calc_current_nav(date)
            drawdown = (self.peak_nav - nav) / self.peak_nav if self.peak_nav > 0 else 0
            # if drawdown > 0.05: # 可定位较大回撤发生的时间点、幅度以及当时的持仓
            #     print(f"[DEBUG] {date.date()} nav={nav:.0f} peak={self.peak_nav:.0f} dd={drawdown:.1%} positions={list(self.positions.keys())}")
            
            # ── 1. 每日风控检查（空仓时跳过）──────
            
            if len(self.positions) == 0:
                risk_triggered = False
            else:
                risk_triggered = self.check_drawdown_protection(date)
                if not risk_triggered:
                    self.check_single_stop_loss(date)
            vol_scale = 1.0
            if self.bm_ret is not None:
                hist_vol = self.bm_ret.loc[:date].tail(20).std() * np.sqrt(252)
                if hist_vol > 0.30:  # 年化波动率超30%
                    vol_scale = 0.5
                    print(f'[波动率过滤] {date.date()} 年化波动={hist_vol:.1%}，目标仓位降至50%')

            # ── 2. 判断是否为调仓日 ───────────────
            is_rebalance_day = date.normalize() in rebalance_dates
            
            if is_rebalance_day and not risk_triggered:
                industry_today = industry_score_df[
                    industry_score_df['date'].dt.normalize() == date.normalize()
                ]
                #print(f"[DEBUG] {date.date()} 调仓日，industry_today行数={len(industry_today)}")
                # 2a. 最小持仓期检查
                if self.last_rebalance_date is not None:
                    self.days_since_rebalance = (
                        (self.all_dates <= date).sum()
                        - (self.all_dates <= self.last_rebalance_date).sum()
                    )
                else:
                    self.days_since_rebalance = self.min_holding_days  # 首次直接允许

                if self.days_since_rebalance < self.min_holding_days:
                    print(f"[持仓期不足] {date.date()} days={self.days_since_rebalance}")
                    self.record_nav(date)
                    continue

                # 2b. 构建目标组合
                industry_today = industry_score_df[
                    industry_score_df['date'].dt.normalize() == date.normalize()
                ]
                # 流动性过滤：近20日日均成交额 < 500万的ETF排除
                liquid_codes = set()
                if self.money_df is not None:
                    for code in self.money_df.columns:
                        avg_money = self.money_df[code].loc[:date].tail(20).mean()
                        if avg_money >= 5_000_000:  # 500万
                            liquid_codes.add(code)
                target_portfolio = construct_portfolio(
                    industry_today,
                    n_top=n_top,
                    weighting=weighting,
                    etf_vol_dict=etf_vol_dict,
                    allowed_codes=liquid_codes,  
                )
                target_portfolio = {k: v * vol_scale for k, v in target_portfolio.items()}

                # 2c. 空信号：维持原仓位
                if not target_portfolio:
                    print(f"[空组合] {date.date()} construct_portfolio返回空")
                    self.record_nav(date)
                    continue

                # 2d. 调仓阈值检查：偏离不足则跳过
                is_empty = len(self.positions) == 0
                portfolio_changed = set(target_portfolio.keys()) != set(last_target_portfolio.keys())
                drift_exceeded    = self.need_rebalance_by_drift(target_portfolio, date)

                if not is_empty and not portfolio_changed and not drift_exceeded:
                    self.record_nav(date)
                    continue

                # 2e. 执行调仓（增量方式）
                idx = list(self.all_dates).index(date)
                if idx + 1 < len(self.all_dates):
                    exec_date = self.all_dates[idx + 1]
                else:
                    exec_date = date  # 最后一天没有次日，当天执行

                self.rebalance_to_target(target_portfolio, exec_date, use_open=True)
                self.last_rebalance_date = exec_date  # 记录实际执行日


            # ── 3. 每日记录净值 ───────────────────
            self.record_nav(date)

        return pd.DataFrame(self.nav_history)

#回测函数，封装一个函数供外部调用，参数包括调仓频率、回测区间、持仓数量、风控参数等，方便后续批量测试不同配置的表现。
def run_backtest(
    freq='W',
    start_date=None,
    end_date=None,
    n_top=5,
    stop_loss_single=0.10,
    drawdown_half=0.25,
    drawdown_clear=0.40,
    weight_drift_threshold=0.05,
    min_holding_days=2,
    weighting='score',
    industry_score_df_override=None,
):
    if start_date is None:
        start_date = config.START_TEST_DATE
    if end_date is None:
        end_date = config.END_DATE

    # ── 1. 加载因子数据 ───────────────────────
    factor_path = os.path.join(config.RESULT_DIR, f'alpha_combined_{freq}.csv')
    if not os.path.exists(factor_path):
        raise FileNotFoundError(f'请先运行 factor_portfolio.py --freq {freq} 生成 {factor_path}')
    factor_df = pd.read_csv(factor_path)
    factor_df['date']  = pd.to_datetime(factor_df['date'])
    factor_df['codes'] = factor_df['codes'].astype(str).str.zfill(6)
    factor_df.set_index(['date', 'codes'], inplace=True)

    # ── 2. 构建行业得分 ───────────────────────
    ind_df = pd.read_csv(config.STOCK_INDUSTRY_FILE)
    ind_df = ind_df.rename(columns={'htsc_code': 'codes'})
    ind_df['codes'] = ind_df['codes'].astype(str).str.split('.').str[0].str.zfill(6)
    if industry_score_df_override is not None:
        industry_score_df = industry_score_df_override
    else:
        industry_score_df = build_industry_scores(
            factor_df.reset_index(), ind_df, method='topk_mean'
        )

    # ── 3. 调仓日期 ───────────────────────────
    rebalance_dates_raw = get_rebalance_dates(start_date, end_date, freq)
    rebalance_dates = set(pd.to_datetime(rebalance_dates_raw).normalize())
    rebalance_dates = {d for d in rebalance_dates if pd.notna(d)}

    # ── 4. 加载ETF价格 ────────────────────────
    from etf_mapping import INDUSTRY_ETF_MAP
    etf_codes = list(INDUSTRY_ETF_MAP.values())
    etf_price_df, etf_money_df, etf_open_df = load_price_data(
        start_date, end_date, etf_codes, price_dir=config.ETF_DAILY_DIR
    )
    etf_vol_dict = calc_etf_volatility(etf_price_df, window=20)

    # ── 5. 基准收益率 ─────────────────────────
    bm_path = os.path.join(config.INDEX_COMPONENT_DIR, '000300.csv')
    bm_df = pd.read_csv(bm_path)
    bm_df['date'] = pd.to_datetime(bm_df['date']).dt.normalize()
    bm_ret = bm_df.set_index('date')['close'].pct_change()

    # ── 6. 运行回测 ───────────────────────────
    bt = SimpleBacktest(
        start_date, end_date,
        initial_capital=1_000_000,
        commission=0.0001,
        min_holding_days=min_holding_days,
        weight_drift_threshold=weight_drift_threshold,
        stop_loss_single=stop_loss_single,
        drawdown_half=drawdown_half,
        drawdown_clear=drawdown_clear,
    )
    nav_df = bt.run(
        industry_score_df=industry_score_df,
        price_df=etf_price_df,
        money_df=etf_money_df,
        open_df=etf_open_df,
        rebalance_dates=rebalance_dates,
        n_top=n_top,
        weighting=weighting,
        etf_vol_dict=etf_vol_dict,
        bm_ret=bm_ret,
    )

    # ── 7. 计算绩效 ───────────────────────────
    perf = calc_performance(nav_df, bm_ret=bm_ret)

    return nav_df, perf


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════

if __name__ == '__main__':
    
    freq = 'W' # 可修改为 'M'、'D' 等不同频率
    nav_df, perf = run_backtest(freq=freq)

    print('\n=== 策略绩效 ===')
    for k, v in perf.items():
        print(f'{k}: {v}')

    # 保存和绘图部分保持不变
    output_dir = config.RESULT_DIR
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'backtest_nav_{freq}_ETF.csv')
    nav_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 6))
    init_nav = nav_df['nav'].iloc[0]
    plt.plot(nav_df['date'], nav_df['nav'] / init_nav, label=f'策略净值 (频率={freq})')

    bm_path = os.path.join(config.INDEX_COMPONENT_DIR, '000300.csv')
    if os.path.exists(bm_path):
        bm_df = pd.read_csv(bm_path)
        bm_df['date'] = pd.to_datetime(bm_df['date']).dt.normalize()
        bm_df = bm_df.set_index('date')
        bm_close = bm_df['close'].reindex(pd.DatetimeIndex(nav_df['date'])).ffill()
        bm_nav = bm_close / bm_close.iloc[0]
        plt.plot(nav_df['date'], bm_nav.values, label='基准(000300)',
                 linestyle='--', alpha=0.7)

    plt.title(f'多因子行业轮动回测净值曲线（{freq} 频调仓）')
    plt.xlabel('日期')
    plt.ylabel('净值')
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(output_dir, f'backtest_nav_{freq}_ETF.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()