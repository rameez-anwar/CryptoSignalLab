import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_returns(pnl_sum_series):
    """Calculate various return metrics from cumulative PnL sum"""
    if len(pnl_sum_series) == 0:
        return {
            'total_return': 0,
            'daily_return': 0,
            'weekly_return': 0,
            'monthly_return': 0,
            'yearly_return': 0,
            'cagr': 0
        }
    
    final_pnl_sum = float(pnl_sum_series.iloc[-1])
    initial_balance = 1000
    total_return = round(final_pnl_sum, 2)
    daily_return = round(float(pnl_sum_series.iloc[0]), 2) if len(pnl_sum_series) > 0 else 0
    weekly_return = round(float(pnl_sum_series.iloc[min(6, len(pnl_sum_series)-1)]), 2) if len(pnl_sum_series) > 0 else 0
    monthly_return = round(float(pnl_sum_series.iloc[min(29, len(pnl_sum_series)-1)]), 2) if len(pnl_sum_series) > 0 else 0
    yearly_return = round(float(pnl_sum_series.iloc[min(364, len(pnl_sum_series)-1)]), 2) if len(pnl_sum_series) > 0 else 0
    years = len(pnl_sum_series) / 365
    cagr = 0
    if years > 0 and initial_balance > 0 and (initial_balance + final_pnl_sum) > 0:
        cagr = round((((initial_balance + final_pnl_sum) / initial_balance) ** (1/years) - 1) * 100, 2)
    
    return {
        'total_return': total_return,
        'daily_return': daily_return,
        'weekly_return': weekly_return,
        'monthly_return': monthly_return,
        'yearly_return': yearly_return,
        'cagr': cagr
    }

def calculate_ratios(pnl_percent_series, risk_free_rate=0.02):
    """Calculate various financial ratios from individual PnL percentages"""
    if len(pnl_percent_series) == 0:
        return {
            'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0,
            'alpha': 0, 'beta': 0, 'r_squared': 0, 'information_ratio': 0,
            'treynor_ratio': 0, 'profit_factor': 0, 'omega_ratio': 0,
            'gain_to_pain_ratio': 0, 'payoff_ratio': 0, 'cpc_ratio': 0,
            'risk_return_ratio': 0, 'common_sense_ratio': 0
        }
    
    returns = pnl_percent_series / 100
    mean_return = returns.mean()
    std_return = returns.std()
    total_return = pnl_percent_series.sum()
    sharpe_ratio = (mean_return - risk_free_rate/252) / std_return if std_return > 0 else 0
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return - risk_free_rate/252) / downside_deviation if downside_deviation > 0 else 0
    max_drawdown = calculate_drawdowns(pnl_percent_series.cumsum())['max_drawdown']
    calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown < 0 else 0
    positive_returns = pnl_percent_series[pnl_percent_series > 0].sum()
    negative_returns = abs(pnl_percent_series[pnl_percent_series < 0].sum())
    profit_factor = positive_returns / negative_returns if negative_returns > 0 else 0
    omega_ratio = positive_returns / negative_returns if negative_returns > 0 else 0
    gain_to_pain_ratio = positive_returns / negative_returns if negative_returns > 0 else 0
    avg_win = pnl_percent_series[pnl_percent_series > 0].mean() if len(pnl_percent_series[pnl_percent_series > 0]) > 0 else 0
    avg_loss = abs(pnl_percent_series[pnl_percent_series < 0].mean()) if len(pnl_percent_series[pnl_percent_series < 0]) > 0 else 0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    cpc_ratio = (positive_returns - negative_returns) / (positive_returns + negative_returns) if (positive_returns + negative_returns) > 0 else 0
    risk_return_ratio = total_return / (std_return * np.sqrt(252)) if std_return > 0 else 0
    common_sense_ratio = positive_returns / (positive_returns + negative_returns) if (positive_returns + negative_returns) > 0 else 0
    market_returns = pd.Series([0.0001] * len(returns))
    beta = returns.cov(market_returns) / market_returns.var() if market_returns.var() > 0 else 0
    alpha = mean_return - (beta * market_returns.mean())
    r_squared = returns.corr(market_returns) ** 2 if len(returns) > 1 else 0
    information_ratio = (mean_return - market_returns.mean()) / std_return if std_return > 0 else 0
    treynor_ratio = (mean_return - risk_free_rate/252) / beta if beta > 0 else 0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'information_ratio': information_ratio,
        'treynor_ratio': treynor_ratio,
        'profit_factor': profit_factor,
        'omega_ratio': omega_ratio,
        'gain_to_pain_ratio': gain_to_pain_ratio,
        'payoff_ratio': payoff_ratio,
        'cpc_ratio': cpc_ratio,
        'risk_return_ratio': risk_return_ratio,
        'common_sense_ratio': common_sense_ratio
    }

def calculate_drawdowns(pnl_sum_series):
    """Calculate drawdown metrics from cumulative PnL sum"""
    if len(pnl_sum_series) == 0:
        return {
            'max_drawdown': 0, 'max_drawdown_days': 0, 'avg_drawdown': 0,
            'avg_drawdown_days': 0, 'current_drawdown': 0, 'current_drawdown_days': 0,
            'drawdown_duration': 0, 'conditional_drawdown_at_risk': 0
        }
    
    # Convert to numeric and handle any NaN values
    cumulative = pd.to_numeric(pnl_sum_series, errors='coerce').fillna(0)
    
    # Calculate running maximum (peak)
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown as percentage from peak
    # If running_max is 0, avoid division by zero
    drawdown = np.where(running_max != 0, 
                       (cumulative - running_max) / running_max * 100, 
                       0)
    
    # Convert back to pandas series for consistency
    drawdown = pd.Series(drawdown, index=cumulative.index)
    
    # Find maximum drawdown (most negative value)
    max_drawdown = drawdown.min()
    
    # Calculate drawdown periods
    drawdown_periods = []
    current_drawdown_start = None
    current_drawdown_days = 0
    
    for i, dd in enumerate(drawdown):
        if dd < 0:
            if current_drawdown_start is None:
                current_drawdown_start = i
            current_drawdown_days += 1
        else:
            if current_drawdown_start is not None:
                drawdown_periods.append(current_drawdown_days)
                current_drawdown_start = None
                current_drawdown_days = 0
    
    # If we're still in a drawdown at the end
    if current_drawdown_start is not None:
        drawdown_periods.append(current_drawdown_days)
    
    current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
    avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
    avg_drawdown_days = np.mean(drawdown_periods) if drawdown_periods else 0
    conditional_drawdown_at_risk = drawdown[drawdown < 0].quantile(0.05) if len(drawdown[drawdown < 0]) > 0 else 0
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_days': max(drawdown_periods) if drawdown_periods else 0,
        'avg_drawdown': avg_drawdown,
        'avg_drawdown_days': avg_drawdown_days,
        'current_drawdown': current_drawdown,
        'current_drawdown_days': current_drawdown_days,
        'drawdown_duration': len(drawdown_periods),
        'conditional_drawdown_at_risk': conditional_drawdown_at_risk
    }

def calculate_risk_metrics(pnl_percent_series):
    """Calculate risk metrics from individual PnL percentages"""
    if len(pnl_percent_series) == 0:
        return {
            'ulcer_index': 0, 'risk_of_ruin': 0, 'var_95': 0, 'cvar_99': 0,
            'downside_deviation': 0, 'volatility': 0, 'annualized_volatility': 0
        }
    
    # Convert to numeric and handle any NaN values
    pnl_percent_series = pd.to_numeric(pnl_percent_series, errors='coerce').fillna(0)
    
    returns = pnl_percent_series / 100
    
    # Calculate cumulative returns for drawdown calculation
    cumulative = pnl_percent_series.cumsum()
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown as percentage from peak
    # If running_max is 0, avoid division by zero
    drawdown = np.where(running_max != 0, 
                       (cumulative - running_max) / running_max, 
                       0)
    
    # Calculate ulcer index (square root of average squared drawdown)
    ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100
    
    # Calculate win rate and risk of ruin
    win_rate = len(pnl_percent_series[pnl_percent_series > 0]) / len(pnl_percent_series) if len(pnl_percent_series) > 0 else 0
    risk_of_ruin = (1 - win_rate) ** 10 if win_rate < 1 else 0
    
    # Calculate VaR and CVaR
    var_95 = returns.quantile(0.05) * 100
    cvar_99 = returns[returns <= returns.quantile(0.01)].mean() * 100 if len(returns) > 0 else 0
    
    # Calculate downside deviation
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * 100 if len(downside_returns) > 0 else 0
    
    # Calculate volatility
    volatility = returns.std() * 100
    annualized_volatility = volatility * np.sqrt(252)
    
    return {
        'ulcer_index': ulcer_index,
        'risk_of_ruin': risk_of_ruin,
        'var_95': var_95,
        'cvar_99': cvar_99,
        'downside_deviation': downside_deviation,
        'volatility': volatility,
        'annualized_volatility': annualized_volatility
    }

def calculate_statistical_metrics(pnl_percent_series):
    """Calculate statistical distribution metrics from individual PnL percentages"""
    if len(pnl_percent_series) == 0:
        return {'skewness': 0, 'kurtosis': 0}
    
    returns = pnl_percent_series / 100
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis
    }

def calculate_monthly_weekly_metrics(pnl_percent_series):
    """Calculate monthly and weekly performance metrics from individual PnL percentages"""
    if len(pnl_percent_series) == 0:
        return {
            'winning_weeks': 0, 'losing_weeks': 0,
            'winning_months': 0, 'losing_months': 0,
            'winning_months_percent': 0, 'negative_months_percent': 0
        }
    
    weekly_returns = []
    monthly_returns = []
    
    for i in range(0, len(pnl_percent_series), 7):
        week_data = pnl_percent_series.iloc[i:i+7]
        if len(week_data) > 0:
            weekly_returns.append(week_data.sum())
    
    for i in range(0, len(pnl_percent_series), 30):
        month_data = pnl_percent_series.iloc[i:i+30]
        if len(month_data) > 0:
            monthly_returns.append(month_data.sum())
    
    winning_weeks = len([r for r in weekly_returns if r > 0])
    losing_weeks = len([r for r in weekly_returns if r < 0])
    winning_months = len([r for r in monthly_returns if r > 0])
    losing_months = len([r for r in monthly_returns if r < 0])
    winning_months_percent = (winning_months / len(monthly_returns)) * 100 if monthly_returns else 0
    negative_months_percent = (losing_months / len(monthly_returns)) * 100 if monthly_returns else 0
    
    return {
        'winning_weeks': winning_weeks,
        'losing_weeks': losing_weeks,
        'winning_months': winning_months,
        'losing_months': losing_months,
        'winning_months_percent': winning_months_percent,
        'negative_months_percent': negative_months_percent
    }

def calculate_profitability_metrics(pnl_percent_series):
    """Calculate profitability metrics from individual PnL percentages"""
    if len(pnl_percent_series) == 0:
        return {
            'total_profit': 0, 'total_loss': 0, 'net_profit': 0, 'avg_profit_per_trade': 0,
            'avg_loss_per_trade': 0, 'profit_loss_ratio': 0
        }
    
    # Convert to numeric and handle any NaN values
    pnl_percent_series = pd.to_numeric(pnl_percent_series, errors='coerce').fillna(0)
    
    # Calculate total profit and loss
    total_profit = pnl_percent_series[pnl_percent_series > 0].sum()
    total_loss = pnl_percent_series[pnl_percent_series < 0].sum()  # Keep as negative
    net_profit = pnl_percent_series.sum()
    
    # Calculate averages
    winning_trades = pnl_percent_series[pnl_percent_series > 0]
    losing_trades = pnl_percent_series[pnl_percent_series < 0]
    
    avg_profit_per_trade = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss_per_trade = losing_trades.mean() if len(losing_trades) > 0 else 0
    
    # Calculate profit/loss ratio (use absolute values for ratio)
    profit_loss_ratio = total_profit / abs(total_loss) if total_loss != 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_loss': total_loss,  # Keep as negative value
        'net_profit': net_profit,
        'avg_profit_per_trade': avg_profit_per_trade,
        'avg_loss_per_trade': avg_loss_per_trade,
        'profit_loss_ratio': profit_loss_ratio
    }

def calculate_trade_metrics(df):
    """Calculate comprehensive trade metrics from backtest dataframe"""
    if df.empty:
        return {
            'number_of_trades': 0, 'win_rate': 0, 'loss_rate': 0,
            'average_win': 0, 'average_loss': 0, 'average_trade_duration': 0,
            'largest_win': 0, 'largest_loss': 0, 'consecutive_wins': 0,
            'consecutive_losses': 0, 'avg_trade_return': 0, 'profitability_per_trade': 0,
            'recovery_factor': 0
        }
    
    # For ML models, we need to handle different action types
    # Look for completed trades (tp, sl, direction_change) or any non-null pnl_percent
    if 'action' in df.columns:
        completed_trades = df[df['action'].isin(['tp', 'sl', 'direction_change'])]
        if completed_trades.empty:
            # If no specific actions found, use all rows with pnl_percent
            completed_trades = df[df['pnl_percent'].notna() & (df['pnl_percent'] != 0)]
    else:
        # For ML models without action column, use all rows with pnl_percent
        completed_trades = df[df['pnl_percent'].notna() & (df['pnl_percent'] != 0)]
    
    if completed_trades.empty:
        return {
            'number_of_trades': 0, 'win_rate': 0, 'loss_rate': 0,
            'average_win': 0, 'average_loss': 0, 'average_trade_duration': 0,
            'largest_win': 0, 'largest_loss': 0, 'consecutive_wins': 0,
            'consecutive_losses': 0, 'avg_trade_return': 0, 'profitability_per_trade': 0,
            'recovery_factor': 0
        }
    
    # Convert pnl_percent to numeric and handle NaN values
    pnl_percent_series = pd.to_numeric(completed_trades['pnl_percent'], errors='coerce').fillna(0)
    
    # Filter out transaction fees or very small values that might be noise
    pnl_percent_series = pnl_percent_series[abs(pnl_percent_series) > 0.01]
    
    if len(pnl_percent_series) == 0:
        return {
            'number_of_trades': 0, 'win_rate': 0, 'loss_rate': 0,
            'average_win': 0, 'average_loss': 0, 'average_trade_duration': 0,
            'largest_win': 0, 'largest_loss': 0, 'consecutive_wins': 0,
            'consecutive_losses': 0, 'avg_trade_return': 0, 'profitability_per_trade': 0,
            'recovery_factor': 0
        }
    
    number_of_trades = len(pnl_percent_series)
    winning_trades = pnl_percent_series[pnl_percent_series > 0]
    losing_trades = pnl_percent_series[pnl_percent_series < 0]
    
    win_rate = (len(winning_trades) / number_of_trades) * 100 if number_of_trades > 0 else 0
    loss_rate = (len(losing_trades) / number_of_trades) * 100 if number_of_trades > 0 else 0
    
    average_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    average_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    largest_win = winning_trades.max() if len(winning_trades) > 0 else 0
    largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0
    
    # Calculate consecutive wins/losses
    consecutive_wins = 0
    max_consecutive_wins = 0
    current_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    current_losses = 0
    
    for pnl in pnl_percent_series:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    consecutive_wins = max_consecutive_wins
    consecutive_losses = max_consecutive_losses
    
    avg_trade_return = pnl_percent_series.mean() if len(pnl_percent_series) > 0 else 0
    profitability_per_trade = (pnl_percent_series[pnl_percent_series > 0].sum() / number_of_trades) if number_of_trades > 0 else 0
    
    # Calculate recovery factor
    max_drawdown = calculate_drawdowns(pnl_percent_series.cumsum())['max_drawdown']
    recovery_factor = (pnl_percent_series.sum() / abs(max_drawdown)) if max_drawdown < 0 else 0
    
    return {
        'number_of_trades': number_of_trades,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'average_win': average_win,
        'average_loss': average_loss,
        'average_trade_duration': 0.52,  # Placeholder
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses,
        'avg_trade_return': avg_trade_return,
        'profitability_per_trade': profitability_per_trade,
        'recovery_factor': recovery_factor
    }

def calculate_long_short_metrics(df):
    """Calculate Long/Short trade metrics from backtest dataframe"""
    if df.empty:
        return {
            'total_long_return': 0, 'avg_long_return_per_trade': 0, 'num_long_trades': 0,
            'win_rate_long_trades': 0, 'avg_long_trade_duration': 0, 'max_long_trade_return': 0,
            'min_long_trade_return': 0, 'long_trades_percent': 0,
            'total_short_return': 0, 'avg_short_return_per_trade': 0, 'num_short_trades': 0,
            'win_rate_short_trades': 0, 'avg_short_trade_duration': 0, 'max_short_trade_return': 0,
            'min_short_trade_return': 0, 'short_trades_percent': 0
        }
    
    # Filter completed trades (tp, sl, direction_change)
    completed_trades = df[df['action'].isin(['tp', 'sl', 'direction_change'])].copy()
    
    if completed_trades.empty:
        return {
            'total_long_return': 0, 'avg_long_return_per_trade': 0, 'num_long_trades': 0,
            'win_rate_long_trades': 0, 'avg_long_trade_duration': 0, 'max_long_trade_return': 0,
            'min_long_trade_return': 0, 'long_trades_percent': 0,
            'total_short_return': 0, 'avg_short_return_per_trade': 0, 'num_short_trades': 0,
            'win_rate_short_trades': 0, 'avg_short_trade_duration': 0, 'max_short_trade_return': 0,
            'min_short_trade_return': 0, 'short_trades_percent': 0
        }
    
    # Add a column to track trade direction based on the entry action
    # We'll assume each completed trade has a corresponding entry action ('buy' or 'sell') earlier in the dataframe
    completed_trades['trade_direction'] = None
    
    # Iterate through completed trades and find their corresponding entry action
    for idx, trade in completed_trades.iterrows():
        # Look for the most recent 'buy' or 'sell' action before this exit
        prior_rows = df[df.index < idx]
        entry_rows = prior_rows[prior_rows['action'].isin(['buy', 'sell'])].tail(1)
        if not entry_rows.empty:
            completed_trades.loc[idx, 'trade_direction'] = 'long' if entry_rows.iloc[0]['action'] == 'buy' else 'short'
    
    # Filter long and short trades
    long_trades = completed_trades[completed_trades['trade_direction'] == 'long']
    short_trades = completed_trades[completed_trades['trade_direction'] == 'short']
    total_trades = len(completed_trades)
    
    # Long trade metrics
    total_long_return = long_trades['pnl_percent'].sum() if not long_trades.empty else 0
    avg_long_return_per_trade = long_trades['pnl_percent'].mean() if not long_trades.empty else 0
    num_long_trades = len(long_trades)
    win_rate_long_trades = (len(long_trades[long_trades['pnl_percent'] > 0]) / num_long_trades * 100) if num_long_trades > 0 else 0
    avg_long_trade_duration = 0.52  # Placeholder
    max_long_trade_return = long_trades['pnl_percent'].max() if not long_trades.empty else 0
    min_long_trade_return = long_trades['pnl_percent'].min() if not long_trades.empty else 0
    long_trades_percent = (num_long_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Short trade metrics
    total_short_return = short_trades['pnl_percent'].sum() if not short_trades.empty else 0
    avg_short_return_per_trade = short_trades['pnl_percent'].mean() if not short_trades.empty else 0
    num_short_trades = len(short_trades)
    win_rate_short_trades = (len(short_trades[short_trades['pnl_percent'] > 0]) / num_short_trades * 100) if num_short_trades > 0 else 0
    avg_short_trade_duration = 0.52  # Placeholder
    max_short_trade_return = short_trades['pnl_percent'].max() if not short_trades.empty else 0
    min_short_trade_return = short_trades['pnl_percent'].min() if not short_trades.empty else 0
    short_trades_percent = (num_short_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'total_long_return': total_long_return,
        'avg_long_return_per_trade': avg_long_return_per_trade,
        'num_long_trades': num_long_trades,
        'win_rate_long_trades': win_rate_long_trades,
        'avg_long_trade_duration': avg_long_trade_duration,
        'max_long_trade_return': max_long_trade_return,
        'min_long_trade_return': min_long_trade_return,
        'long_trades_percent': long_trades_percent,
        'total_short_return': total_short_return,
        'avg_short_return_per_trade': avg_short_return_per_trade,
        'num_short_trades': num_short_trades,
        'win_rate_short_trades': win_rate_short_trades,
        'avg_short_trade_duration': avg_short_trade_duration,
        'max_short_trade_return': max_short_trade_return,
        'min_short_trade_return': min_short_trade_return,
        'short_trades_percent': short_trades_percent
    }

def generate_stats_from_backtest(df):
    """Main function to generate all stats from backtest dataframe"""
    if df.empty:
        return pd.DataFrame()
    
    required_columns = ['pnl_percent', 'pnl_sum', 'balance', 'action']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in dataframe: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    pnl_percent_series = df['pnl_percent']
    pnl_sum_series = df['pnl_sum']
    
    returns = calculate_returns(pnl_sum_series)
    ratios = calculate_ratios(pnl_percent_series)
    drawdowns = calculate_drawdowns(pnl_sum_series)
    risk_metrics = calculate_risk_metrics(pnl_percent_series)
    statistical_metrics = calculate_statistical_metrics(pnl_percent_series)
    monthly_weekly = calculate_monthly_weekly_metrics(pnl_percent_series)
    profitability = calculate_profitability_metrics(pnl_percent_series)
    trade_metrics = calculate_trade_metrics(df)
    long_short_metrics = calculate_long_short_metrics(df)
    
    all_metrics = {
        **returns,
        **ratios,
        **drawdowns,
        **risk_metrics,
        **statistical_metrics,
        **monthly_weekly,
        **profitability,
        **trade_metrics,
        **long_short_metrics
    }
    
    stats_df = pd.DataFrame([all_metrics])
    return stats_df