import pandas as pd
from data.utils.db_utils import get_pg_engine
from backtest import Backtester

def main(strategy_name: str):
    engine = get_pg_engine()
    print("Fetching data from DB...")
    # Directly fetch from DB
    signals_table = f"signals.{strategy_name}"
    ohlcv_table = "binance_data.btc_1m"  

    signals = pd.read_sql_query(f"SELECT * FROM {signals_table}", engine)
    ohlcv = pd.read_sql_query(f"SELECT * FROM {ohlcv_table}", engine)

    ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
    signals['datetime'] = pd.to_datetime(signals['datetime'])
    ohlcv = ohlcv.sort_values('datetime')
    signals = signals.sort_values('datetime')
    print(f"ohlcv: {ohlcv.head()}")
    print(f"signals: {signals.head()}")
    print("Backtesting...")
    backtester = Backtester(ohlcv_df=ohlcv, signals_df=signals)
    result = backtester.run()
    if not result.empty:
        result[['buy_price', 'sell_price', 'pnl_percent', 'pnl_sum', 'balance']] = result[['buy_price', 'sell_price', 'pnl_percent', 'pnl_sum', 'balance']].round(2)
        print(f"\n Final Balance: {result.iloc[-1]['balance']:.2f}")
        print(f"Total Trades: {len(result[result['action'].isin(['tp', 'sl', 'direction_change'])])}")
        print(result.tail(10))
        # Save to DB as strategies_backtest.{strategy_name}_backtest
        table_name = f"{strategy_name}_backtest"
        result.to_sql(table_name, engine, schema='strategies_backtest', if_exists='replace', index=False)
        print(f"Backtest results stored in DB as strategies_backtest.{table_name}")
    else:
        print("No trades were executed.")

if __name__ == "__main__":
    main("strategy_01")