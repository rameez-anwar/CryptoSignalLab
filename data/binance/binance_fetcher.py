from binance.client import Client
import pandas as pd
import datetime
import time
import os
from data.utils.db_utils import get_pg_engine
from data.utils.data_inserter_utils import DataInserter
from sqlalchemy import text

class BinanceDataFetcher:
    def __init__(self, api_key: str, api_secret: str, symbol: str, timeframe: str):
        self.client = Client(api_key, api_secret)
        self.timeout_limit = 9 * 60
        self.symbol = symbol.lower()
        self.timeframe = timeframe.lower()
        self.engine = get_pg_engine()
        self.data_inserter = DataInserter(self.engine)
        self.schema = 'binance_data'
        self.table_name = f"{self.symbol}_{self.timeframe}"
        self._init_table()

    def _init_table(self):
        # Ensure schema and table exist
        self.data_inserter._schema_exists(self.schema)
        self.data_inserter._create_table(self.table_name, self.schema)

    def _insert_data_to_db(self, df: pd.DataFrame):
        if df.empty:
            return
        self.data_inserter.save_dataframe(df, 'binance', self.symbol, self.timeframe)

    def _get_existing_data_range(self):
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT MIN(datetime), MAX(datetime) FROM {self.schema}.{self.table_name}")
                result = conn.execute(query).fetchone()
                if result and result[0] and result[1]:
                    return (pd.to_datetime(result[0]), pd.to_datetime(result[1]))
                else:
                    return (None, None)
        except Exception as e:
            print(f"Error getting existing data range: {e}")
            return (None, None)

    def _get_existing_data_gaps(self, start_time: datetime.datetime, end_time: datetime.datetime):
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT datetime FROM {self.schema}.{self.table_name} WHERE datetime >= :start AND datetime <= :end ORDER BY datetime ASC")
                rows = conn.execute(query, {"start": start_time, "end": end_time}).fetchall()
                existing_times = [pd.to_datetime(row[0]) for row in rows]
                if not existing_times:
                    return [(start_time, end_time)]
                gaps = []
                if existing_times[0] > start_time:
                    gaps.append((start_time, existing_times[0]))
                for i in range(len(existing_times) - 1):
                    current_time = existing_times[i]
                    next_time = existing_times[i + 1]
                    expected_next = self._get_next_datetime(current_time)
                    if next_time > expected_next:
                        gaps.append((expected_next, next_time))
                if existing_times[-1] < end_time:
                    last_expected = self._get_next_datetime(existing_times[-1])
                    if last_expected <= end_time:
                        gaps.append((last_expected, end_time))
                return gaps
        except Exception as e:
            print(f"Error getting existing data gaps: {e}")
            return [(start_time, end_time)]

    def _get_next_datetime(self, current_time: datetime.datetime) -> datetime.datetime:
        if 'm' in self.timeframe:
            minutes = int(self.timeframe.replace('m', ''))
        elif 'h' in self.timeframe:
            minutes = int(self.timeframe.replace('h', '')) * 60
        else:
            minutes = 1
        return current_time + datetime.timedelta(minutes=minutes)

    def _fetch_missing_data(self, start_time: datetime.datetime, end_time: datetime.datetime) -> pd.DataFrame:
        gaps = self._get_existing_data_gaps(start_time, end_time)
        if not gaps:
            print("All requested data already exists in database")
            return pd.DataFrame()
        all_missing_data = []
        for gap_start, gap_end in gaps:
            print(f"Fetching missing data: {gap_start} to {gap_end}")
            gap_data = self._fetch_all_data(self.symbol, gap_start, gap_end, self.timeframe)
            if not gap_data.empty:
                all_missing_data.append(gap_data)
        if not all_missing_data:
            return pd.DataFrame()
        df = pd.concat(all_missing_data, ignore_index=True)
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
        return df

    def _fetch_all_data(self, symbol: str, start_time: datetime.datetime, end_time: datetime.datetime, timeframe: str = '1m') -> pd.DataFrame:
        symbol_pair = symbol.upper() + 'USDT'
        print(f"Fetching {symbol_pair} data from {start_time} to {end_time}")
        print(f"Using smart batching to avoid 10-minute timeout")
        all_data = []
        current_start = start_time
        batch_count = 0
        while current_start < end_time:
            batch_count += 1
            batch_minutes = self._calculate_batch_size(current_start, end_time)
            batch_duration = datetime.timedelta(minutes=batch_minutes)
            current_end = min(current_start + batch_duration, end_time)
            print(f"Fetching batch {batch_count}: {current_start} to {current_end} ({batch_minutes:.0f} minutes)")
            batch_df = self._fetch_batch(symbol, current_start, current_end)
            if not batch_df.empty:
                all_data.append(batch_df)
                print(f"  Retrieved {len(batch_df)} records")
            else:
                print(f"  No data for this batch")
            time.sleep(0.1)
            current_start = current_end
        if not all_data:
            print("No data retrieved")
            return pd.DataFrame()
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
        if timeframe != '1m':
            df.set_index('datetime', inplace=True)
            if 'm' in timeframe:
                minutes = int(timeframe.replace('m', ''))
            elif 'h' in timeframe:
                minutes = int(timeframe.replace('h', '')) * 60
            else:
                minutes = 1
            df = df.resample(f'{minutes}min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            df['volume'] = df['volume'].round(2)
            df.reset_index(inplace=True)
        print(f"Total records fetched: {len(df)}")
        return df

    def _calculate_batch_size(self, start_time: datetime.datetime, end_time: datetime.datetime) -> int:
        total_minutes = (end_time - start_time).total_seconds() / 60
        if total_minutes <= self.timeout_limit:
            return total_minutes
        else:
            return min(8 * 60, total_minutes)

    def _fetch_batch(self, symbol: str, start_time: datetime.datetime, end_time: datetime.datetime) -> pd.DataFrame:
        symbol_pair = symbol.upper() + 'USDT'
        try:
            # Use the Binance Futures endpoint for USDT-margined futures
            raw_klines = self.client.futures_historical_klines(
                symbol=symbol_pair,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                start_str=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_str=end_time.strftime('%Y-%m-%d %H:%M:%S')
            )
            if not raw_klines:
                return pd.DataFrame()
            chunk_df = pd.DataFrame(raw_klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            chunk_df[['Open', 'High', 'Low', 'Close', 'Volume']] = chunk_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            chunk_df['datetime'] = pd.to_datetime(chunk_df['Open Time'], unit='ms', )
            chunk_df['open'] = chunk_df['Open']
            chunk_df['high'] = chunk_df['High']
            chunk_df['low'] = chunk_df['Low']
            chunk_df['close'] = chunk_df['Close']
            chunk_df['volume'] = chunk_df['Volume']
            chunk_df = chunk_df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            chunk_df['volume'] = chunk_df['volume'].round(2)
            return chunk_df
        except Exception as e:
            print(f"  Error fetching batch: {e}")
            return pd.DataFrame()

    def get_data_from_db(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                query = f'SELECT datetime, open, high, low, close, volume FROM {self.schema}.{self.table_name}'
                params = {}
                if start_time:
                    query += ' WHERE datetime >= :start'
                    params['start'] = start_time
                if end_time:
                    if params:
                        query += ' AND datetime <= :end'
                    else:
                        query += ' WHERE datetime <= :end'
                    params['end'] = end_time
                query += ' ORDER BY datetime ASC'
                df = pd.read_sql_query(text(query), conn, params=params)
                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df['volume'] = df['volume'].round(2)
                return df
        except Exception as e:
            print(f"Error retrieving data from database: {e}")
            return pd.DataFrame()

    def fetch_data(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None, drop_last_candle: bool = True, use_cache: bool = True) -> pd.DataFrame:
        if end_time is None:
            end_time = datetime.datetime.now()
        if start_time is None:
            start_time = end_time - datetime.timedelta(days=30)
        if use_cache:
            new_data = self._fetch_missing_data(start_time, end_time)
            if not new_data.empty:
                self._insert_data_to_db(new_data)
            df = self.get_data_from_db(start_time, end_time)
        else:
            df = self._fetch_all_data(self.symbol, start_time, end_time, self.timeframe)
            if not df.empty:
                self._insert_data_to_db(df)
        if drop_last_candle and len(df) > 0:
            df = df.iloc[:-1]
            print(f"Dropped last candle (incomplete)")
        print(f"Total records returned: {len(df)}")
        return df

    def interpolate_missing(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if df.empty:
            return df
        df.set_index('datetime', inplace=True)
        if timeframe == '1m':
            freq = '1min'
        elif timeframe == '5m':
            freq = '5min'
        elif timeframe == '15m':
            freq = '15min'
        elif timeframe == '1h':
            freq = '1H'
        else:
            freq = '1min'
        df = df.asfreq(freq)
        df.interpolate(method='linear', inplace=True)
        df.reset_index(inplace=True)
        return df

if __name__ == "__main__":
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    symbol = 'BTC'
    timeframe = '1m'
    start = datetime.datetime(2025, 6, 1)
    fetcher = BinanceDataFetcher(api_key, api_secret, symbol, timeframe)
    fetcher.fetch_data(start_time=start, 
                       end_time=None, 
                       drop_last_candle=True, 
                       use_cache=True)

