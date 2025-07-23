from pybit.unified_trading import HTTP
import pandas as pd
import datetime
import time
import os
from data.utils.db_utils import get_pg_engine
from data.utils.data_inserter_utils import DataInserter
from sqlalchemy import text

class BybitDataFetcher:
    def __init__(self, api_key: str, api_secret: str, symbol: str, timeframe: str):
        """
        Initialize Bybit Data Fetcher.
        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            symbol: Trading symbol (e.g., 'BTC')
            timeframe: Timeframe (e.g., '1m', '5m')
        """
        self.client = HTTP(
            demo=True,
            api_key=api_key,
            api_secret=api_secret
        )
        self.timeout_limit = 9 * 60
        self.symbol = symbol.lower()
        self.timeframe = timeframe.lower()
        self.engine = get_pg_engine()
        self.data_inserter = DataInserter(self.engine)
        self.schema = 'bybit_data'
        self.table_name = f"{self.symbol}_{self.timeframe}"
        self._init_table()

    def _init_table(self):
        """
        Initialize PostgreSQL table with 6 columns: datetime, open, high, low, close, volume.
        """
        try:
            # Ensure schema and table exist
            self.data_inserter._schema_exists(self.schema)
            self.data_inserter._create_table(self.table_name, self.schema)
            print(f"Database initialized: PostgreSQL, schema: {self.schema}, table: {self.table_name}")
        except Exception as e:
            print(f"Error initializing database: {e}")

    def _insert_data_to_db(self, df: pd.DataFrame):
        """
        Insert DataFrame data into PostgreSQL database with proper duplicate handling and sorting.
        Args:
            df: DataFrame with columns: datetime, open, high, low, close, volume
        """
        if df.empty:
            print("No data to insert (empty DataFrame)")
            return
        
        print(f"Attempting to insert {len(df)} records to database table: {self.schema}.{self.table_name}")
        
        try:
            # Use the DataInserter utility for PostgreSQL
            self.data_inserter.save_dataframe(df, 'bybit', self.symbol, self.timeframe)
            print(f"Successfully inserted {len(df)} records into PostgreSQL ({self.schema}.{self.table_name})")
        except Exception as e:
            print(f"Error inserting data to database: {e}")
            import traceback
            traceback.print_exc()

    def _get_existing_data_range(self) -> tuple:
        """
        Get the date range of existing data in the table.
        Returns:
            Tuple of (min_date, max_date) or (None, None) if no data exists
        """
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

    def _get_existing_data_gaps(self, start_time: datetime.datetime, end_time: datetime.datetime) -> list:
        """
        Get gaps in existing data within the specified time range.
        Returns:
            List of tuples (gap_start, gap_end) representing missing data periods
        """
        try:
            with self.engine.connect() as conn:
                # Get all existing data points in the range, sorted by datetime
                query = text(f"""
                    SELECT datetime 
                    FROM {self.schema}.{self.table_name}
                    WHERE datetime >= :start AND datetime <= :end
                    ORDER BY datetime ASC
                """)
                result = conn.execute(query, {'start': start_time, 'end': end_time})
                existing_datetimes = [pd.to_datetime(row[0]) for row in result.fetchall()]
                
                if not existing_datetimes:
                    return [(start_time, end_time)]
                
                gaps = []
                current_time = start_time
                
                # Find gaps between existing data points
                for existing_time in existing_datetimes:
                    if current_time < existing_time:
                        gaps.append((current_time, existing_time))
                    current_time = existing_time + datetime.timedelta(minutes=1)
                
                # Check for gap after the last existing data point
                if current_time < end_time:
                    gaps.append((current_time, end_time))
                
                return gaps
                
        except Exception as e:
            print(f"Error getting existing data gaps: {e}")
            return [(start_time, end_time)]

    def _get_next_datetime(self, current_time: datetime.datetime) -> datetime.datetime:
        """
        Get the next datetime based on the timeframe.
        """
        if 'm' in self.timeframe:
            minutes = int(self.timeframe.replace('m', ''))
            return current_time + datetime.timedelta(minutes=minutes)
        elif 'h' in self.timeframe:
            hours = int(self.timeframe.replace('h', ''))
            return current_time + datetime.timedelta(hours=hours)
        else:
            return current_time + datetime.timedelta(minutes=1)

    def _fetch_missing_data(self, start_time: datetime.datetime, end_time: datetime.datetime) -> pd.DataFrame:
        """
        Fetch only missing data within the specified time range.
        """
        gaps = self._get_existing_data_gaps(start_time, end_time)
        all_data = []
        
        for gap_start, gap_end in gaps:
            print(f"Fetching missing data from {gap_start} to {gap_end}")
            gap_data = self._fetch_all_data(self.symbol, gap_start, gap_end, self.timeframe)
            if not gap_data.empty:
                all_data.append(gap_data)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def _fetch_all_data(self, symbol: str, start_time: datetime.datetime, 
                        end_time: datetime.datetime, timeframe: str = '1m') -> pd.DataFrame:
        """
        Fetch all data from Bybit API with smart batching to avoid timeouts.
        Args:
            symbol: Trading symbol
            start_time: Start time for data fetching
            end_time: End time for data fetching
            timeframe: Timeframe for data
        Returns:
            DataFrame with OHLC data
        """
        print(f"Fetching all data from {start_time} to {end_time}")
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            # Calculate batch size to avoid timeout
            batch_size_minutes = self._calculate_batch_size(current_start, end_time)
            current_end = min(current_start + datetime.timedelta(minutes=batch_size_minutes), end_time)
            
            print(f"  Fetching batch: {current_start} to {current_end}")
            
            batch_data = self._fetch_batch(symbol, current_start, current_end)
            if not batch_data.empty:
                all_data.append(batch_data)
            
            current_start = current_end
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.drop_duplicates(subset=['datetime'])
            df = df.sort_values('datetime')
            
            # Resample if timeframe is not 1m
            if timeframe != '1m':
                df.set_index('datetime', inplace=True)
                
                # Parse timeframe
                if 'm' in timeframe:
                    minutes = int(timeframe.replace('m', ''))
                elif 'h' in timeframe:
                    minutes = int(timeframe.replace('h', '')) * 60
                else:
                    minutes = 1  # default to 1 minute
                
                df = df.resample(f'{minutes}min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # Round volume to 2 decimal places after resampling
                df['volume'] = df['volume'].round(2)
                df.reset_index(inplace=True)
            
            print(f"Total records fetched: {len(df)}")
            return df
        else:
            return pd.DataFrame()
    
    def _calculate_batch_size(self, start_time: datetime.datetime, end_time: datetime.datetime) -> int:
        """
        Calculate optimal batch size based on time range to avoid timeout.
        Bybit API has rate limits, so we use smaller batches to be safe.
        """
        total_days = (end_time - start_time).days
        total_minutes = (end_time - start_time).total_seconds() / 60
        
        # For 1-minute data, we can fetch about 8 minutes worth of data safely
        # For longer timeframes, we can fetch more data
        if total_minutes <= self.timeout_limit:
            return total_minutes
        else:
            return min(7 * 60, total_minutes)
    
    def _fetch_batch(self, symbol: str, start_time: datetime.datetime, end_time: datetime.datetime) -> pd.DataFrame:
        """
        Fetch a single batch of data from Bybit.
        
        Args:
            symbol: Trading symbol
            start_time: Start time for this batch
            end_time: End time for this batch
            
        Returns:
            DataFrame with OHLC data for this batch
        """
        symbol_pair = symbol.upper() + 'USDT'
        
        try:
            # Convert timeframe to Bybit format
            interval_map = {
                '1m': '1',
                '5m': '5',
                '15m': '15',
                '30m': '30',
                '1h': '60',
                '4h': '240',
                '1d': 'D'
            }
            
            interval = interval_map.get(self.timeframe, '1')
            
            # Convert datetime to milliseconds timestamp for Bybit API
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            print(f"    API call params: symbol={symbol_pair}, interval={interval}, start={start_timestamp}, end={end_timestamp}")
            
            # Fetch kline data from Bybit
            klines = self.client.get_kline(
                category="linear",
                symbol=symbol_pair,
                interval=interval,
                start=start_timestamp,
                end=end_timestamp,
                limit=1000
            )
            
            print(f"    API response: {klines}")
            
            if not klines or 'result' not in klines or 'list' not in klines['result']:
                print(f"  No data returned from Bybit API: {klines}")
                return pd.DataFrame()
            
            raw_klines = klines['result']['list']
            
            if not raw_klines:
                print(f"  Empty klines list returned")
                return pd.DataFrame()
            
            # Convert to DataFrame
            chunk_df = pd.DataFrame(raw_klines, columns=[
                'startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnOver'
            ])
            
            # Convert numeric columns
            chunk_df[['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume']] = chunk_df[['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume']].astype(float)
            
            # Convert timestamp and rename columns to lowercase
            chunk_df['datetime'] = pd.to_datetime(pd.to_numeric(chunk_df['startTime']), unit='ms')
            chunk_df['open'] = chunk_df['openPrice']
            chunk_df['high'] = chunk_df['highPrice']
            chunk_df['low'] = chunk_df['lowPrice']
            chunk_df['close'] = chunk_df['closePrice']
            chunk_df['volume'] = chunk_df['volume']
            
            # Keep only OHLCV columns with lowercase names
            chunk_df = chunk_df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            # Round volume to 2 decimal places
            chunk_df['volume'] = chunk_df['volume'].round(2)
            return chunk_df
            
        except Exception as e:
            print(f"  Error fetching batch: {e}")
            return pd.DataFrame()
    
    def get_data_from_db(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None) -> pd.DataFrame:
        """
        Retrieve data from PostgreSQL database with proper datetime sorting.
        Args:
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
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
                    df['volume'] = df['volume'].round(2) # Round volume here as well
                return df
        except Exception as e:
            print(f"Error retrieving data from database: {e}")
            return pd.DataFrame()

    def fetch_data(
        self,
        start_time: datetime.datetime = None,
        end_time: datetime.datetime = None,
        drop_last_candle: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch Bybit data with database caching and smart batching.
        Args:
            start_time: Start time for data fetching
            end_time: End time for data fetching (defaults to now)
            drop_last_candle: Whether to drop the last candle (incomplete)
            use_cache: Whether to use database cache
        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        # Set default times if not provided
        if end_time is None:
            end_time = datetime.datetime.now()
        if start_time is None:
            start_time = end_time - datetime.timedelta(days=30)
        
        print(f"Fetching data from {start_time} to {end_time}, use_cache={use_cache}")
        
        if use_cache:
            new_data = self._fetch_missing_data(start_time, end_time)
            print(f"Fetched {len(new_data)} new records from API")
            if not new_data.empty:
                print(f"Inserting {len(new_data)} records to database")
                self._insert_data_to_db(new_data)
            df = self.get_data_from_db(start_time, end_time)
            print(f"Retrieved {len(df)} records from database")
        else:
            df = self._fetch_all_data(self.symbol, start_time, end_time, self.timeframe)
            print(f"Fetched {len(df)} records from API (no cache)")
            if not df.empty:
                print(f"Inserting {len(df)} records to database")
                self._insert_data_to_db(df)
        
        if drop_last_candle and len(df) > 0:
            df = df.iloc[:-1]
            print(f"Dropped last candle (incomplete)")
        print(f"Total records returned: {len(df)}")
        return df

    def interpolate_missing(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Interpolate missing values in the DataFrame.
        Args:
            df: DataFrame with OHLC data
            timeframe: Timeframe string (e.g., '1m', '5m')
        Returns:
            DataFrame with interpolated missing values
        """
        if df.empty:
            return df
        
        # Set datetime as index for resampling
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
        df_copy.set_index('datetime', inplace=True)
        
        # Parse timeframe to get frequency
        if 'm' in timeframe:
            minutes = int(timeframe.replace('m', ''))
            freq = f'{minutes}min'
        elif 'h' in timeframe:
            hours = int(timeframe.replace('h', ''))
            freq = f'{hours}H'
        else:
            freq = '1min'  # default
        
        # Resample to create regular intervals and interpolate
        df_resampled = df_copy.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Forward fill missing values
        df_filled = df_resampled.fillna(method='ffill')
        
        # Reset index to get datetime as column
        df_filled.reset_index(inplace=True)
        
        print(f"Interpolated missing values: {len(df)} -> {len(df_filled)} records")
        return df_filled


if __name__ == "__main__":
    api_key = os.getenv('bybit_api')
    api_secret = os.getenv('bybit_secret')
    symbol = 'BTC'
    timeframe = '1m'
    start = datetime.datetime(2025, 6, 1)
    fetcher = BybitDataFetcher(api_key, api_secret, symbol, timeframe)
    fetcher.fetch_data(start_time=start, 
                       end_time=None, 
                       drop_last_candle=True, 
                       use_cache=True) 