from pybit.unified_trading import HTTP
import pandas as pd
import datetime
import time
import os
import sqlite3

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
            testnet=False,
            api_key=api_key,
            api_secret=api_secret
        )
        self.timeout_limit = 9 * 60
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.db_filename = f"Data_DB.db"
        # Go two steps back from this file's directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = os.path.join(project_root, self.db_filename)
        self.table_name = f"bybit_{self.symbol.lower()}_{self.timeframe}"
        self._init_database()

    def _init_database(self):
        """
        Initialize SQLite database with 6 columns: datetime, open, high, low, close, volume.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        datetime DATETIME PRIMARY KEY,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL
                    )
                ''')
                cursor.execute(f'''CREATE INDEX IF NOT EXISTS idx_datetime ON {self.table_name}(datetime)''')
                conn.commit()
                print(f"Database initialized: {self.db_path}, table: {self.table_name}")
        except Exception as e:
            print(f"Error initializing database: {e}")

    def _insert_data_to_db(self, df: pd.DataFrame):
        """
        Insert DataFrame data into SQLite database with proper duplicate handling and sorting.
        Args:
            df: DataFrame with columns: datetime, open, high, low, close, volume
        """
        if df.empty:
            print("No data to insert (empty DataFrame)")
            return
        
        print(f"Attempting to insert {len(df)} records to database table: {self.table_name}")
        df = df.copy()
        
        # Ensure datetime is properly formatted and sorted
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Remove duplicates before insertion
        df = df.drop_duplicates(subset=['datetime'])
        print(f"After removing duplicates: {len(df)} records to insert")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, check for existing data to avoid conflicts
                cursor = conn.cursor()
                
                # Get existing datetime values for the data we're about to insert
                datetime_values = df['datetime'].tolist()
                placeholders = ','.join(['?' for _ in datetime_values])
                
                cursor.execute(f'''
                    SELECT datetime FROM {self.table_name} 
                    WHERE datetime IN ({placeholders})
                ''', datetime_values)
                
                existing_datetimes = {row[0] for row in cursor.fetchall()}
                print(f"Found {len(existing_datetimes)} existing records")
                
                # Filter out data that already exists
                new_data = df[~df['datetime'].isin(existing_datetimes)]
                print(f"After filtering existing data: {len(new_data)} new records to insert")
                
                if new_data.empty:
                    print("All data already exists in database")
                    return
                
                # Insert only new data
                data_to_insert = [tuple(row) for row in new_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].values]
                
                cursor.executemany(f'''
                    INSERT INTO {self.table_name} (datetime, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', data_to_insert)
                
                conn.commit()
                print(f"Successfully inserted {len(data_to_insert)} new records into {self.db_path} ({self.table_name})")
                
                # Verify data integrity by checking for any remaining gaps
                self._verify_data_integrity()
                
        except Exception as e:
            print(f"Error inserting data to database: {e}")
            import traceback
            traceback.print_exc()
    
    def _verify_data_integrity(self):
        """
        Verify that data in the database is properly sorted and has no duplicates.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for duplicates
                cursor.execute(f'''
                    SELECT datetime, COUNT(*) as count
                    FROM {self.table_name}
                    GROUP BY datetime
                    HAVING COUNT(*) > 1
                ''')
                
                duplicates = cursor.fetchall()
                if duplicates:
                    print(f"Warning: Found {len(duplicates)} duplicate datetime entries")
                    
                    # Remove duplicates keeping only the first occurrence
                    for dup_datetime, count in duplicates:
                        cursor.execute(f'''
                            DELETE FROM {self.table_name}
                            WHERE datetime = ? AND rowid NOT IN (
                                SELECT MIN(rowid) FROM {self.table_name} WHERE datetime = ?
                            )
                        ''', (dup_datetime, dup_datetime))
                    
                    conn.commit()
                    print(f"Removed {sum(count-1 for _, count in duplicates)} duplicate entries")
                
                # Verify sorting
                cursor.execute(f'''
                    SELECT datetime FROM {self.table_name} ORDER BY datetime ASC
                ''')
                
                all_datetimes = [row[0] for row in cursor.fetchall()]
                sorted_datetimes = sorted(all_datetimes)
                
                if all_datetimes != sorted_datetimes:
                    print("Warning: Data is not properly sorted, fixing...")
                    # Recreate table with proper sorting
                    cursor.execute(f'''
                        CREATE TABLE {self.table_name}_temp AS
                        SELECT * FROM {self.table_name} ORDER BY datetime ASC
                    ''')
                    cursor.execute(f'DROP TABLE {self.table_name}')
                    cursor.execute(f'ALTER TABLE {self.table_name}_temp RENAME TO {self.table_name}')
                    conn.commit()
                    print("Data reordered successfully")
                
        except Exception as e:
            print(f"Error verifying data integrity: {e}")

    def _get_existing_data_range(self) -> tuple:
        """
        Get the date range of existing data in the table.
        Returns:
            Tuple of (min_date, max_date) or (None, None) if no data exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    SELECT MIN(datetime), MAX(datetime)
                    FROM {self.table_name}
                ''')
                result = cursor.fetchone()
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all existing data points in the range, sorted by datetime
                cursor.execute(f'''
                    SELECT datetime 
                    FROM {self.table_name}
                    WHERE datetime >= ? AND datetime <= ?
                    ORDER BY datetime ASC
                ''', (start_time.strftime('%Y-%m-%d %H:%M:%S'), 
                      end_time.strftime('%Y-%m-%d %H:%M:%S')))
                
                existing_datetimes = [pd.to_datetime(row[0]) for row in cursor.fetchall()]
                
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
        if self.timeframe == '1m':
            return current_time + datetime.timedelta(minutes=1)
        elif self.timeframe == '5m':
            return current_time + datetime.timedelta(minutes=5)
        elif self.timeframe == '15m':
            return current_time + datetime.timedelta(minutes=15)
        elif self.timeframe == '1h':
            return current_time + datetime.timedelta(hours=1)
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
        Fetch all data for the given symbol and time range using batching.
        """
        print(f"Fetching {symbol} data from {start_time} to {end_time}")
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            batch_size = self._calculate_batch_size(current_start, end_time)
            current_end = min(current_start + datetime.timedelta(minutes=batch_size), end_time)
            
            print(f"  Fetching batch: {current_start} to {current_end}")
            batch_data = self._fetch_batch(symbol, current_start, current_end)
            
            if not batch_data.empty:
                all_data.append(batch_data)
            
            current_start = current_end
            
            # Rate limiting
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
                category="spot",
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
        Retrieve data from SQLite database with proper datetime sorting.
        Args:
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f'SELECT datetime, open, high, low, close, volume FROM {self.table_name}'
                params = []
                if start_time:
                    query += ' WHERE datetime >= ?'
                    params.append(start_time.strftime('%Y-%m-%d %H:%M:%S'))
                if end_time:
                    if params:
                        query += ' AND datetime <= ?'
                    else:
                        query += ' WHERE datetime <= ?'
                    params.append(end_time.strftime('%Y-%m-%d %H:%M:%S'))
                query += ' ORDER BY datetime ASC'
                df = pd.read_sql_query(query, conn, params=params)
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
        Fill missing datetime values using interpolation.
        """
        if df.empty:
            return df
        df.set_index('datetime', inplace=True)
        # Convert timeframe to pandas frequency format
        if timeframe == '1m':
            freq = '1min'
        elif timeframe == '5m':
            freq = '5min'
        elif timeframe == '15m':
            freq = '15min'
        elif timeframe == '1h':
            freq = '1H'
        else:
            freq = '1min'  # default
        df = df.asfreq(freq)
        df.interpolate(method='linear', inplace=True)
        df.reset_index(inplace=True)
        return df

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