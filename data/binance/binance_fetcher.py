from binance.client import Client
import pandas as pd
import datetime
import time
import os
import sqlite3

class BinanceDataFetcher:
    def __init__(self, api_key: str, api_secret: str, symbol: str, timeframe: str):
        """
        Initialize Binance Data Fetcher.
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            symbol: Trading symbol (e.g., 'BTC')
            timeframe: Timeframe (e.g., '1m', '5m')
        """
        self.client = Client(api_key, api_secret)
        self.timeout_limit = 9 * 60
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.db_filename = f"Data_DB.db"
        # Go two steps back from this file's directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = os.path.join(project_root, self.db_filename)
        self.table_name = f"binance_{self.symbol.lower()}_{self.timeframe}"
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
            return
        
        df = df.copy()
        
        # Ensure datetime is properly formatted and sorted
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Remove duplicates before insertion
        df = df.drop_duplicates(subset=['datetime'])
        
        # Process in batches to avoid "too many SQL variables" error
        batch_size = 1000  # SQLite limit is 999, so we use 1000 to be safe
        total_inserted = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Process data in batches
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i + batch_size]
                    
                    # Get existing datetime values for this batch
                    datetime_values = batch_df['datetime'].tolist()
                    placeholders = ','.join(['?' for _ in datetime_values])
                    
                    cursor.execute(f'''
                        SELECT datetime FROM {self.table_name} 
                        WHERE datetime IN ({placeholders})
                    ''', datetime_values)
                    
                    existing_datetimes = {row[0] for row in cursor.fetchall()}
                    
                    # Filter out data that already exists
                    new_data = batch_df[~batch_df['datetime'].isin(existing_datetimes)]
                    
                    if not new_data.empty:
                        # Insert only new data
                        data_to_insert = [tuple(row) for row in new_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].values]
                        
                        cursor.executemany(f'''
                            INSERT INTO {self.table_name} (datetime, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', data_to_insert)
                        
                        total_inserted += len(data_to_insert)
                        print(f"Inserted batch {i//batch_size + 1}: {len(data_to_insert)} records")
                
                conn.commit()
                print(f"Total inserted: {total_inserted} new records into {self.db_path} ({self.table_name})")
                
                # Verify data integrity by checking for any remaining gaps
                if total_inserted > 0:
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
                
                existing_times = [pd.to_datetime(row[0]) for row in cursor.fetchall()]
                
                if not existing_times:
                    # No data exists in range, return the entire range as a gap
                    return [(start_time, end_time)]
                
                gaps = []
                
                # Check for gap before first existing data point
                if existing_times[0] > start_time:
                    gaps.append((start_time, existing_times[0]))
                
                # Check for gaps between existing data points
                for i in range(len(existing_times) - 1):
                    current_time = existing_times[i]
                    next_time = existing_times[i + 1]
                    
                    # Calculate expected next time based on timeframe
                    expected_next = self._get_next_datetime(current_time)
                    
                    # If there's a gap, add it to gaps list
                    if next_time > expected_next:
                        gaps.append((expected_next, next_time))
                
                # Check for gap after last existing data point
                if existing_times[-1] < end_time:
                    last_expected = self._get_next_datetime(existing_times[-1])
                    if last_expected <= end_time:
                        gaps.append((last_expected, end_time))
                
                return gaps
                
        except Exception as e:
            print(f"Error getting existing data gaps: {e}")
            return [(start_time, end_time)]
    
    def _get_next_datetime(self, current_time: datetime.datetime) -> datetime.datetime:
        """
        Get the next expected datetime based on the timeframe.
        """
        if 'm' in self.timeframe:
            minutes = int(self.timeframe.replace('m', ''))
        elif 'h' in self.timeframe:
            minutes = int(self.timeframe.replace('h', '')) * 60
        else:
            minutes = 1
        
        return current_time + datetime.timedelta(minutes=minutes)

    def _fetch_missing_data(self, start_time: datetime.datetime, end_time: datetime.datetime) -> pd.DataFrame:
        """
        Fetch only missing data from Binance API by identifying gaps in existing data.
        Args:
            start_time: Start time for data fetching
            end_time: End time for data fetching
        Returns:
            DataFrame with missing data
        """
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
        
        # Combine all missing data
        df = pd.concat(all_missing_data, ignore_index=True)
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
        
        return df
    
    def _fetch_all_data(self, symbol: str, start_time: datetime.datetime, 
                        end_time: datetime.datetime, timeframe: str = '1m') -> pd.DataFrame:
        """
        Fetch all data from Binance API (original fetch logic).
        
        Args:
            symbol: Trading symbol
            start_time: Start time for data fetching
            end_time: End time for data fetching
            timeframe: Timeframe of the data
        Returns:
            DataFrame with OHLC data
        """
        symbol_pair = symbol.upper() + 'USDT'
        print(f"Fetching {symbol_pair} data from {start_time} to {end_time}")
        print(f"Using smart batching to avoid 10-minute timeout")
        
        all_data = []
        current_start = start_time
        batch_count = 0
        
        while current_start < end_time:
            batch_count += 1
            
            # Calculate optimal batch size based on remaining time
            remaining_time = end_time - current_start
            batch_minutes = self._calculate_batch_size(current_start, end_time)
            
            # Convert minutes to timedelta
            batch_duration = datetime.timedelta(minutes=batch_minutes)
            current_end = min(current_start + batch_duration, end_time)
            
            print(f"Fetching batch {batch_count}: {current_start} to {current_end} ({batch_minutes:.0f} minutes)")
            
            # Fetch batch data
            batch_df = self._fetch_batch(symbol, current_start, current_end)
            
            if not batch_df.empty:
                all_data.append(batch_df)
                print(f"  Retrieved {len(batch_df)} records")
            else:
                print(f"  No data for this batch")
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
            current_start = current_end
        
        if not all_data:
            print("No data retrieved")
            return pd.DataFrame()
        
        # Combine all batches efficiently
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
        
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
    
    def _calculate_batch_size(self, start_time: datetime.datetime, end_time: datetime.datetime) -> int:
        """
        Calculate optimal batch size based on time range to avoid 10-minute timeout.
        Binance API has a 10-minute timeout, so we use 9 minutes to be safe.
        """
        total_days = (end_time - start_time).days
        total_minutes = (end_time - start_time).total_seconds() / 60
        
        # For 1-minute data, we can fetch about 9 minutes worth of data safely
        # For longer timeframes, we can fetch more data
        if total_minutes <= self.timeout_limit:
            return total_minutes
        else:
            return min(8 * 60, total_minutes)
    
    def _fetch_batch(self, symbol: str, start_time: datetime.datetime, end_time: datetime.datetime) -> pd.DataFrame:
        """
        Fetch a single batch of data.
        
        Args:
            symbol: Trading symbol
            start_time: Start time for this batch
            end_time: End time for this batch
            
        Returns:
            DataFrame with OHLC data for this batch
        """
        symbol_pair = symbol.upper() + 'USDT'
        
        try:
            raw_klines = self.client.get_historical_klines(
                symbol=symbol_pair,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                start_str=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_str=end_time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            if not raw_klines:
                return pd.DataFrame()
            
            # Convert to DataFrame efficiently
            chunk_df = pd.DataFrame(raw_klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            
            # Convert numeric columns efficiently
            chunk_df[['Open', 'High', 'Low', 'Close', 'Volume']] = chunk_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
            # Convert timestamp and rename columns to lowercase
            chunk_df['datetime'] = pd.to_datetime(chunk_df['Open Time'], unit='ms', )
            chunk_df['open'] = chunk_df['Open']
            chunk_df['high'] = chunk_df['High']
            chunk_df['low'] = chunk_df['Low']
            chunk_df['close'] = chunk_df['Close']
            chunk_df['volume'] = chunk_df['Volume']
            
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
        Fetch Binance data with database caching and smart batching.
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

