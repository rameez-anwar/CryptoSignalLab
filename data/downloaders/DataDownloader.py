import pandas as pd
import datetime
import os
import sys
import time
from sqlalchemy import text

from data.binance.binance_fetcher import BinanceDataFetcher
from data.bybit.bybit_fetcher import BybitDataFetcher
from data.utils.db_utils import get_pg_engine
from data.utils.data_inserter_utils import DataInserter


class DataDownloader:
    def __init__(self, exchange, symbol, time_horizon):
        """
        Initialize DataDownloader with exchange, symbol, and time_horizon
        """
        self.exchange = exchange.lower()
        self.symbol = symbol.lower()
        self.time_horizon = time_horizon.lower()
        
        # Initialize PostgreSQL connection
        self.engine = get_pg_engine()
        self.data_inserter = DataInserter(self.engine)
        
        # Schema and table names for PostgreSQL
        self.schema = f"{self.exchange}_data"
        self.table_name = f"{self.symbol}_1m"  # Always use 1m as base
        
        # Parse time horizon to minutes
        self.time_horizon_minutes = self._parse_time_horizon(time_horizon)
        
        # Initialize API credentials (will be set when needed)
        self.api_key = None
        self.api_secret = None
        
        print(f"DataDownloader initialized:")
        print(f"  Exchange: {self.exchange}")
        print(f"  Symbol: {self.symbol}")
        print(f"  Time Horizon: {self.time_horizon} ({self.time_horizon_minutes} minutes)")
        print(f"  Database: PostgreSQL")
        print(f"  Schema: {self.schema}")
        print(f"  Table: {self.table_name}")
        print()

    def _parse_time_horizon(self, time_horizon):
        """Parse time horizon string to minutes"""
        time_horizon = time_horizon.lower()
        
        if 'h' in time_horizon:
            hours = int(time_horizon.replace('h', ''))
            return hours * 60
        elif 'm' in time_horizon:
            return int(time_horizon.replace('m', ''))
        elif 'd' in time_horizon:
            days = int(time_horizon.replace('d', ''))
            return days * 24 * 60
        else:
            try:
                return int(time_horizon)
            except ValueError:
                return 1

    def _check_data_availability(self):
        """
        Check if 1m data exists in the database for the requested symbol
        Returns:
            bool: True if data exists, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                # Check if schema exists
                query = text(f"""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.schemata 
                        WHERE schema_name = '{self.schema}'
                    )
                """)
                result = conn.execute(query).fetchone()
                if not result[0]:
                    return False
                
                # Check if table exists
                query = text(f"""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = '{self.schema}' 
                        AND table_name = '{self.table_name}'
                    )
                """)
                result = conn.execute(query).fetchone()
                if not result[0]:
                    return False
                
                # Check if table has data
                query = text(f"SELECT COUNT(*) FROM {self.schema}.{self.table_name}")
                result = conn.execute(query).fetchone()
                return result[0] > 0
                
        except Exception as e:
            print(f"Error checking data availability: {e}")
            return False

    def _get_existing_data_range(self):
        """
        Get the date range of existing data in the database
        Returns:
            tuple: (min_date, max_date) or (None, None) if no data exists
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

    def _get_missing_data_periods(self, start_time, end_time):
        """
        Identify missing data periods between start_time and end_time
        Returns:
            list: List of tuples (gap_start, gap_end) representing missing data periods
        """
        try:
            with self.engine.connect() as conn:
                # Get all existing data points in the range
                query = text(f"""
                    SELECT datetime 
                    FROM {self.schema}.{self.table_name}
                    WHERE datetime >= :start AND datetime <= :end
                    ORDER BY datetime ASC
                """)
                result = conn.execute(query, {'start': start_time, 'end': end_time})
                existing_times = [pd.to_datetime(row[0]) for row in result.fetchall()]
                
                if not existing_times:
                    return [(start_time, end_time)]
                
                gaps = []
                
                # Check for gap before first existing data point
                if existing_times[0] > start_time:
                    gaps.append((start_time, existing_times[0]))
                
                # Check for gaps between existing data points
                for i in range(len(existing_times) - 1):
                    current_time = existing_times[i]
                    next_time = existing_times[i + 1]
                    expected_next = current_time + datetime.timedelta(minutes=1)
                    
                    if next_time > expected_next:
                        gaps.append((expected_next, next_time))
                
                # Check for gap after last existing data point
                if existing_times[-1] < end_time:
                    last_expected = existing_times[-1] + datetime.timedelta(minutes=1)
                    if last_expected <= end_time:
                        gaps.append((last_expected, end_time))
                
                return gaps
                
        except Exception as e:
            print(f"Error getting missing data periods: {e}")
            return [(start_time, end_time)]

    def _download_missing_data(self, start_time=None, end_time=None):
        """
        Download missing 1m data from the exchange API
        """
        print(f"Checking for missing data for {self.symbol.upper()} from {self.exchange.upper()}...")
        
        # Get API credentials from environment based on exchange
        if self.exchange == "binance":
            self.api_key = os.getenv('API_KEY')
            self.api_secret = os.getenv('API_SECRET')
        elif self.exchange == "bybit":
            self.api_key = os.getenv('bybit_api')
            self.api_secret = os.getenv('bybit_secret')
        else:
            print(f"ERROR: Unsupported exchange '{self.exchange}'. Supported exchanges: binance, bybit")
            return False
            
        if not self.api_key or not self.api_secret:
            print(f"ERROR: API credentials are required for downloading data from {self.exchange}")
            if self.exchange == "binance":
                print("Set API_KEY and API_SECRET environment variables for Binance")
            elif self.exchange == "bybit":
                print("Set bybit_api and bybit_secret environment variables for Bybit")
            return False

        try:
            # Set default time range if not provided
            if end_time is None:
                end_time = datetime.datetime.now()
            if start_time is None:
                start_time = end_time - datetime.timedelta(days=30)  # Default to last 30 days

            print(f"Checking data availability from {start_time} to {end_time}")

            # Check if we have existing data
            existing_min, existing_max = self._get_existing_data_range()
            
            if existing_min is None or existing_max is None:
                # No existing data - fetch from start
                print(f"No existing data found for {self.symbol.upper()}. Fetching from start...")
                return self._fetch_new_data(start_time, end_time)
            else:
                # We have existing data - check for gaps and fetch missing data
                print(f"Existing data found: {existing_min} to {existing_max}")
                
                # Identify missing data periods
                missing_periods = self._get_missing_data_periods(start_time, end_time)
                
                # Skip gaps less than 1 minute
                filtered_periods = []
                for gap_start, gap_end in missing_periods:
                    if (gap_end - gap_start) >= datetime.timedelta(minutes=1):
                        filtered_periods.append((gap_start, gap_end))
                
                if not filtered_periods:
                    print("All requested data already exists in database or only trivial gaps found.")
                    return True
                
                print(f"Found {len(filtered_periods)} missing data periods:")
                for i, (gap_start, gap_end) in enumerate(filtered_periods, 1):
                    print(f"  {i}. {gap_start} to {gap_end}")
                
                # Fetch missing data for each period
                success = True
                for gap_start, gap_end in filtered_periods:
                    print(f"\nFetching missing data: {gap_start} to {gap_end}")
                    fetch_result = self._fetch_new_data(gap_start, gap_end)
                    if fetch_result is False:
                        success = False
                        print(f"Failed to fetch data for period {gap_start} to {gap_end}")
                
                return success

        except Exception as e:
            print(f"Error downloading missing data: {e}")
            return False

    def _fetch_new_data(self, start_time, end_time):
        """
        Fetch new data from the exchange API
        """
        try:
            # Initialize the appropriate fetcher
            if self.exchange == "binance":
                fetcher = BinanceDataFetcher(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    symbol=self.symbol.upper(),
                    timeframe='1m'
                )
            elif self.exchange == "bybit":
                fetcher = BybitDataFetcher(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    symbol=self.symbol.upper(),
                    timeframe='1m'
                )
            else:
                print(f"ERROR: Unsupported exchange '{self.exchange}'")
                return False

            # Fetch data with use_cache=False to force fresh download
            df = fetcher.fetch_data(
                start_time=start_time,
                end_time=end_time,
                drop_last_candle=True,
                use_cache=False  # Force fresh download
            )
            
            if df is not None and not df.empty:
                print(f"Successfully downloaded {len(df)} records for {self.symbol.upper()}")
                return True
            else:
                print(f"No data downloaded for {self.symbol.upper()} (gap {start_time} to {end_time}) - skipping.")
                return None  # Not a failure, just nothing to do
                
        except Exception as e:
            print(f"Error fetching new data: {e}")
            return False

    def fetch_data(self, start_time=None, end_time=None, num_records=None, auto_download=True):
        """
        Fetch data from database and return (1_minute_data, time_horizon_data)
        If data is not available and auto_download is True, automatically download it
        Args:
            start_time: Start time for data fetching (optional)
            end_time: End time for data fetching (optional)
            num_records: Number of 1-minute records to fetch (optional)
            auto_download: Whether to automatically download missing data (default: True)
        """
        # Check if 1m data is available
        if not self._check_data_availability():
            if auto_download:
                print(f"1m data for {self.symbol.upper()} not found in database")
                print(f"Attempting to download 1m data automatically...")
                if self._download_missing_data(start_time, end_time):
                    print(f"1m data downloaded successfully! Now fetching from database...")
                    time.sleep(1)
                else:
                    print(f"Failed to download 1m data for {self.symbol.upper()}")
                    return None, None
            else:
                print(f"No 1m data found for {self.symbol.upper()} in database")
                print(f"Use auto_download=True to automatically download missing data")
                return None, None
        else:
            # Data exists, check for missing data and download if needed
            if auto_download:
                print(f"1m data for {self.symbol.upper()} found in database")
                print(f"Checking for missing data and downloading if needed...")
                if self._download_missing_data(start_time, end_time):
                    print(f"Data check and update completed successfully!")
                    time.sleep(1)
                else:
                    print(f"Failed to update missing data for {self.symbol.upper()}")
                    # Continue anyway as we have some data
        
        try:
            # Connect to database and query the 1m table
            query = f"SELECT datetime, open, high, low, close, volume FROM {self.schema}.{self.table_name}"
            params = {}
            
            if start_time or end_time:
                conditions = []
                if start_time:
                    conditions.append("datetime >= :start")
                    params['start'] = start_time
                if end_time:
                    conditions.append("datetime <= :end")
                    params['end'] = end_time
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY datetime DESC"  # Get latest records first
            
            # Fetch 1-minute data
            with self.engine.connect() as conn:
                df_1min = pd.read_sql_query(text(query), conn, params=params)
            
            if df_1min.empty:
                print("No 1m data found in database")
                return None, None
            
            # Convert datetime
            df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])
            df_1min.set_index('datetime', inplace=True)
            # Sort by datetime (ascending)
            df_1min = df_1min.sort_index()
            # If num_records specified, take the latest records
            if num_records is not None:
                df_1min = df_1min.tail(num_records)
                print(f"Fetched {len(df_1min)} 1-minute records (latest {num_records})")
            else:
                print(f"Fetched {len(df_1min)} 1-minute records")
            print(f"Date range: {df_1min.index.min()} to {df_1min.index.max()}")
            # Aggregate to time horizon if needed
            if self.time_horizon_minutes == 1:
                df_horizon = df_1min.copy()
            else:
                df_horizon = self._aggregate_data_precise(df_1min, self.time_horizon_minutes)
            print(f"Created {len(df_horizon)} {self.time_horizon} records")
            print()
            return df_1min, df_horizon
        except Exception as e:
            print(f"ERROR: {e}")
            return None, None

    def _aggregate_data_precise(self, df, minutes):
        """
        Aggregate data precisely to get exact number of records
        """
        if df.empty or minutes == 1:
            return df
        
        # Calculate how many complete periods we can make
        total_minutes = len(df)
        complete_periods = total_minutes // minutes
        
        if complete_periods == 0:
            return pd.DataFrame()
        
        # Take only the data that fits complete periods (from the beginning)
        records_needed = complete_periods * minutes
        df_trimmed = df.head(records_needed)
        
        # Group by periods manually
        aggregated_data = []
        
        for i in range(0, len(df_trimmed), minutes):
            period_data = df_trimmed.iloc[i:i+minutes]
            if len(period_data) == minutes:  # Only complete periods
                aggregated_row = {
                    'open': period_data.iloc[0]['open'],
                    'high': period_data['high'].max(),
                    'low': period_data['low'].min(),
                    'close': period_data.iloc[-1]['close'],
                    'volume': period_data['volume'].sum()
                }
                aggregated_data.append(aggregated_row)
        
        if aggregated_data:
            # Create DataFrame with proper datetime index
            result_df = pd.DataFrame(aggregated_data)
            # Use the start time of each period as index
            start_times = [df_trimmed.index[i] for i in range(0, len(df_trimmed), minutes) if i + minutes <= len(df_trimmed)]
            result_df.index = start_times[:len(aggregated_data)]
            return result_df
        else:
            return pd.DataFrame()
    
    def display_data(self, df_1min, df_horizon):
        """Display all data in terminal"""
        if df_1min is not None and not df_1min.empty:
            print("=== ALL 1-MINUTE DATA ===")
            print(df_1min)
            print()
        
        if df_horizon is not None and not df_horizon.empty:
            print(f"=== ALL {self.time_horizon.upper()} DATA ===")
            print(df_horizon)
            print()
            

if __name__ == "__main__":
    # Set your parameters here
    exchange = "binance"  # or "bybit"
    symbol = "btc"
    time_horizon = "1h"
    
    downloader = DataDownloader(exchange, symbol, time_horizon)
    
    print(f"Fetching ALL records from database for {time_horizon}")
    print("=" * 60)
    
    # Fetch all records from database (with auto-download enabled)
    df_1min, df_horizon = downloader.fetch_data(auto_download=True)
    
    if df_1min is not None and df_horizon is not None:
        downloader.display_data(df_1min, df_horizon)
    else:
        print("No data found.") 