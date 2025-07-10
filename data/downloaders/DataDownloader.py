import sqlite3
import pandas as pd
import datetime
import os


class DataDownloader:
    def __init__(self, exchange, symbol, time_horizon):
        """
        Initialize DataDownloader with exchange, symbol, and time_horizon
        """
        self.exchange = exchange.lower()
        self.symbol = symbol.lower()
        self.time_horizon = time_horizon.lower()
        
        # Database filename (1-minute data is the base)
        self.db_filename = f"Data_DB.db"
        
        # Get project root path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = os.path.join(project_root, self.db_filename)
        
        # Table name
        self.table_name = f"{self.exchange}_{self.symbol}_1m"
        
        # Parse time horizon to minutes
        self.time_horizon_minutes = self._parse_time_horizon(time_horizon)
        
        print(f"DataDownloader initialized:")
        print(f"  Exchange: {self.exchange}")
        print(f"  Symbol: {self.symbol}")
        print(f"  Time Horizon: {self.time_horizon} ({self.time_horizon_minutes} minutes)")
        print(f"  Database: {self.db_path}")
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
    
    def fetch_data(self, start_time=None, end_time=None, num_records=None):
        """
        Fetch data from database and return (1_minute_data, time_horizon_data)
        
        Args:
            start_time: Start time for data fetching (optional)
            end_time: End time for data fetching (optional)
            num_records: Number of 1-minute records to fetch (optional)
        """
        if not os.path.exists(self.db_path):
            print(f"ERROR: Database not found: {self.db_path}")
            return None, None
        
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Build query
            query = f"SELECT datetime, open, high, low, close, volume FROM {self.table_name}"
            params = []
            
            if start_time or end_time:
                conditions = []
                if start_time:
                    conditions.append("datetime >= ?")
                    params.append(start_time.strftime('%Y-%m-%d %H:%M:%S'))
                if end_time:
                    conditions.append("datetime <= ?")
                    params.append(end_time.strftime('%Y-%m-%d %H:%M:%S'))
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY datetime DESC"  # Get latest records first
            
            # Fetch 1-minute data
            df_1min = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df_1min.empty:
                print("No data found in database")
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
                # Use precise aggregation
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
    exchange = "binance"
    symbol = "btc"
    time_horizon = "4m"
    
    downloader = DataDownloader(exchange, symbol, time_horizon)
    
    print(f"Fetching ALL records from database for {time_horizon}")
    print("=" * 60)
    
    # Fetch all records from database
    df_1min, df_horizon = downloader.fetch_data()
    
    if df_1min is not None and df_horizon is not None:
        downloader.display_data(df_1min, df_horizon)
    else:
        print("No data found.") 