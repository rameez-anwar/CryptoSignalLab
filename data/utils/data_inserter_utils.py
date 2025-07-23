import pandas as pd
import logging
from sqlalchemy import create_engine, Column, Float, DateTime, MetaData, Table, inspect, text
from sqlalchemy.exc import ProgrammingError, OperationalError, IntegrityError
import time

class DataInserter:
    """
    Utility class for efficient data insertion into PostgreSQL.
    """
    def __init__(self, engine):
        """
        Initialize DataInserter with a SQLAlchemy engine.
        Args:
            engine: SQLAlchemy engine for PostgreSQL connection
        """
        self.engine = engine
        self.max_retries = 3

    def _schema_exists(self, schema_name: str):
        """
        Check if schema exists, create it if it doesn't.
        Args:
            schema_name: Name of the schema to check/create
        """
        try:
            inspector = inspect(self.engine)
            if not inspector.has_schema(schema_name):
                with self.engine.connect() as conn:
                    conn.execute(text(f"CREATE SCHEMA {schema_name}"))
                    conn.commit()
            else:
                pass # No logging here
        except Exception as e:
            raise

    def _create_table(self, table_name: str, schema: str) -> Table:
        """
        Create table if it doesn't exist.
        Args:
            table_name: Name of the table
            schema: Schema name
        Returns:
            SQLAlchemy Table object
        """
        try:
            metadata = MetaData(schema=schema)
            table = Table(
                table_name, metadata,
                Column('datetime', DateTime, primary_key=True),
                Column('open', Float, nullable=False),
                Column('high', Float, nullable=False),
                Column('low', Float, nullable=False),
                Column('close', Float, nullable=False),
                Column('volume', Float, nullable=False),
                schema=schema
            )
            metadata.create_all(self.engine, checkfirst=True)
            return table
        except Exception as e:
            raise

    def save_dataframe(self, df: pd.DataFrame, exchange: str, symbol: str, interval: str):
        """
        Save DataFrame to PostgreSQL with upsert functionality to handle duplicates.
        Args:
            df: DataFrame with columns: datetime, open, high, low, close, volume
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading symbol (e.g., 'BTC')
            interval: Timeframe (e.g., '1m')
        """
        if df.empty:
            return

        table_name = f"{symbol.lower()}_{interval}"
        schema = f"{exchange.lower()}_data"
        self._schema_exists(schema)
        self._create_table(table_name, schema)

        try:
            start_time = time.time()
            df_to_save = df.copy()
            df_to_save['datetime'] = pd.to_datetime(df_to_save['datetime'])
            df_to_save = df_to_save.sort_values('datetime').drop_duplicates(subset=['datetime'])

            # Use upsert functionality to handle duplicates
            self._upsert_dataframe(df_to_save, schema, table_name)
            
            print(f"Successfully processed {len(df_to_save)} records for {schema}.{table_name}")
            
        except Exception as e:
            print(f"Error saving dataframe: {e}")
            raise

    def _upsert_dataframe(self, df: pd.DataFrame, schema: str, table_name: str):
        """
        Insert DataFrame with upsert functionality using PostgreSQL's ON CONFLICT.
        Args:
            df: DataFrame to insert
            schema: Database schema
            table_name: Table name
        """
        if df.empty:
            return

        # Prepare data for insertion
        data_to_insert = []
        for _, row in df.iterrows():
            data_to_insert.append({
                'datetime': row['datetime'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        # Use PostgreSQL's ON CONFLICT for upsert
        upsert_query = text(f"""
            INSERT INTO {schema}.{table_name} (datetime, open, high, low, close, volume)
            VALUES (:datetime, :open, :high, :low, :close, :volume)
            ON CONFLICT (datetime) 
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """)

        try:
            with self.engine.connect() as conn:
                # Execute in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, len(data_to_insert), batch_size):
                    batch = data_to_insert[i:i + batch_size]
                    conn.execute(upsert_query, batch)
                    conn.commit()
                    
        except Exception as e:
            print(f"Error in upsert operation: {e}")
            raise

    def save_dataframe_legacy(self, df: pd.DataFrame, exchange: str, symbol: str, interval: str):
        """
        Legacy save method using to_sql (kept for backward compatibility).
        Args:
            df: DataFrame with columns: datetime, open, high, low, close, volume
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading symbol (e.g., 'BTC')
            interval: Timeframe (e.g., '1m')
        """
        if df.empty:
            return

        table_name = f"{symbol.lower()}_{interval}"
        schema = f"{exchange.lower()}_data"
        self._schema_exists(schema)
        self._create_table(table_name, schema)

        try:
            start_time = time.time()
            df_to_save = df.copy()
            df_to_save['datetime'] = pd.to_datetime(df_to_save['datetime'])
            df_to_save = df_to_save.sort_values('datetime').drop_duplicates(subset=['datetime'])

            for attempt in range(self.max_retries):
                try:
                    df_to_save.to_sql(
                        table_name,
                        self.engine,
                        schema=schema,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                    return
                except OperationalError as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(1)
        except Exception as e:
            raise