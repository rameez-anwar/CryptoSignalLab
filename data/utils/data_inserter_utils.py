import pandas as pd
import logging
from sqlalchemy import create_engine, Column, Float, DateTime, MetaData, Table, inspect
from sqlalchemy.exc import ProgrammingError, OperationalError
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
                    conn.execute(f"CREATE SCHEMA {schema_name}")
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
        Save DataFrame to PostgreSQL with bulk insert.
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