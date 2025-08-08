from data.utils.db_utils import get_pg_engine
from stats.generate_stats import generate_stats_from_backtest
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sqlalchemy import VARCHAR, DECIMAL, INTEGER, TIMESTAMP

def create_stats_schema_and_table(cursor):
    """Create stats schema and table if they don't exist"""
    try:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS stats")
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stats.strategy_stats (
            strategy_name VARCHAR(255),
            total_return DECIMAL(10,4),
            daily_return DECIMAL(10,4),
            weekly_return DECIMAL(10,4),
            monthly_return DECIMAL(10,4),
            yearly_return DECIMAL(10,4),
            cagr DECIMAL(10,4),
            sharpe_ratio DECIMAL(10,4),
            sortino_ratio DECIMAL(10,4),
            calmar_ratio DECIMAL(10,4),
            alpha DECIMAL(10,4),
            beta DECIMAL(10,4),
            r_squared DECIMAL(10,4),
            information_ratio DECIMAL(10,4),
            treynor_ratio DECIMAL(10,4),
            profit_factor DECIMAL(10,4),
            omega_ratio DECIMAL(10,4),
            gain_to_pain_ratio DECIMAL(10,4),
            payoff_ratio DECIMAL(10,4),
            cpc_ratio DECIMAL(10,4),
            risk_return_ratio DECIMAL(10,4),
            common_sense_ratio DECIMAL(10,4),
            max_drawdown DECIMAL(10,4),
            max_drawdown_days INTEGER,
            avg_drawdown DECIMAL(10,4),
            avg_drawdown_days DECIMAL(10,4),
            current_drawdown DECIMAL(10,4),
            current_drawdown_days INTEGER,
            drawdown_duration INTEGER,
            conditional_drawdown_at_risk DECIMAL(10,4),
            ulcer_index DECIMAL(10,4),
            risk_of_ruin DECIMAL(10,4),
            var_95 DECIMAL(10,4),
            cvar_99 DECIMAL(10,4),
            downside_deviation DECIMAL(10,4),
            volatility DECIMAL(10,4),
            annualized_volatility DECIMAL(10,4),
            skewness DECIMAL(10,4),
            kurtosis DECIMAL(10,4),
            winning_weeks INTEGER,
            losing_weeks INTEGER,
            winning_months INTEGER,
            losing_months INTEGER,
            winning_months_percent DECIMAL(10,4),
            negative_months_percent DECIMAL(10,4),
            total_profit DECIMAL(10,4),
            net_profit DECIMAL(10,4),
            avg_profit_per_trade DECIMAL(10,4),
            avg_loss_per_trade DECIMAL(10,4),
            profit_loss_ratio DECIMAL(10,4),
            number_of_trades INTEGER,
            win_rate DECIMAL(10,4),
            loss_rate DECIMAL(10,4),
            average_win DECIMAL(10,4),
            average_loss DECIMAL(10,4),
            average_trade_duration DECIMAL(10,4),
            largest_win DECIMAL(10,4),
            largest_loss DECIMAL(10,4),
            consecutive_wins INTEGER,
            consecutive_losses INTEGER,
            avg_trade_return DECIMAL(10,4),
            profitability_per_trade DECIMAL(10,4),
            recovery_factor DECIMAL(10,4),
            total_long_return DECIMAL(10,4),
            avg_long_return_per_trade DECIMAL(10,4),
            num_long_trades INTEGER,
            win_rate_long_trades DECIMAL(10,4),
            avg_long_trade_duration DECIMAL(10,4),
            max_long_trade_return DECIMAL(10,4),
            min_long_trade_return DECIMAL(10,4),
            long_trades_percent DECIMAL(10,4),
            total_short_return DECIMAL(10,4),
            avg_short_return_per_trade DECIMAL(10,4),
            num_short_trades INTEGER,
            win_rate_short_trades DECIMAL(10,4),
            avg_short_trade_duration DECIMAL(10,4),
            max_short_trade_return DECIMAL(10,4),
            min_short_trade_return DECIMAL(10,4),
            short_trades_percent DECIMAL(10,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_sql)
        print("Stats schema and table created successfully")
    except Exception as e:
        print(f"Error creating schema/table: {e}")

def process_strategy_batch(strategy_tables_batch, engine):
    """Process a batch of strategies and return results"""
    results = []
    
    for table_name in strategy_tables_batch:
        try:
            strategy_name = table_name.replace('_backtest', '')
            
            # Use chunking for large datasets
            chunk_size = 50000
            df_chunks = []
            
            # Read data in chunks to handle large datasets efficiently
            for chunk in pd.read_sql(f'SELECT * FROM strategies_backtest."{table_name}"', engine, chunksize=chunk_size):
                df_chunks.append(chunk)
            
            if not df_chunks:
                print(f"No data found for strategy: {strategy_name}")
                continue
                
            # Combine chunks
            df = pd.concat(df_chunks, ignore_index=True) if len(df_chunks) > 1 else df_chunks[0]
            
            if df.empty:
                print(f"No data found for strategy: {strategy_name}")
                continue
            
            stats_df = generate_stats_from_backtest(df)
            
            if not stats_df.empty:
                stats_df["strategy_name"] = strategy_name
                results.append(stats_df)
                print(f"✓ Processed: {strategy_name}")
            else:
                print(f"✗ No stats generated for: {strategy_name}")
                
        except Exception as e:
            print(f"✗ Failed for {strategy_name}: {e}")
            continue
    
    return results

def main():
    start_time = time.time()
    engine = get_pg_engine()
    connection = engine.raw_connection()
    cursor = connection.cursor()
    
    try:
        create_stats_schema_and_table(cursor)
        
        # Get all strategy tables from strategies_backtest schema - optimized query
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'strategies_backtest' 
            AND table_name LIKE 'strategy_%_backtest'
            ORDER BY table_name
        """)
        strategy_tables = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(strategy_tables)} strategy tables to process")
        
        if not strategy_tables:
            print("No strategy tables found!")
            return
        
        # Clear existing stats
        cursor.execute("DELETE FROM stats.strategy_stats")
        print("Cleared existing stats")
        
        # Process in batches for better performance
        batch_size = 5  # Process 5 strategies at a time
        all_stats_dfs = []
        
        for i in range(0, len(strategy_tables), batch_size):
            batch = strategy_tables[i:i + batch_size]
            print(f"\n--- Processing batch {i//batch_size + 1}/{(len(strategy_tables) + batch_size - 1)//batch_size} ---")
            
            # Process batch
            batch_results = process_strategy_batch(batch, engine)
            all_stats_dfs.extend(batch_results)
            
            # Progress update
            processed = min(i + batch_size, len(strategy_tables))
            elapsed = time.time() - start_time
            avg_time_per_strategy = elapsed / processed if processed > 0 else 0
            remaining = len(strategy_tables) - processed
            eta = remaining * avg_time_per_strategy
            
            print(f"Progress: {processed}/{len(strategy_tables)} strategies processed")
            print(f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        # Bulk insert all results at once
        if all_stats_dfs:
            print(f"\n--- Inserting {len(all_stats_dfs)} strategy results to database ---")
            combined_stats = pd.concat(all_stats_dfs, ignore_index=True)
            
            # Use fast bulk insert with executemany
            print(f"Using fast bulk insert for {len(combined_stats)} records...")
            
            # Prepare the insert statement
            columns = list(combined_stats.columns)
            placeholders = ', '.join(['%s'] * len(columns))
            insert_sql = f"INSERT INTO stats.strategy_stats ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Convert DataFrame to list of tuples for bulk insert
            data_tuples = [tuple(row) for row in combined_stats.values]
            
            # Execute bulk insert
            cursor.executemany(insert_sql, data_tuples)
            connection.commit()
            
            print(f"✓ Successfully inserted {len(combined_stats)} strategy records using bulk insert")
        else:
            print("No stats to insert")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Stats generation completed in {total_time:.1f} seconds!")
        print(f"Average time per strategy: {total_time/len(strategy_tables):.1f}s")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        connection.commit()
        cursor.close()
        connection.close()

if __name__ == "__main__":
    main()