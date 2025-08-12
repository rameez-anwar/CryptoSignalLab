from data.utils.db_utils import get_pg_engine
from stats.generate_stats import generate_stats_from_backtest
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sqlalchemy import VARCHAR, DECIMAL, INTEGER, TIMESTAMP
import sys
import argparse

def create_stats_schema_and_table(cursor, table_type="strategy"):
    """Create stats schema and table if they don't exist"""
    try:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS stats")
        
        if table_type == "strategy":
            table_name = "strategy_stats"
        else:  # ml
            table_name = "ml_stats"
            
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS stats.{table_name} (
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
            total_loss DECIMAL(10,4),
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
        print(f"Stats schema and {table_name} table created successfully")
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

def process_ml_model_batch(ml_tables_batch, engine):
    """Process a batch of ML models and return results using real ledger data"""
    results = []
    
    for table_name in ml_tables_batch:
        try:
            # Use full table name as identifier (mirror strategies)
            model_identifier = table_name
            
            # Read ledger data directly from DB
            df = pd.read_sql(f'SELECT * FROM ml_ledger."{table_name}"', engine)
            
            if df.empty:
                print(f"No data found for ML ledger: {model_identifier}")
                continue
            
            # Ensure required columns are present or derived from real data
            # Required by generate_stats_from_backtest: ['pnl_percent', 'pnl_sum', 'balance', 'action']
            # 1) pnl_percent must exist in ledger per pipeline
            if 'pnl_percent' not in df.columns:
                print(f"Missing 'pnl_percent' in {model_identifier}; skipping.")
                continue
            
            # 2) pnl_sum: derive cumulatively from real pnl_percent if not present
            if 'pnl_sum' not in df.columns:
                df['pnl_sum'] = df['pnl_percent'].cumsum().round(2)
            
            # 3) balance: prefer existing; if missing, derive from pnl_sum and known initial 1000 used in ML backtests
            if 'balance' not in df.columns:
                df['balance'] = (1000 + df['pnl_sum']).round(2)
            
            # 4) action: prefer existing; if missing, conservatively map based on realized pnl sign
            if 'action' not in df.columns:
                df['action'] = np.where(df['pnl_percent'] > 0, 'tp', 'sl')
            
            # Ensure numeric types are proper
            df['pnl_percent'] = pd.to_numeric(df['pnl_percent'], errors='coerce').fillna(0)
            df['pnl_sum'] = pd.to_numeric(df['pnl_sum'], errors='coerce').fillna(0)
            df['balance'] = pd.to_numeric(df['balance'], errors='coerce').fillna(method='ffill').fillna(1000)
            
            # Generate stats from the real ledger dataframe
            stats_df = generate_stats_from_backtest(df)
            
            if not stats_df.empty:
                stats_df["strategy_name"] = model_identifier
                results.append(stats_df)
                print(f"✓ Processed ML ledger: {model_identifier}")
            else:
                print(f"✗ No stats generated for ML ledger: {model_identifier}")
                
        except Exception as e:
            print(f"✗ Failed for ML ledger {table_name}: {e}")
            continue
    
    return results

def generate_strategy_stats():
    """Generate stats for strategies"""
    start_time = time.time()
    engine = get_pg_engine()
    connection = engine.raw_connection()
    cursor = connection.cursor()
    
    try:
        create_stats_schema_and_table(cursor, "strategy")
        
        # Get all strategy tables from strategies_backtest schema
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
        print("Cleared existing strategy stats")
        
        # Process in batches for better performance
        batch_size = 5
        all_stats_dfs = []
        
        for i in range(0, len(strategy_tables), batch_size):
            batch = strategy_tables[i:i + batch_size]
            print(f"\n--- Processing strategy batch {i//batch_size + 1}/{(len(strategy_tables) + batch_size - 1)//batch_size} ---")
            
            batch_results = process_strategy_batch(batch, engine)
            all_stats_dfs.extend(batch_results)
            
            processed = min(i + batch_size, len(strategy_tables))
            elapsed = time.time() - start_time
            avg_time_per_strategy = elapsed / processed if processed > 0 else 0
            remaining = len(strategy_tables) - processed
            eta = remaining * avg_time_per_strategy
            
            print(f"Progress: {processed}/{len(strategy_tables)} strategies processed")
            print(f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        # Bulk insert all results
        if all_stats_dfs:
            print(f"\n--- Inserting {len(all_stats_dfs)} strategy results to database ---")
            combined_stats = pd.concat(all_stats_dfs, ignore_index=True)
            
            columns = list(combined_stats.columns)
            placeholders = ', '.join(['%s'] * len(columns))
            insert_sql = f"INSERT INTO stats.strategy_stats ({', '.join(columns)}) VALUES ({placeholders})"
            
            data_tuples = [tuple(row) for row in combined_stats.values]
            cursor.executemany(insert_sql, data_tuples)
            connection.commit()
            
            print(f"✓ Successfully inserted {len(combined_stats)} strategy records")
        else:
            print("No strategy stats to insert")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Strategy stats generation completed in {total_time:.1f} seconds!")
        
    except Exception as e:
        print(f"Error in strategy stats generation: {e}")
    finally:
        connection.commit()
        cursor.close()
        connection.close()

def generate_ml_stats():
    """Generate stats for ML models"""
    start_time = time.time()
    engine = get_pg_engine()
    connection = engine.raw_connection()
    cursor = connection.cursor()
    
    try:
        create_stats_schema_and_table(cursor, "ml")
        
        # Get all ML ledger tables from ml_ledger schema
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'ml_ledger' 
            ORDER BY table_name
        """)
        ml_tables = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(ml_tables)} ML model tables to process")
        
        if not ml_tables:
            print("No ML model tables found!")
            return
        
        # Clear existing ML stats
        cursor.execute("DELETE FROM stats.ml_stats")
        print("Cleared existing ML stats")
        
        # Process in batches for better performance
        batch_size = 5
        all_stats_dfs = []
        
        for i in range(0, len(ml_tables), batch_size):
            batch = ml_tables[i:i + batch_size]
            print(f"\n--- Processing ML batch {i//batch_size + 1}/{(len(ml_tables) + batch_size - 1)//batch_size} ---")
            
            batch_results = process_ml_model_batch(batch, engine)
            all_stats_dfs.extend(batch_results)
            
            processed = min(i + batch_size, len(ml_tables))
            elapsed = time.time() - start_time
            avg_time_per_model = elapsed / processed if processed > 0 else 0
            remaining = len(ml_tables) - processed
            eta = remaining * avg_time_per_model
            
            print(f"Progress: {processed}/{len(ml_tables)} ML models processed")
            print(f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        # Bulk insert all results
        if all_stats_dfs:
            print(f"\n--- Inserting {len(all_stats_dfs)} ML model results to database ---")
            combined_stats = pd.concat(all_stats_dfs, ignore_index=True)
            
            columns = list(combined_stats.columns)
            placeholders = ', '.join(['%s'] * len(columns))
            insert_sql = f"INSERT INTO stats.ml_stats ({', '.join(columns)}) VALUES ({placeholders})"
            
            data_tuples = [tuple(row) for row in combined_stats.values]
            cursor.executemany(insert_sql, data_tuples)
            connection.commit()
            
            print(f"✓ Successfully inserted {len(combined_stats)} ML model records")
        else:
            print("No ML stats to insert")
        
        total_time = time.time() - start_time
        print(f"\n🎉 ML stats generation completed in {total_time:.1f} seconds!")
        
    except Exception as e:
        print(f"Error in ML stats generation: {e}")
    finally:
        connection.commit()
        cursor.close()
        connection.close()

def main(stats_type="strategies"):
    """
    Main function to generate stats for strategies or ML models
    
    Args:
        stats_type (str): Either 'strategies' or 'ml' to specify which type of stats to generate
    """
    if stats_type == 'strategies':
        print("🚀 Generating stats for strategies...")
        generate_strategy_stats()
    elif stats_type == 'ml':
        print("🤖 Generating stats for ML models...")
        generate_ml_stats()
    else:
        print("Invalid type. Use 'strategies' or 'ml'")

if __name__ == "__main__":
    # Default to strategies if no argument provided
    main("ml")