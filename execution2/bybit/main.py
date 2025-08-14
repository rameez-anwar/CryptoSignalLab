#!/usr/bin/env python3
"""
Main Execution Script for Unified Trading System
Runs the unified signal generator and trading system for users
"""

import os
import sys
import argparse
import time
import logging
import pandas as pd
import json
from typing import List, Optional
from sqlalchemy import text

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'utils'))

# Import our modules
from unified_signal_generator import UnifiedSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_trading_main.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingExecutor:
    def __init__(self):
        """Initialize trading executor"""
        self.unified_generator = UnifiedSignalGenerator()
    
    def run_single_user(self, user_id: int):
        """Run trading for a single user"""
        try:
            logger.info(f"Starting trading for user {user_id}")
            
            # Get user configuration
            user_config = self.unified_generator.get_user_config(user_id)
            if user_config is None:
                logger.error(f"User {user_id} not found or invalid configuration")
                return False
            
            # Validate user configuration
            if not user_config['strategies']:
                logger.error(f"User {user_id} has no strategies configured")
                return False
            
            if not user_config['api_key'] or not user_config['api_secret']:
                logger.error(f"User {user_id} has invalid API credentials")
                return False
            
            logger.info(f"User configuration validated for {user_config['name']}")
            logger.info(f"Strategies: {user_config['strategies']}")
            logger.info(f"Use ML: {user_config['use_ml']}")
            
            # Run continuous trading
            self.unified_generator.run_continuous_trading(user_id)
            return True
            
        except Exception as e:
            logger.error(f"Error running trading for user {user_id}: {e}")
            return False
    
    def run_all_users(self):
        """Run trading for all users"""
        try:
            logger.info("Starting trading for all users")
            
            # Get all users from database
            query = text("SELECT id FROM users.users ORDER BY id")
            with self.unified_generator.engine.connect() as conn:
                result = conn.execute(query)
                user_ids = [row[0] for row in result.fetchall()]
            
            if not user_ids:
                logger.warning("No users found in database")
                return False
            
            logger.info(f"Found {len(user_ids)} users")
            
            # Run trading for each user
            for user_id in user_ids:
                logger.info(f"Starting trading for user {user_id}")
                
                try:
                    self.run_single_user(user_id)
                except Exception as e:
                    logger.error(f"Error running trading for user {user_id}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            logger.error(f"Error running trading for all users: {e}")
            return False
    
    def run_user_batch(self, user_ids: List[int]):
        """Run trading for a batch of users"""
        try:
            logger.info(f"Starting trading for user batch: {user_ids}")
            
            for user_id in user_ids:
                logger.info(f"Starting trading for user {user_id}")
                
                try:
                    self.run_single_user(user_id)
                except Exception as e:
                    logger.error(f"Error running trading for user {user_id}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            logger.error(f"Error running trading for user batch: {e}")
            return False
    
    def test_user_configuration(self, user_id: int):
        """Test user configuration without starting trading"""
        try:
            logger.info(f"Testing configuration for user {user_id}")
            
            # Get user configuration
            user_config = self.unified_generator.get_user_config(user_id)
            if user_config is None:
                logger.error(f"User {user_id} not found")
                return False
            
            # Test Bybit connection
            client = self.unified_generator.get_bybit_client(user_config)
            if client is None:
                logger.error(f"Failed to connect to Bybit for user {user_id}")
                return False
            
            # Test account balance
            balance = self.unified_generator.get_account_balance(client)
            logger.info(f"Account balance: {balance} USDT")
            
            # Test strategy configurations
            strategy_configs = self.unified_generator.get_strategy_configs(user_config['strategies'])
            if not strategy_configs:
                logger.error(f"No valid strategy configurations found for user {user_id}")
                return False
            
            logger.info(f"Found {len(strategy_configs)} valid strategy configurations")
            
            # Test ML models if enabled
            if user_config['use_ml']:
                logger.info("Testing ML model availability...")
                for strategy_config in strategy_configs:
                    best_model = self.unified_generator.ml_generator.find_best_model(
                        strategy_config['exchange'],
                        strategy_config['symbol'],
                        strategy_config['time_horizon']
                    )
                    if best_model:
                        logger.info(f"ML model found for {strategy_config['name']}: {best_model['model_name']}")
                    else:
                        logger.warning(f"No ML model found for {strategy_config['name']}")
            
            logger.info(f"Configuration test passed for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error testing configuration for user {user_id}: {e}")
            return False
    
    def list_users(self):
        """List all users from database"""
        try:
            from sqlalchemy import text
            
            query = text("""
                SELECT id, name, email, strategies, use_ml, created_at
                FROM users.users 
                ORDER BY id
            """)
            
            result = pd.read_sql_query(query, self.unified_generator.engine)
            
            if result.empty:
                print("No users found")
                return
            
            print("\n=== Users ===")
            print(f"{'ID':<5} {'Name':<20} {'Email':<30} {'Strategies':<15} {'ML':<5} {'Created':<20}")
            print("-" * 100)
            
            for _, row in result.iterrows():
                # Handle strategies field - it might be JSON string or already a list
                strategies_data = row['strategies']
                if isinstance(strategies_data, str):
                    strategies_count = len(json.loads(strategies_data)) if strategies_data else 0
                elif isinstance(strategies_data, list):
                    strategies_count = len(strategies_data)
                else:
                    strategies_count = 0
                
                ml_status = "Yes" if row['use_ml'] else "No"
                created_date = row['created_at'].strftime('%Y-%m-%d %H:%M')
                
                print(f"{row['id']:<5} {row['name']:<20} {row['email']:<30} "
                      f"{strategies_count:<15} {ml_status:<5} {created_date:<20}")
            
        except Exception as e:
            print(f"Error listing users: {e}")
    
    def list_strategies(self):
        """List all available strategies"""
        try:
            query = text("""
                SELECT name, exchange, symbol, time_horizon, take_profit, stop_loss
                FROM public.config_strategies 
                ORDER BY name
            """)
            
            result = pd.read_sql_query(query, self.unified_generator.engine)
            
            if result.empty:
                print("No strategies found")
                return
            
            print("\n=== Available Strategies ===")
            print(f"{'Name':<20} {'Exchange':<10} {'Symbol':<10} {'Time':<8} {'TP':<6} {'SL':<6}")
            print("-" * 70)
            
            for _, row in result.iterrows():
                print(f"{row['name']:<20} {row['exchange']:<10} {row['symbol']:<10} "
                      f"{row['time_horizon']:<8} {row['take_profit']:<6.1%} {row['stop_loss']:<6.1%}")
            
        except Exception as e:
            print(f"Error listing strategies: {e}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Unified Trading System')
    parser.add_argument('--user', type=int, help='Run trading for specific user ID')
    parser.add_argument('--all', action='store_true', help='Run trading for all users')
    parser.add_argument('--batch', nargs='+', type=int, help='Run trading for specific user IDs')
    parser.add_argument('--test', type=int, help='Test configuration for specific user ID')
    parser.add_argument('--list-users', action='store_true', help='List all users')
    parser.add_argument('--list-strategies', action='store_true', help='List all available strategies')
    
    args = parser.parse_args()
    
    executor = TradingExecutor()
    
    try:
        if args.list_users:
            # List all users
            executor.list_users()
            
        elif args.list_strategies:
            # List all strategies
            executor.list_strategies()
            
        elif args.test:
            # Test user configuration
            success = executor.test_user_configuration(args.test)
            if success:
                print(f"Configuration test passed for user {args.test}")
            else:
                print(f"Configuration test failed for user {args.test}")
                sys.exit(1)
                
        elif args.user:
            # Run for specific user
            print(f"Starting trading for user {args.user}")
            success = executor.run_single_user(args.user)
            if not success:
                print(f"Failed to start trading for user {args.user}")
                sys.exit(1)
                
        elif args.batch:
            # Run for batch of users
            print(f"Starting trading for users: {args.batch}")
            success = executor.run_user_batch(args.batch)
            if not success:
                print(f"Failed to start trading for user batch")
                sys.exit(1)
                
        elif args.all:
            # Run for all users
            print("Starting trading for all users")
            success = executor.run_all_users()
            if not success:
                print("Failed to start trading for all users")
                sys.exit(1)
                
        else:
            # Default: run for user ID 1
            print("No arguments provided. Starting trading for user ID 1 (default)")
            success = executor.run_single_user(1)
            if not success:
                print("Failed to start trading for user 1")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 