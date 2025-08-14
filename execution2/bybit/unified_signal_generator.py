#!/usr/bin/env python3
"""
Unified Signal Generator and Trading System
Combines Strategy and ML signals for automated trading on Bybit
"""

import os
import sys
import pandas as pd
import numpy as np
import datetime
import time
import logging
import json
import pickle
from typing import Dict, Any, Optional, Tuple, List
from sqlalchemy import text, MetaData, Table, Column, String, DateTime, Float, Integer, Boolean, JSON
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# Add paths for existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'downloaders'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'strategies', 'strategy_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'indicators'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'signals'))

# Import existing modules
from DataDownloader import DataDownloader
from db_utils import get_pg_engine
from strategies.strategy_pipeline.signal_generator import StrategySignalGenerator
from ml.signal_generator import SignalGenerator as MLSignalGenerator
from learner.base_learner import BaseLearner

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UnifiedSignalGenerator:
    def __init__(self):
        """Initialize the unified signal generator"""
        self.engine = get_pg_engine()
        self.metadata = MetaData()
        self.base_learner = BaseLearner()
        self.strategy_generator = StrategySignalGenerator()
        self.ml_generator = MLSignalGenerator()
        
        # Trading state
        self.active_positions = {}
        self.trading_clients = {}
        self.ledger_data = {}  # Store ledger data per user/strategy
        self.last_signal_times = {}  # Track last signal generation time per strategy
        
        # Initialize database tables
        self._init_database_tables()
        
    def _init_database_tables(self):
        """Initialize database tables for execution tracking"""
        try:
            # Create execution schema if not exists
            with self.engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS execution;"))
                conn.commit()
            
            logger.info("Database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
    
    def get_ledger_table_name(self, user_id: int, strategy_config: Dict[str, Any]) -> str:
        """Generate consistent ledger table name"""
        return f"user_{user_id}_{strategy_config['exchange']}_{strategy_config['symbol']}_{strategy_config['time_horizon']}"
    
    def get_user_config(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user configuration from database"""
        try:
            query = text("""
                SELECT id, name, email, api_key, api_secret, strategies, use_ml, created_at, updated_at
                FROM users.users 
                WHERE id = :user_id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'user_id': user_id})
                row = result.fetchone()
                
            if row:
                # Handle strategies field - it might be JSON string or already a list
                strategies_data = row[5]
                if isinstance(strategies_data, str):
                    strategies = json.loads(strategies_data) if strategies_data else []
                elif isinstance(strategies_data, list):
                    strategies = strategies_data
                else:
                    strategies = []
                
                user_config = {
                    'id': row[0],
                    'name': row[1],
                    'email': row[2],
                    'api_key': row[3],
                    'api_secret': row[4],
                    'strategies': strategies,
                    'use_ml': row[6],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
                
                logger.info(f"Loaded user config for {user_config['name']}: {len(user_config['strategies'])} strategies, ML: {user_config['use_ml']}")
                return user_config
            else:
                logger.error(f"User {user_id} not found in database")
                return None
                
        except Exception as e:
            logger.error(f"Error getting user config: {e}")
            return None
    
    def get_strategy_configs(self, strategy_names: List[str]) -> List[Dict[str, Any]]:
        """Get strategy configurations from database"""
        try:
            if not strategy_names:
                return []
            
            placeholders = ','.join([f"'{name}'" for name in strategy_names])
            query = text(f"""
                SELECT * FROM public.config_strategies 
                WHERE name IN ({placeholders})
                ORDER BY name
            """)
            
            result = pd.read_sql_query(query, self.engine)
            
            if result.empty:
                logger.warning(f"No strategy configs found for: {strategy_names}")
                return []
            
            configs = []
            for _, row in result.iterrows():
                config = row.to_dict()
                configs.append(config)
                logger.info(f"Loaded strategy config: {config['name']} - {config['exchange']}/{config['symbol']}/{config['time_horizon']}")
            
            return configs
            
        except Exception as e:
            logger.error(f"Error getting strategy configs: {e}")
            return []
    
    def validate_strategy_combinations(self, strategy_configs: List[Dict[str, Any]]) -> bool:
        """Validate that strategies don't have conflicting symbols"""
        try:
            symbol_configs = {}
            
            for config in strategy_configs:
                symbol_key = f"{config['exchange']}_{config['symbol']}_{config['time_horizon']}"
                
                if symbol_key in symbol_configs:
                    logger.error(f"Duplicate symbol configuration found: {symbol_key}")
                    logger.error(f"  Strategy 1: {symbol_configs[symbol_key]['name']}")
                    logger.error(f"  Strategy 2: {config['name']}")
                    return False
                
                symbol_configs[symbol_key] = config
            
            logger.info(f"Strategy validation passed: {len(strategy_configs)} unique symbol configurations")
            return True
            
        except Exception as e:
            logger.error(f"Error validating strategy combinations: {e}")
            return False
    
    def get_bybit_client(self, user_config: Dict[str, Any]) -> HTTP:
        """Get or create Bybit client for user"""
        user_id = user_config['id']
        
        if user_id not in self.trading_clients:
            try:
                # Create client with proper timestamp sync
                client = HTTP(
                    demo=True,  # Use demo account
                    api_key=user_config['api_key'],
                    api_secret=user_config['api_secret'],
                    recv_window=60000  # Increased recv_window
                )
                
                # Sync with server time
                try:
                    server_time = client.get_server_time()
                    if server_time['retCode'] == 0:
                        server_timestamp = server_time['result']['timeSecond']
                        logger.info(f"Server time synced: {server_timestamp}")
                    else:
                        logger.warning("Could not sync server time, using local time")
                except Exception as e:
                    logger.warning(f"Server time sync failed: {e}")
                
                # Test connection
                try:
                    test_response = client.get_wallet_balance(accountType="UNIFIED")
                    if test_response['retCode'] == 0:
                        self.trading_clients[user_id] = client
                        logger.info(f"Bybit client created for user {user_config['name']}")
                    else:
                        logger.error(f"Failed to connect to Bybit for user {user_config['name']}: {test_response}")
                        return None
                except Exception as e:
                    logger.error(f"Connection test failed for user {user_config['name']}: {e}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error creating Bybit client for user {user_config['name']}: {e}")
                return None
        
        return self.trading_clients[user_id]
    
    def get_account_balance(self, client: HTTP) -> float:
        """Get current account balance from Bybit using the correct API endpoint"""
        try:
            # Use the correct endpoint for wallet balance
            balance_data = client.get_wallet_balance(accountType="UNIFIED")
            
            if balance_data['retCode'] == 0:
                # The response has a nested structure with coin array
                for account in balance_data['result']['list']:
                    if account.get('coin'):  # Check if coin array exists
                        for coin in account['coin']:
                            if coin['coin'] == 'USDT':
                                # Use walletBalance as the total balance
                                balance = float(coin['walletBalance'])
                                logger.info(f"Real account balance: {balance} USDT")
                                return balance
                    else:
                        # Direct coin structure (fallback)
                        if account.get('coin') == 'USDT':
                            balance = float(account['walletBalance'])
                            logger.info(f"Real account balance: {balance} USDT")
                            return balance
            
            logger.error(f"Failed to get balance from Bybit API: {balance_data}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    def get_current_price(self, client: HTTP, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            ticker = client.get_tickers(category="linear", symbol=symbol)
            
            if ticker['retCode'] == 0 and ticker['result']['list']:
                price = float(ticker['result']['list'][0]['lastPrice'])
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_position(self, client: HTTP, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for symbol"""
        try:
            positions = client.get_positions(category="linear", symbol=symbol)
            
            if positions['retCode'] == 0:
                for pos in positions['result']['list']:
                    if float(pos['size']) > 0:
                        position_info = {
                            'size': float(pos['size']),
                            'side': pos['side'],
                            'entry_price': float(pos['avgPrice']),
                            'unrealized_pnl': float(pos.get('unrealisedPnl', 0)),
                            'position_value': float(pos.get('positionValue', 0)),
                            'mark_price': float(pos.get('markPrice', 0)),
                            'leverage': float(pos.get('leverage', 1)),
                            'position_idx': pos.get('positionIdx', 0)
                        }
                        return position_info
            return None
            
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None
    
    def get_order_history(self, client: HTTP, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order history for symbol"""
        try:
            orders = client.get_order_history(
                category="linear",
                symbol=symbol,
                limit=limit
            )
            
            if orders['retCode'] == 0:
                return orders['result']['list']
            return []
            
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    def get_transaction_log(self, client: HTTP, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transaction log for account"""
        try:
            transactions = client.get_transaction_log(
                accountType="UNIFIED",
                limit=limit
            )
            
            if transactions['retCode'] == 0:
                return transactions['result']['list']
            return []
            
        except Exception as e:
            logger.error(f"Error getting transaction log: {e}")
            return []
    
    def get_closed_pnl(self, client: HTTP, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get closed PnL history for symbol"""
        try:
            closed_pnl = client.get_closed_pnl(
                category="linear",
                symbol=symbol,
                limit=limit
            )
            
            if closed_pnl['retCode'] == 0:
                return closed_pnl['result']['list']
            return []
            
        except Exception as e:
            logger.error(f"Error getting closed PnL: {e}")
            return []
    
    def should_generate_signal(self, strategy_config: Dict[str, Any]) -> bool:
        """Check if we should generate a signal based on time horizon"""
        try:
            time_horizon = strategy_config.get('time_horizon', '1h')
            
            # Convert time horizon to minutes
            if time_horizon == '1h':
                interval_minutes = 60
            elif time_horizon == '4h':
                interval_minutes = 240
            elif time_horizon == '1d':
                interval_minutes = 1440
            else:
                interval_minutes = 60  # Default to 1 hour
            
            # Get current time
            now = datetime.datetime.now()
            
            # Create a unique key for this strategy to track if we've generated the initial signal
            strategy_key = f"{strategy_config['name']}_{strategy_config['exchange']}_{strategy_config['symbol']}_{strategy_config['time_horizon']}"
            
            # Check if we've already generated the initial signal for this strategy
            if strategy_key not in self.last_signal_times:
                # First time running - generate signal from last hour boundary
                logger.info(f"First run for {strategy_config['name']} - generating signal from last hour boundary")
                self.last_signal_times[strategy_key] = now
                return True
            
            # Check if we're at the start of an hour (for 1h), 4-hour boundary (for 4h), etc.
            if time_horizon == '1h':
                # Should generate at exact hour boundaries (00:00, 01:00, 02:00, etc.)
                if now.minute == 0 and now.second < 30:  # Within first 30 seconds of the hour
                    self.last_signal_times[strategy_key] = now
                    return True
            elif time_horizon == '4h':
                # Should generate at 4-hour boundaries (00:00, 04:00, 08:00, etc.)
                if now.hour % 4 == 0 and now.minute == 0 and now.second < 30:
                    self.last_signal_times[strategy_key] = now
                    return True
            elif time_horizon == '1d':
                # Should generate at daily boundaries (00:00)
                if now.hour == 0 and now.minute == 0 and now.second < 30:
                    self.last_signal_times[strategy_key] = now
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking signal generation time: {e}")
            return False
    
    def generate_strategy_signal(self, strategy_config: Dict[str, Any]) -> Optional[int]:
        """Generate signal from strategy using existing StrategySignalGenerator"""
        try:
            signal = self.strategy_generator.generate_latest_signal(strategy_config['name'])
            
            # The strategy generator might return 0 for "hold" but we need to interpret it correctly
            if signal == 0:
                # Check if this is a "hold" signal by looking at the last signal from database
                last_signal_time, last_signal = self.strategy_generator.get_latest_signal_from_db(strategy_config['name'])
                if last_signal is not None and last_signal in [1, -1]:
                    # This is a hold signal, return the last signal direction
                    logger.info(f"Strategy signal for {strategy_config['name']}: HOLD {('LONG' if last_signal == 1 else 'SHORT')} ({last_signal})")
                    return last_signal
                else:
                    # This is truly a neutral signal
                    logger.info(f"Strategy signal for {strategy_config['name']}: NEUTRAL (0)")
                    return 0
            elif signal == 1:
                logger.info(f"Strategy signal for {strategy_config['name']}: LONG (1)")
                return 1
            elif signal == -1:
                logger.info(f"Strategy signal for {strategy_config['name']}: SHORT (-1)")
                return -1
            else:
                logger.warning(f"Strategy signal for {strategy_config['name']}: UNKNOWN ({signal})")
                return 0  # Default to neutral
            
        except Exception as e:
            logger.error(f"Error generating strategy signal: {e}")
            return None
    
    def generate_ml_signal(self, strategy_config: Dict[str, Any]) -> Optional[int]:
        """Generate signal from ML model using existing MLSignalGenerator"""
        try:
            # Find best ML model for this configuration
            best_model = self.ml_generator.find_best_model(
                strategy_config['exchange'],
                strategy_config['symbol'],
                strategy_config['time_horizon']
            )
            
            if best_model is None:
                logger.warning(f"No ML model found for {strategy_config['exchange']}/{strategy_config['symbol']}/{strategy_config['time_horizon']}")
                return None
            
            # Load model - fix the path to use the correct structure
            model_name = best_model['model_name']
            clean_name = model_name
            if model_name.endswith('_model'):
                clean_name = model_name[:-6]
            
            # Try different possible paths
            possible_paths = [
                os.path.join("..", "..", "ml", "trainer", strategy_config['symbol'].lower(), strategy_config['time_horizon'], clean_name, f"{clean_name}_best.pkl"),
                os.path.join("ml", "trainer", strategy_config['symbol'].lower(), strategy_config['time_horizon'], clean_name, f"{clean_name}_best.pkl"),
                os.path.join("trainer", strategy_config['symbol'].lower(), strategy_config['time_horizon'], clean_name, f"{clean_name}_best.pkl")
            ]
            
            model_loaded = False
            model_path = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                logger.error(f"Failed to find ML model file: {model_name} in any of the paths: {possible_paths}")
                return None
            
            # Load the model directly
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Map model name to base_learner model name (add _model suffix if not present)
                base_learner_model_name = model_name
                if not model_name.endswith('_model'):
                    base_learner_model_name = f"{model_name}_model"
                
                # Check if the mapped model name exists in base_learner
                if base_learner_model_name not in self.base_learner.models:
                    logger.error(f"Model {base_learner_model_name} not found in base_learner")
                    logger.error(f"Available models: {list(self.base_learner.models.keys())}")
                    return None
                
                model_instance = self.base_learner.models[base_learner_model_name]
                
                # Load the model data into the instance
                if hasattr(model_instance, 'load_model'):
                    model_instance.load_model(model_path)
                else:
                    # For models that don't have load_model method, try to set attributes directly
                    if 'model' in model_data:
                        model_instance.model = model_data['model']
                    if 'scaler' in model_data:
                        model_instance.scaler = model_data['scaler']
                
                # Initialize parameters
                self.ml_generator.current_params = model_instance.get_default_params()
                
                # Determine the correct lookback period from the model's feature count
                if hasattr(model_instance.model, 'n_features_in_'):
                    actual_lookback = model_instance.model.n_features_in_ // 4
                    self.ml_generator.current_params['lookback'] = actual_lookback
                    logger.info(f"Model expects {model_instance.model.n_features_in_} features")
                    logger.info(f"Calculated lookback period: {actual_lookback}")
                else:
                    # Fallback to default
                    actual_lookback = 60
                    self.ml_generator.current_params['lookback'] = actual_lookback
                    logger.info(f"Could not determine lookback period, using default: {actual_lookback}")
                
                # Store current model info
                self.ml_generator.current_model = model_instance
                self.ml_generator.current_model_path = model_path
                
                # Add to trained_models dict using the base_learner model name
                self.base_learner.trained_models[base_learner_model_name] = model_instance
                
                logger.info(f"Successfully loaded model: {base_learner_model_name}")
                logger.info(f"Model path: {model_path}")
                logger.info(f"Model parameters: {self.ml_generator.current_params}")
                
                model_loaded = True
                
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
                return None
            
            if not model_loaded:
                logger.error(f"Failed to load ML model: {model_name}")
                return None
            
            # Get latest data
            lookback = self.ml_generator.current_params.get('lookback', 60)
            data = self.ml_generator.get_latest_data(
                strategy_config['exchange'],
                strategy_config['symbol'],
                strategy_config['time_horizon'],
                lookback + 10
            )
            
            if data is None:
                logger.error("Failed to get data for ML signal generation")
                return None
            
            # Generate signal
            signal, percent_change, signal_text = self.ml_generator.generate_current_signal(
                model_name,
                data
            )
            
            # Ensure we get proper signal values
            if signal == 1:
                logger.info(f"ML signal for {strategy_config['symbol']}: LONG (1) - {signal_text}, {percent_change:.2f}%")
                return 1
            elif signal == -1:
                logger.info(f"ML signal for {strategy_config['symbol']}: SHORT (-1) - {signal_text}, {percent_change:.2f}%")
                return -1
            elif signal == 0:
                logger.info(f"ML signal for {strategy_config['symbol']}: NEUTRAL (0) - {signal_text}, {percent_change:.2f}%")
                return 0
            else:
                logger.warning(f"ML signal for {strategy_config['symbol']}: UNKNOWN ({signal}) - {signal_text}, {percent_change:.2f}%")
                return 0  # Default to neutral
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return None
    
    def combine_signals(self, strategy_signal: Optional[int], ml_signal: Optional[int]) -> int:
        """Combine strategy and ML signals according to rules"""
        try:
            # Convert None signals to 0 (neutral)
            strategy_signal = strategy_signal if strategy_signal is not None else 0
            ml_signal = ml_signal if ml_signal is not None else 0
            
            logger.info(f"Combining signals - Strategy: {strategy_signal}, ML: {ml_signal}")
            
            # Signal combination rules:
            # Both 1 (long) -> 1 (long)
            # Both -1 (short) -> -1 (short)
            # Both 0 (neutral) -> 0 (neutral)
            # Opposite signals -> 0 (neutral)
            
            if strategy_signal == ml_signal:
                combined_signal = strategy_signal
                if combined_signal == 1:
                    logger.info(f"Signals agree: LONG (1)")
                elif combined_signal == -1:
                    logger.info(f"Signals agree: SHORT (-1)")
                else:
                    logger.info(f"Signals agree: NEUTRAL (0)")
            else:
                combined_signal = 0  # Neutral when signals disagree
                logger.info(f"Signals disagree, using NEUTRAL (0)")
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return 0  # Default to neutral
    
    def format_quantity(self, qty: float, symbol: str) -> float:
        """Format quantity according to Bybit's requirements"""
        try:
            # Get symbol info for minimum quantity and step size
            symbol_info = self._get_symbol_info(symbol)
            min_qty = symbol_info.get('min_qty', 0.001)
            qty_step = symbol_info.get('qty_step', 0.001)
            
            # Ensure quantity meets minimum requirement
            if qty < min_qty:
                qty = min_qty
                logger.warning(f"Quantity {qty} below minimum {min_qty} for {symbol}, using minimum")
            
            # Round to step size
            qty = round(qty / qty_step) * qty_step
            
            # Ensure we don't exceed reasonable limits
            if qty > 1000000:  # 1M max quantity
                qty = 1000000
                logger.warning(f"Quantity too large for {symbol}, capping at 1M")
            
            logger.info(f"Formatted quantity for {symbol}: {qty}")
            return qty
            
        except Exception as e:
            logger.error(f"Error formatting quantity: {e}")
            return qty  # Return original if formatting fails
    
    def _get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information including minimum quantity and step size"""
        try:
            # Default values for common symbols
            if 'BTC' in symbol:
                return {'min_qty': 0.001, 'qty_step': 0.001, 'tick_size': 0.1}
            elif 'XRP' in symbol:
                return {'min_qty': 1, 'qty_step': 1, 'tick_size': 0.0001}
            elif 'ETH' in symbol:
                return {'min_qty': 0.001, 'qty_step': 0.001, 'tick_size': 0.01}
            elif 'SOL' in symbol:
                return {'min_qty': 0.1, 'qty_step': 0.1, 'tick_size': 0.01}
            elif 'ADA' in symbol:
                return {'min_qty': 1, 'qty_step': 1, 'tick_size': 0.0001}
            else:
                return {'min_qty': 0.001, 'qty_step': 0.001, 'tick_size': 0.01}
                
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            # Return safe defaults
            return {'min_qty': 0.001, 'qty_step': 0.001, 'tick_size': 0.01}
    
    def place_order(self, client: HTTP, symbol: str, side: str, qty: float, 
                   tp_price: Optional[float] = None, sl_price: Optional[float] = None) -> Optional[str]:
        """Place order with TP/SL"""
        try:
            # Format quantity according to Bybit requirements
            formatted_qty = self.format_quantity(qty, symbol)
            
            # Get symbol info for price formatting
            symbol_info = self._get_symbol_info(symbol)
            tick_size = symbol_info.get('tick_size', 0.01)
            
            # Format prices to tick size
            decimal_places = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
            
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(formatted_qty),
                "timeInForce": "GTC"
            }
            
            if tp_price:
                formatted_tp = round(tp_price, decimal_places)
                order_params["takeProfit"] = str(formatted_tp)
                logger.info(f"Take Profit: {formatted_tp}")
            
            if sl_price:
                formatted_sl = round(sl_price, decimal_places)
                order_params["stopLoss"] = str(formatted_sl)
                logger.info(f"Stop Loss: {formatted_sl}")
            
            logger.info(f"Placing order: {side} {formatted_qty} {symbol}")
            logger.info(f"Order params: {order_params}")
            
            response = client.place_order(**order_params)
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                logger.info(f"Order placed successfully: {side} {formatted_qty} {symbol}, ID: {order_id}")
                return order_id
            else:
                logger.error(f"Order failed: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def close_position(self, client: HTTP, symbol: str, side: str, qty: float) -> bool:
        """Close existing position"""
        try:
            close_side = "Sell" if side == "Buy" else "Buy"
            
            # Format quantity according to Bybit requirements
            formatted_qty = self.format_quantity(qty, symbol)
            
            logger.info(f"Closing position: {close_side} {formatted_qty} {symbol}")
            
            response = client.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=str(formatted_qty),
                timeInForce="GTC",
                reduceOnly=True
            )
            
            if response['retCode'] == 0:
                logger.info(f"Position closed successfully: {close_side} {formatted_qty} {symbol}")
                return True
            else:
                logger.error(f"Failed to close position: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def calculate_position_size(self, balance: float, strategy_config: Dict[str, Any], user_id: int) -> float:
        """Calculate position size based on available balance and compounding logic"""
        try:
            # Check if this is the first trade or if we have trading history
            ledger_key = f"{user_id}_{strategy_config['name']}"
            
            if ledger_key in self.ledger_data and len(self.ledger_data[ledger_key]) > 0:
                # We have trading history - use the last trade amount for compounding
                # Get the last trade amount from the ledger (after PnL adjustments)
                last_trade_amount = self.ledger_data[ledger_key].get('last_trade_amount', 1000.0)
                position_size = last_trade_amount
                logger.info(f"Using compounding amount from ledger: {position_size} USDT")
            else:
                # First trade - use $1000
                position_size = 1000.0
                logger.info(f"Using initial amount: {position_size} USDT")
            
            # Ensure we don't exceed available balance
            if position_size > balance * 0.8:  # 80% of balance for safety
                position_size = balance * 0.8
                logger.warning(f"Position size adjusted to {position_size:.2f} USDT due to balance constraints")
            
            # Ensure minimum position size
            if position_size < 10:
                position_size = 10
                logger.warning(f"Position size set to minimum: {position_size} USDT")
            
            logger.info(f"Final position size: {position_size:.2f} USDT from balance: {balance:.2f}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1000.0  # Default to $1000
    
    def get_trading_fee(self, client: HTTP, symbol: str, order_id: str) -> float:
        """Get real trading fee from Bybit execution API"""
        try:
            # Use the execution API to get real fees
            response = client.get_executions(
                category="linear",
                symbol=symbol,
                orderId=order_id
            )
            
            if response['retCode'] == 0 and response['result']['list']:
                total_fee = 0.0
                for execution in response['result']['list']:
                    if 'execFee' in execution and execution['execFee']:
                        fee = float(execution['execFee'])
                        total_fee += fee
                        logger.info(f"Execution fee for {symbol} order {order_id}: {fee} USDT")
                
                if total_fee > 0:
                    logger.info(f"Total execution fee for {symbol} order {order_id}: {total_fee} USDT")
                    return total_fee
            
            # Fallback to transaction log if execution API doesn't work
            transactions = self.get_transaction_log(client, limit=100)
            for transaction in transactions:
                if (transaction.get('orderId') == order_id and 
                    transaction.get('type') == 'TRADING_FEE'):
                    fee = float(transaction.get('amount', 0))
                    logger.info(f"Transaction log fee for {symbol} order {order_id}: {fee} USDT")
                    return abs(fee)
            
            logger.error(f"Could not find real fee for order {order_id}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting trading fee: {e}")
            return 0.0
    
    def update_ledger(self, user_id: int, strategy_config: Dict[str, Any], 
                     action: str, signal: int, entry_price: float, exit_price: float, 
                     balance: float, pnl: float, pnl_sum: float, trade_amount: float = 0.0, order_id: str = ""):
        """Update ledger for user's strategy"""
        try:
            # Create table name with prefix to ensure it starts with a letter
            table_name = self.get_ledger_table_name(user_id, strategy_config)
            
            # Check if table already exists in MetaData
            if table_name not in self.metadata.tables:
                # Create table if not exists - matching CSV format exactly
                ledger_table = Table(
                    table_name,
                    self.metadata,
                    Column('datetime', DateTime, primary_key=True),
                    Column('predicted_direction', String),  # 'long', 'short', 'neutral'
                    Column('action', String),  # 'buy', 'sell - take_profit', 'sell - direction change', 'same direction', 'manually_closed'
                    Column('buy_price', Float),
                    Column('sell_price', Float),
                    Column('pnl_percent', Float),  # Single trade PnL
                    Column('pnl_sum', Float),  # Cumulative PnL
                    Column('balance', Float),
                    Column('trade_amount', Float),  # Amount used for this specific trade
                    Column('order_id', String),  # Order ID from Bybit
                    schema='execution',
                    extend_existing=True
                )
                
                # Create table if not exists
                self.metadata.create_all(self.engine, tables=[ledger_table])
            
            # Convert numpy types to native Python types and round to 2 decimal places
            def convert_to_native(value):
                if hasattr(value, 'item'):  # numpy scalar
                    return value.item()
                elif isinstance(value, (np.integer, np.floating)):
                    return float(value) if isinstance(value, np.floating) else int(value)
                else:
                    return value
            
            native_entry_price = round(convert_to_native(entry_price), 2)
            native_exit_price = round(convert_to_native(exit_price), 2)
            native_balance = round(convert_to_native(balance), 2)
            native_pnl = round(convert_to_native(pnl), 2)
            native_pnl_sum = round(convert_to_native(pnl_sum), 2)
            native_trade_amount = round(convert_to_native(trade_amount), 2)
            
            # Apply -0.05% PnL for open/close actions (representing fees)
            if action in ['open', 'tp', 'sl', 'direction_change', 'neutral_signal', 'manually_closed']:
                native_pnl = -0.05
                # Deduct the fee from trade amount
                fee_amount = (native_trade_amount * 0.0005)  # 0.05% of trade amount
                native_trade_amount -= fee_amount
                native_trade_amount = round(native_trade_amount, 2)
                logger.info(f"Applied -0.05% PnL for {action} action, fee: {fee_amount:.2f}, new trade amount: {native_trade_amount}")
            else:
                # For other actions, apply the actual PnL to trade amount
                if native_pnl != 0:
                    pnl_amount = (native_trade_amount * native_pnl / 100)  # Convert percentage to amount
                    native_trade_amount += pnl_amount
                    native_trade_amount = round(native_trade_amount, 2)
                    logger.info(f"Applied {native_pnl}% PnL to trade amount, change: {pnl_amount:.2f}, new trade amount: {native_trade_amount}")
            
            # Convert signal to direction string
            if signal == 1:
                predicted_direction = 'long'
            elif signal == -1:
                predicted_direction = 'short'
            else:
                predicted_direction = 'neutral'
            
            # Convert action to match CSV format
            if action == 'open':
                action_str = 'buy'
            elif action == 'tp':
                action_str = 'sell - take_profit'
            elif action == 'sl':
                action_str = 'sell - stop_loss'
            elif action == 'direction_change':
                action_str = 'sell - direction change'
            elif action == 'neutral_signal':
                action_str = 'sell - neutral signal'
            elif action == 'same_direction':
                action_str = 'same direction'
            elif action == 'manually_closed':
                action_str = 'manually_closed'
            else:
                action_str = action
            
            # Format datetime as YYYY-MM-DD HH:MM:SS
            current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Insert ledger entry
            query = text(f"""
                INSERT INTO execution.{table_name} 
                (datetime, predicted_direction, action, buy_price, sell_price, balance, pnl_percent, pnl_sum, trade_amount, order_id)
                VALUES (:datetime, :predicted_direction, :action, :buy_price, :sell_price, :balance, :pnl_percent, :pnl_sum, :trade_amount, :order_id)
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'datetime': current_datetime,
                    'predicted_direction': predicted_direction,
                    'action': action_str,
                    'buy_price': native_entry_price,
                    'sell_price': native_exit_price,
                    'balance': native_balance,
                    'pnl_percent': native_pnl,
                    'pnl_sum': native_pnl_sum,
                    'trade_amount': native_trade_amount,
                    'order_id': order_id
                })
                conn.commit()
            
            logger.info(f"Ledger updated for {table_name}: {action_str} {predicted_direction}, Trade Amount: {native_trade_amount}")
            
        except Exception as e:
            logger.error(f"Error updating ledger: {e}")
    
    def execute_trading_logic(self, user_id: int, user_config: Dict[str, Any]):
        """Execute trading logic for a user"""
        try:
            logger.info(f"Executing trading logic for user {user_config['name']}")
            
            # Get Bybit client
            client = self.get_bybit_client(user_config)
            if client is None:
                logger.error(f"Failed to get Bybit client for user {user_config['name']}")
                return
            
            # Get strategy configurations
            strategy_configs = self.get_strategy_configs(user_config['strategies'])
            if not strategy_configs:
                logger.warning(f"No strategy configs found for user {user_config['name']}")
                return
            
            # Validate strategy combinations
            if not self.validate_strategy_combinations(strategy_configs):
                logger.error(f"Invalid strategy combinations for user {user_config['name']}")
                return
            
            # Get account balance
            balance = self.get_account_balance(client)
            
            # Track signals and positions for each strategy
            for strategy_config in strategy_configs:
                symbol = f"{strategy_config['symbol'].upper()}USDT"
                
                # Check if it's time to generate signal based on time horizon
                if not self.should_generate_signal(strategy_config):
                    continue
                
                # Check old trades from ledger before generating new signals
                old_trade_updated = self.check_old_trades_from_ledger(client, user_id, strategy_config)
                if old_trade_updated:
                    logger.info(f"Old trade updated for {symbol}, proceeding with new signal generation")
                
                # Generate signals using existing generators
                strategy_signal = self.generate_strategy_signal(strategy_config)
                
                if user_config['use_ml']:
                    ml_signal = self.generate_ml_signal(strategy_config)
                    combined_signal = self.combine_signals(strategy_signal, ml_signal)
                    logger.info(f"Combined signal for {symbol}: {combined_signal}")
                else:
                    combined_signal = strategy_signal
                    logger.info(f"Strategy-only signal for {symbol}: {combined_signal}")
                
                # Define position_key at the beginning
                position_key = f"{user_id}_{symbol}"
                
                # Check for manual closure first
                if position_key in self.active_positions:
                    last_order_id = self.active_positions[position_key].get('order_id', "")
                    if last_order_id and self.check_manual_closure(client, symbol, last_order_id):
                        # Handle manual closure and update ledger
                        if self.handle_manual_closure(client, user_id, strategy_config, symbol, position_key):
                            logger.info(f"Manual closure handled for {symbol}")
                            continue  # Skip to next iteration
                
                # Check if we should take action
                last_signal = self.active_positions.get(position_key, {}).get('last_signal')
                
                # Check for open orders first
                open_orders = self.get_open_orders(client, symbol)
                has_open_order = len(open_orders) > 0
                
                # Get current position
                current_position = self.get_position(client, symbol)
                
                # If we have a tracked position but no current position, check what happened
                if position_key in self.active_positions and (current_position is None or float(current_position.get('size', 0)) == 0):
                    # Position was closed - determine why and update ledger
                    self.handle_position_closure(client, user_id, strategy_config, symbol, position_key, combined_signal)
                    continue
                
                if current_position:
                    # Position exists - check if we should close it
                    position_side = current_position['side']
                    should_close = False
                    close_reason = ""
                    
                    # Check for TP/SL hits first
                    closed_pnl_data = self.get_closed_pnl(client, symbol, limit=10)
                    for pnl_record in closed_pnl_data:
                        if pnl_record['side'] == position_side and float(pnl_record['size']) > 0:
                            # Position was closed by TP/SL
                            should_close = True
                            close_reason = "tp" if float(pnl_record.get('closedPnl', 0)) > 0 else "sl"
                            break
                    
                    # If not TP/SL, check signal changes
                    if not should_close:
                        if (combined_signal == 1 and position_side == "Sell") or (combined_signal == -1 and position_side == "Buy"):
                            should_close = True
                            close_reason = "direction_change"
                    
                    if should_close:
                        # Close position
                        if self.close_position(client, symbol, position_side, current_position['size']):
                            # Get exit price and calculate PnL
                            exit_price = self.get_current_price(client, symbol) or current_position['mark_price']
                            
                            # Get real balance after closing
                            new_balance = self.get_account_balance(client)
                            
                            # Calculate PnL including fees
                            if position_side == "Buy":  # Long position
                                pnl_percent = ((exit_price - current_position['entry_price']) / current_position['entry_price']) * 100
                            else:  # Short position
                                pnl_percent = ((current_position['entry_price'] - exit_price) / current_position['entry_price']) * 100
                            
                            # Get closing fee
                            close_fee = 0.0
                            if 'order_id' in self.active_positions.get(position_key, {}):
                                close_fee = self.get_trading_fee(client, symbol, self.active_positions[position_key]['order_id'])
                            
                            # Adjust PnL for fees
                            position_value = current_position['size'] * current_position['entry_price']
                            fee_percent = (close_fee / position_value) * 100
                            pnl_percent -= fee_percent
                            
                            # Calculate cumulative PnL
                            ledger_key = f"{user_id}_{strategy_config['name']}"
                            if ledger_key not in self.ledger_data:
                                self.ledger_data[ledger_key] = {'pnl_sum': 0.0, 'last_trade_amount': 1000.0}
                            
                            # Update pnl_sum with the actual PnL (including the -0.05% fee)
                            # pnl_percent already includes the fee deduction, so add it directly to pnl_sum
                            self.ledger_data[ledger_key]['pnl_sum'] += pnl_percent
                            
                            # Get order ID for ledger
                            order_id = self.active_positions.get(position_key, {}).get('order_id', "")
                            
                            # Update ledger with real balance and actual PnL
                            self.update_ledger(
                                user_id, strategy_config, close_reason,
                                combined_signal, current_position['entry_price'], exit_price,
                                self.get_account_balance(client), pnl_percent, self.ledger_data[ledger_key]['pnl_sum'],
                                actual_position_size, order_id
                            )
                            
                            # Remove from active positions
                            if position_key in self.active_positions:
                                del self.active_positions[position_key]
                            
                            logger.info(f"Position closed for {symbol}: {close_reason}, PnL {pnl_percent:.2f}%, Fee: {close_fee:.4f} USDT")
                    else:
                        # Same direction - log "same direction" action
                        if last_signal == combined_signal and combined_signal in [1, -1]:
                            # Get real current balance
                            real_balance = self.get_account_balance(client)
                            
                            # Update ledger with "same direction" entry
                            ledger_key = f"{user_id}_{strategy_config['name']}"
                            if ledger_key not in self.ledger_data:
                                self.ledger_data[ledger_key] = {'pnl_sum': 0.0, 'last_trade_amount': 1000.0}
                            
                            order_id = self.active_positions.get(position_key, {}).get('order_id', "")
                            trade_amount = self.active_positions.get(position_key, {}).get('trade_amount', 0.0)
                            
                            # Same direction - no PnL change, just log the current state
                            self.update_ledger(
                                user_id, strategy_config, "same_direction",
                                combined_signal, current_position['entry_price'], 0.0,
                                self.get_account_balance(client), 0.0, self.ledger_data[ledger_key]['pnl_sum'],
                                trade_amount, order_id
                            )
                            logger.info(f"Same direction for {symbol} - no action needed")
                        continue
                
                # Check if we should open new position
                if combined_signal in [1, -1] and last_signal != combined_signal and not has_open_order:
                    # Get real current balance for position sizing
                    real_balance = self.get_account_balance(client)
                    
                    # Calculate position size based on real balance
                    position_size = self.calculate_position_size(real_balance, strategy_config, user_id)
                    current_price = self.get_current_price(client, symbol)
                    
                    if current_price is None:
                        logger.error(f"Could not get current price for {symbol}")
                        continue
                    
                    # Calculate quantity
                    qty = position_size / current_price
                    
                    # Format quantity according to symbol requirements
                    qty = self.format_quantity(qty, symbol)
                    
                    # For XRP, ensure whole number quantity
                    if 'XRP' in symbol:
                        qty = int(qty)
                        if qty < 1:
                            qty = 1
                        logger.info(f"XRP quantity adjusted to whole number: {qty}")
                    
                    # Recalculate position size based on formatted quantity
                    actual_position_size = qty * current_price
                    logger.info(f"Actual position size: {actual_position_size:.2f} USDT for {qty} {symbol}")
                    
                    # Calculate TP/SL prices
                    tp_percent = strategy_config.get('take_profit', 0.05)
                    sl_percent = strategy_config.get('stop_loss', 0.03)
                    
                    if combined_signal == 1:  # Long
                        side = "Buy"
                        tp_price = current_price * (1 + tp_percent)
                        sl_price = current_price * (1 - sl_percent)
                    elif combined_signal == -1:  # Short
                        side = "Sell"
                        tp_price = current_price * (1 - tp_percent)
                        sl_price = current_price * (1 + sl_percent)
                    else:
                        continue  # Neutral signal
                    
                    # Check margin availability
                    if not self.check_margin_availability(client, symbol, qty, side):
                        logger.warning(f"Insufficient margin for {symbol} {side} {qty}. Skipping order.")
                        continue

                    # Place order
                    order_id = self.place_order(client, symbol, side, qty, tp_price, sl_price)
                    
                    if order_id:
                        # Get opening fee
                        open_fee = self.get_trading_fee(client, symbol, order_id)
                        
                        # Get balance after opening (including fee deduction)
                        balance_after_open = self.get_account_balance(client)
                        
                        # Update active positions
                        self.active_positions[position_key] = {
                            'last_signal': combined_signal,
                            'order_id': order_id,
                            'entry_price': current_price,
                            'side': side,
                            'qty': qty,
                            'trade_amount': actual_position_size
                        }
                        
                        # Calculate cumulative PnL
                        ledger_key = f"{user_id}_{strategy_config['name']}"
                        if ledger_key not in self.ledger_data:
                            self.ledger_data[ledger_key] = {'pnl_sum': 0.0, 'last_trade_amount': 1000.0}
                        
                        # Update the last trade amount for compounding (use the adjusted amount after PnL)
                        self.ledger_data[ledger_key]['last_trade_amount'] = actual_position_size
                        
                        # Add -0.05% fee to pnl_sum for opening trade
                        self.ledger_data[ledger_key]['pnl_sum'] += (-0.05)
                        
                        # Update ledger with real balance after opening
                        self.update_ledger(
                            user_id, strategy_config, "open",
                            combined_signal, current_price, 0.0,
                            self.get_account_balance(client), -0.05, self.ledger_data[ledger_key]['pnl_sum'],
                            actual_position_size, order_id
                        )
                        
                        logger.info(f"Position opened for {symbol}: {side} {qty} @ {current_price}, Fee: {open_fee:.4f} USDT")
                    else:
                        logger.error(f"Failed to place order for {symbol}")
                else:
                    if combined_signal == 0:
                        logger.info(f"Neutral signal for {symbol} - no action")
                    elif has_open_order:
                        logger.info(f"Open order exists for {symbol} - waiting for execution")
                    else:
                        logger.info(f"Same signal for {symbol} - no action needed")
            
        except Exception as e:
            logger.error(f"Error executing trading logic for user {user_config['name']}: {e}")
    
    def run_continuous_trading(self, user_id: int):
        """Run continuous trading for a user"""
        try:
            logger.info(f"Starting continuous trading for user {user_id}")
            
            while True:
                try:
                    # Get user configuration
                    user_config = self.get_user_config(user_id)
                    if user_config is None:
                        logger.error(f"Could not get user config for {user_id}")
                        time.sleep(300)  # Wait 5 minutes before retrying
                        continue
                    
                    # Execute trading logic
                    self.execute_trading_logic(user_id, user_config)
                    
                    # Wait before next iteration
                    time.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.info("Trading stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in continuous trading loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            logger.error(f"Error in continuous trading: {e}")

    def get_open_orders(self, client: HTTP, symbol: str) -> List[Dict[str, Any]]:
        """Get open orders for a symbol"""
        try:
            response = client.get_open_orders(
                category="linear",
                symbol=symbol
            )
            
            if response['retCode'] == 0:
                return response['result']['list']
            else:
                logger.error(f"Failed to get open orders: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def check_manual_closure(self, client: HTTP, symbol: str, last_order_id: str) -> bool:
        """Check if the last order was manually closed"""
        try:
            # First check if position still exists
            current_position = self.get_position(client, symbol)
            if current_position is None or float(current_position.get('size', 0)) == 0:
                logger.info(f"Position for {symbol} is closed (no position found)")
                return True
            
            # Get order history to check if order was cancelled
            orders = self.get_order_history(client, symbol, limit=50)
            
            for order in orders:
                if order['orderId'] == last_order_id:
                    # Check if order was cancelled (manual closure)
                    if order['orderStatus'] == 'Cancelled':
                        logger.info(f"Order {last_order_id} was manually cancelled")
                        return True
                    # Check if order was partially filled and then cancelled
                    elif (order['orderStatus'] == 'PartiallyFilled' and 
                          float(order.get('cumExecQty', 0)) > 0):
                        logger.info(f"Order {last_order_id} was partially filled and cancelled")
                        return True
                    # Check if order was fully filled but position is closed
                    elif order['orderStatus'] == 'Filled' and current_position is None:
                        logger.info(f"Order {last_order_id} was filled but position is closed (manual closure)")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking manual closure: {e}")
            return False
    
    def handle_manual_closure(self, client: HTTP, user_id: int, strategy_config: Dict[str, Any], 
                            symbol: str, position_key: str) -> bool:
        """Handle manual closure and update ledger"""
        try:
            if position_key not in self.active_positions:
                return False
            
            position_data = self.active_positions[position_key]
            order_id = position_data.get('order_id', "")
            entry_price = position_data.get('entry_price', 0)
            
            # Get current price for PnL calculation
            current_price = self.get_current_price(client, symbol) or entry_price
            
            # Calculate PnL (manual closure typically at market price)
            if position_data.get('side') == "Buy":  # Long position
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:  # Short position
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            
            # Get real balance after manual closure
            new_balance = self.get_account_balance(client)
            
            # Calculate cumulative PnL
            ledger_key = f"{user_id}_{strategy_config['name']}"
            if ledger_key not in self.ledger_data:
                self.ledger_data[ledger_key] = {'pnl_sum': 0.0, 'last_trade_amount': 1000.0}
            
            self.ledger_data[ledger_key]['pnl_sum'] += pnl_percent
            
            # Get trade amount
            trade_amount = position_data.get('trade_amount', 0)
            
            # Update ledger with manual closure
            self.update_ledger(
                user_id, strategy_config, "manually_closed",
                0, entry_price, current_price,  # signal 0 for manual closure
                new_balance, pnl_percent, self.ledger_data[ledger_key]['pnl_sum'],
                trade_amount, order_id
            )
            
            # Remove from active positions
            del self.active_positions[position_key]
            
            logger.info(f"Manual closure recorded for {symbol}: PnL {pnl_percent:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error handling manual closure: {e}")
            return False
    
    def get_execution_details(self, client: HTTP, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get execution details for a specific order"""
        try:
            response = client.get_executions(
                category="linear",
                symbol=symbol,
                orderId=order_id
            )
            
            if response['retCode'] == 0 and response['result']['list']:
                return response['result']['list'][0]  # Return first execution
            else:
                logger.error(f"Failed to get execution details: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting execution details: {e}")
            return None

    def check_margin_availability(self, client: HTTP, symbol: str, qty: float, side: str) -> bool:
        """Check if we have enough margin to place the order"""
        try:
            # Get wallet balance to check available funds
            balance_data = client.get_wallet_balance(accountType="UNIFIED")
            
            if balance_data['retCode'] == 0:
                for account in balance_data['result']['list']:
                    if account.get('coin'):
                        for coin in account['coin']:
                            if coin['coin'] == 'USDT':
                                # Use walletBalance as the primary balance source
                                wallet_balance = float(coin['walletBalance'])
                                
                                # Try to get availableToWithdraw, fallback to walletBalance if empty
                                available_str = coin.get('availableToWithdraw', '')
                                if available_str and available_str.strip():
                                    available_balance = float(available_str)
                                else:
                                    # If availableToWithdraw is empty, use walletBalance
                                    available_balance = wallet_balance
                                
                                # Get current price to calculate required margin
                                current_price = self.get_current_price(client, symbol)
                                if current_price is None:
                                    logger.warning(f"Could not get current price for {symbol}")
                                    return False
                                
                                # Calculate required margin (position value)
                                position_value = qty * current_price
                                
                                # Check if we have enough available balance
                                if available_balance >= position_value * 1.1:  # 10% buffer for fees
                                    logger.info(f"Sufficient margin: {available_balance:.2f} USDT available, {position_value:.2f} USDT required")
                                    return True
                                else:
                                    logger.warning(f"Insufficient margin: {available_balance:.2f} USDT available, {position_value:.2f} USDT required")
                                    return False
            
            logger.warning("Could not find USDT balance in wallet data")
            return False
            
        except Exception as e:
            logger.error(f"Error checking margin availability: {e}")
            return False

    def handle_position_closure(self, client: HTTP, user_id: int, strategy_config: Dict[str, Any], 
                              symbol: str, position_key: str, new_signal: int) -> bool:
        """Handle position closure and determine what happened"""
        try:
            if position_key not in self.active_positions:
                return False
            
            position_data = self.active_positions[position_key]
            order_id = position_data.get('order_id', "")
            entry_price = position_data.get('entry_price', 0)
            last_signal = position_data.get('last_signal', 0)
            trade_amount = position_data.get('trade_amount', 0)
            
            # Determine what happened to the position
            closure_reason = ""
            exit_price = 0.0
            
            # Check for TP/SL hit first
            closed_pnl_data = self.get_closed_pnl(client, symbol, limit=10)
            for pnl_record in closed_pnl_data:
                if pnl_record['side'] == position_data.get('side', '') and float(pnl_record['size']) > 0:
                    # Position was closed by TP/SL
                    closure_reason = "tp" if float(pnl_record.get('closedPnl', 0)) > 0 else "sl"
                    exit_price = float(pnl_record.get('execPrice', 0))
                    logger.info(f"TP/SL hit detected for {symbol}: {closure_reason}")
                    break
            
            # If not TP/SL, check for direction change
            if not closure_reason:
                if (new_signal == 1 and last_signal == -1) or (new_signal == -1 and last_signal == 1):
                    closure_reason = "direction_change"
                    exit_price = self.get_current_price(client, symbol) or entry_price
                    logger.info(f"Direction change detected for {symbol}")
                else:
                    # Must be manual closure
                    closure_reason = "manually_closed"
                    exit_price = self.get_current_price(client, symbol) or entry_price
                    logger.info(f"Manual closure detected for {symbol}")
            
            # Calculate PnL
            if position_data.get('side') == "Buy":  # Long position
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:  # Short position
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100
            
            # Get real balance after closure
            new_balance = self.get_account_balance(client)
            
            # Calculate cumulative PnL
            ledger_key = f"{user_id}_{strategy_config['name']}"
            if ledger_key not in self.ledger_data:
                self.ledger_data[ledger_key] = {'pnl_sum': 0.0, 'last_trade_amount': 1000.0}
            
            # Update pnl_sum with actual PnL (including the -0.05% fee)
            self.ledger_data[ledger_key]['pnl_sum'] += pnl_percent
            
            # Update ledger with closure
            self.update_ledger(
                user_id, strategy_config, closure_reason,
                new_signal, entry_price, exit_price,
                self.get_account_balance(client), pnl_percent, self.ledger_data[ledger_key]['pnl_sum'],
                trade_amount, order_id
            )
            
            # Remove from active positions
            del self.active_positions[position_key]
            
            logger.info(f"Position closure handled for {symbol}: {closure_reason}, PnL {pnl_percent:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error handling position closure: {e}")
            return False

    def check_old_trades_from_ledger(self, client: HTTP, user_id: int, strategy_config: Dict[str, Any]) -> bool:
        """Check old trades from ledger using order_id and update their status"""
        try:
            # Get the ledger table name
            table_name = self.get_ledger_table_name(user_id, strategy_config)
            
            # Query the ledger for the most recent 'buy' action (open trade)
            query = text(f"""
                SELECT order_id, buy_price, trade_amount, datetime 
                FROM execution.{table_name} 
                WHERE action = 'buy' 
                ORDER BY datetime DESC 
                LIMIT 1
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                row = result.fetchone()
                
                if row and row[0]:  # If we have an order_id
                    order_id = row[0]
                    buy_price = row[1]
                    trade_amount = row[2]
                    open_time = row[3]
                    
                    logger.info(f"Found old trade in ledger: order_id={order_id}, buy_price={buy_price}")
                    
                    # Check if this order was executed and closed
                    symbol = f"{strategy_config['symbol'].upper()}USDT"
                    
                    # Check for TP/SL hits first
                    closed_pnl_data = self.get_closed_pnl(client, symbol, limit=20)
                    for pnl_record in closed_pnl_data:
                        if pnl_record.get('orderId') == order_id:
                            # Order was closed by TP/SL
                            exit_price = float(pnl_record.get('execPrice', 0))
                            closed_pnl = float(pnl_record.get('closedPnl', 0))
                            closure_reason = "tp" if closed_pnl > 0 else "sl"
                            
                            # Calculate PnL
                            pnl_percent = ((exit_price - buy_price) / buy_price) * 100
                            
                            # Get real balance
                            new_balance = self.get_account_balance(client)
                            
                            # Update ledger data
                            ledger_key = f"{user_id}_{strategy_config['name']}"
                            if ledger_key not in self.ledger_data:
                                self.ledger_data[ledger_key] = {'pnl_sum': 0.0, 'last_trade_amount': 1000.0}
                            
                            self.ledger_data[ledger_key]['pnl_sum'] += pnl_percent
                            
                            # Update ledger with closure
                            self.update_ledger(
                                user_id, strategy_config, closure_reason,
                                0, buy_price, exit_price,  # signal 0 for closure
                                new_balance, pnl_percent, self.ledger_data[ledger_key]['pnl_sum'],
                                trade_amount, order_id
                            )
                            
                            logger.info(f"Updated ledger for old trade {order_id}: {closure_reason}, PnL {pnl_percent:.2f}%")
                            return True
                    
                    # Check if order was manually closed
                    orders = self.get_order_history(client, symbol, limit=50)
                    for order in orders:
                        if order['orderId'] == order_id:
                            if order['orderStatus'] == 'Cancelled':
                                # Order was manually cancelled
                                current_price = self.get_current_price(client, symbol) or buy_price
                                
                                # Calculate PnL at current price
                                pnl_percent = ((current_price - buy_price) / buy_price) * 100
                                
                                # Get real balance
                                new_balance = self.get_account_balance(client)
                                
                                # Update ledger data
                                ledger_key = f"{user_id}_{strategy_config['name']}"
                                if ledger_key not in self.ledger_data:
                                    self.ledger_data[ledger_key] = {'pnl_sum': 0.0, 'last_trade_amount': 1000.0}
                                
                                self.ledger_data[ledger_key]['pnl_sum'] += pnl_percent
                                
                                # Update ledger with manual closure
                                self.update_ledger(
                                    user_id, strategy_config, "manually_closed",
                                    0, buy_price, current_price,  # signal 0 for closure
                                    new_balance, pnl_percent, self.ledger_data[ledger_key]['pnl_sum'],
                                    trade_amount, order_id
                                )
                                
                                logger.info(f"Updated ledger for manually closed trade {order_id}: PnL {pnl_percent:.2f}%")
                                return True
                    
                    # Check if position still exists
                    current_position = self.get_position(client, symbol)
                    if current_position is None or float(current_position.get('size', 0)) == 0:
                        # Position was closed but we couldn't determine how - assume manual closure
                        current_price = self.get_current_price(client, symbol) or buy_price
                        
                        # Calculate PnL at current price
                        pnl_percent = ((current_price - buy_price) / buy_price) * 100
                        
                        # Get real balance
                        new_balance = self.get_account_balance(client)
                        
                        # Update ledger data
                        ledger_key = f"{user_id}_{strategy_config['name']}"
                        if ledger_key not in self.ledger_data:
                            self.ledger_data[ledger_key] = {'pnl_sum': 0.0, 'last_trade_amount': 1000.0}
                        
                        self.ledger_data[ledger_key]['pnl_sum'] += pnl_percent
                        
                        # Update ledger with manual closure
                        self.update_ledger(
                            user_id, strategy_config, "manually_closed",
                            0, buy_price, current_price,  # signal 0 for closure
                            new_balance, pnl_percent, self.ledger_data[ledger_key]['pnl_sum'],
                            trade_amount, order_id
                        )
                        
                        logger.info(f"Updated ledger for closed trade {order_id} (unknown reason): PnL {pnl_percent:.2f}%")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking old trades from ledger: {e}")
            return False

def main():
    """Main function"""
    try:
        # Initialize unified signal generator
        generator = UnifiedSignalGenerator()
        
        # Example: Run for user ID 1
        user_id = 1
        generator.run_continuous_trading(user_id)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 