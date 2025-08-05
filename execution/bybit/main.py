import os
import pandas as pd
import datetime
import time
import logging
import sys
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from sqlalchemy import text, MetaData, Table, Column, String, DateTime, Float, Integer
from data.utils.db_utils import get_pg_engine
from strategies.strategy_pipeline.signal_generator import StrategySignalGenerator

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bybit_trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BybitTrader:
    def __init__(self, strategy_name="strategy_02"):
        """Initialize Bybit trader with demo account"""
        self.api_key = os.getenv('bybit_api')
        self.api_secret = os.getenv('bybit_secret')
        self.strategy_name = strategy_name
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Please set bybit_api and bybit_secret environment variables")
        
        # Initialize client with demo account and proper settings
        self.client = HTTP(
            demo=True,  # Use demo account
            api_key=self.api_key,
            api_secret=self.api_secret,
            recv_window=20000  # Large recv_window to handle time differences
        )
        
        # Initialize database connection
        self.engine = get_pg_engine()
        self.metadata = MetaData()
        
        self.active_positions = {}
        self.initial_balance = 1000.0  # Start with $1,000 like in CSV
        self.current_balance = self.initial_balance  # Track current balance
        self.cumulative_pnl = 0.0  # Track cumulative PnL
        
        # Create ledger table for this strategy
        self._create_ledger_table()
        self._init_ledger()
        logger.info("BybitTrader initialized successfully")
    
    def _create_ledger_table(self):
        """Create ledger table for the strategy in database"""
        try:
            # Create ledger table schema
            ledger_table = Table(
                f'ledger_{self.strategy_name}', 
                self.metadata,
                Column('datetime', DateTime, primary_key=True),
                Column('predicted_direction', String),
                Column('action', String),
                Column('buy_price', Float),
                Column('sell_price', Float),
                Column('balance', Float),
                Column('pnl', Float),
                Column('pnl_sum', Float),
                schema='execution'
            )
            
            # Create schema if not exists
            with self.engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS execution;"))
                conn.commit()
            
            # Create table if not exists
            self.metadata.create_all(self.engine, tables=[ledger_table])
            logger.info(f"Created ledger table: execution.ledger_{self.strategy_name}")
            
        except Exception as e:
            logger.error(f"Error creating ledger table: {e}")
    
    def _init_ledger(self):
        """Initialize ledger from database"""
        try:
            # Check if ledger table has data
            query = text(f"SELECT * FROM execution.ledger_{self.strategy_name} ORDER BY datetime DESC LIMIT 1")
            result = pd.read_sql_query(query, self.engine)
            
            if not result.empty:
                last_row = result.iloc[-1]
                self.current_balance = last_row['balance']
                self.cumulative_pnl = last_row['pnl_sum']
                logger.info(f"Loaded existing ledger - Balance: ${self.current_balance:.2f}, PnL: {self.cumulative_pnl:.2f}%")
            else:
                logger.info(f"No existing ledger data found for {self.strategy_name}, starting fresh")
                
        except Exception as e:
            logger.warning(f"Could not load existing ledger: {e}, using defaults")
    
    def _get_server_time(self):
        """Get server time to sync with local time"""
        try:
            response = self.client.get_server_time()
            return int(response['result']['timeSecond'])
        except Exception as e:
            logger.error(f"Error getting server time: {e}")
            return int(time.time())
    
    def _sync_time(self):
        """Sync local time with server time"""
        server_time = self._get_server_time()
        local_time = int(time.time())
        diff = abs(server_time - local_time)
        
        if diff > 1000:
            logger.warning(f"Time difference detected: {diff}ms")
        
        return server_time
    
    def _get_account_balance(self):
        """Get current account balance from Bybit demo account"""
        try:
            self._sync_time()  # Sync time before API call
            
            # Get wallet balance
            balance_data = self.client.get_wallet_balance(accountType="UNIFIED")
            
            if balance_data['retCode'] == 0:
                for coin in balance_data['result']['list']:
                    if coin['coin'] == 'USDT':
                        balance = round(float(coin['walletBalance']), 2)
                        logger.info(f"Live USDT balance from Bybit: {balance}")
                        # Update our local balance to match Bybit
                        self.current_balance = balance
                        return balance
            
            # Try alternative balance endpoint
            try:
                account_data = self.client.get_account_info(accountType="UNIFIED")
                if account_data['retCode'] == 0 and account_data['result']['list']:
                    balance = round(float(account_data['result']['list'][0]['totalWalletBalance']), 2)
                    logger.info(f"Live balance (alternative): {balance} USDT")
                    # Update our local balance to match Bybit
                    self.current_balance = balance
                    return balance
            except Exception as e:
                logger.warning(f"Alternative balance check failed: {e}")
            
            logger.warning("Could not fetch balance from Bybit, using local balance")
            return round(self.current_balance, 2)
            
        except Exception as e:
            logger.error(f"Error getting balance from Bybit: {e}")
            return round(self.current_balance, 2)
    
    def _get_trading_fees(self, symbol):
        """Get trading fees for a symbol from Bybit"""
        try:
            self._sync_time()  # Sync time before API call
            
            # Try to get fee rates from Bybit
            fee_data = self.client.get_fee_rates(category="linear", symbol=symbol)
            
            if fee_data['retCode'] == 0 and fee_data['result']['list']:
                taker_fee = float(fee_data['result']['list'][0]['takerFeeRate'])
                maker_fee = float(fee_data['result']['list'][0]['makerFeeRate'])
                logger.info(f"Trading fees for {symbol}: Taker={taker_fee*100:.4f}%, Maker={maker_fee*100:.4f}%")
                return taker_fee, maker_fee
            else:
                # Use default Bybit fees
                default_taker = 0.0006  # 0.06%
                default_maker = 0.0001  # 0.01%
                logger.info(f"Using default fees for {symbol}: Taker={default_taker*100:.4f}%, Maker={default_maker*100:.4f}%")
                return default_taker, default_maker
                
        except Exception as e:
            logger.error(f"Error getting trading fees: {e}")
            # Use default Bybit fees
            default_taker = 0.0006  # 0.06%
            default_maker = 0.0001  # 0.01%
            return default_taker, default_maker
    
    def _get_current_price(self, symbol):
        """Get current market price"""
        try:
            self._sync_time()  # Sync time before API call
            ticker = self.client.get_tickers(category="linear", symbol=symbol)
            
            if ticker['retCode'] == 0 and ticker['result']['list']:
                price = float(ticker['result']['list'][0]['lastPrice'])
                logger.info(f"Live price for {symbol}: {price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def _set_leverage(self, symbol, leverage=1):
        """Set leverage for a symbol"""
        try:
            self._sync_time()  # Sync time before API call
            
            response = self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if response['retCode'] == 0:
                logger.info(f"Leverage set to {leverage}x for {symbol}")
                return True
            else:
                logger.warning(f"Failed to set leverage: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False
    
    def _get_position(self, symbol):
        """Get current position for symbol with detailed info"""
        try:
            self._sync_time()  # Sync time before API call
            positions = self.client.get_positions(category="linear", symbol=symbol)
            
            if positions['retCode'] == 0:
                for pos in positions['result']['list']:
                    if float(pos['size']) > 0:
                        position_info = {
                            'size': float(pos['size']),
                            'side': pos['side'],
                            'entry_price': float(pos['avgPrice']),
                            'unrealized_pnl': float(pos.get('unrealisedPnl', 0)),
                            'realized_pnl': float(pos.get('realisedPnl', 0)),
                            'leverage': float(pos.get('leverage', 1)),
                            'margin_mode': pos.get('marginMode', 'REGULAR_MARGIN'),
                            'position_value': float(pos.get('positionValue', 0)),
                            'mark_price': float(pos.get('markPrice', 0))
                        }
                        logger.info(f"Live position from Bybit: {position_info}")
                        return position_info
            return None
            
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None
    
    def _place_order(self, symbol, side, qty, tp_price=None, sl_price=None):
        """Place order with TP/SL"""
        try:
            self._sync_time()  # Sync time before API call
            
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "GTC"
            }
            
            if tp_price:
                order_params["takeProfit"] = str(round(tp_price, 2))
            if sl_price:
                order_params["stopLoss"] = str(round(sl_price, 2))
            
            response = self.client.place_order(**order_params)
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                logger.info(f"Order placed successfully: {side} {qty} {symbol}, ID: {order_id}")
                return order_id, True
            else:
                logger.error(f"Order failed: {response}")
                return None, False
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None, False
    
    def _close_position(self, symbol, side, qty):
        """Close existing position"""
        try:
            self._sync_time()  # Sync time before API call
            
            # First check if position actually exists
            current_position = self._get_position(symbol)
            if not current_position:
                logger.info(f"No position found for {symbol} - nothing to close")
                return True
            
            # Check if position size is valid
            if current_position['size'] <= 0:
                logger.info(f"Position size is zero for {symbol} - nothing to close")
                return True
            
            # Check if position size matches
            if abs(current_position['size'] - qty) > 0.001:  # Allow small rounding differences
                logger.warning(f"Position size mismatch: expected {qty}, actual {current_position['size']}")
                qty = current_position['size']  # Use actual size
            
            # Check if position side matches
            if current_position['side'] != side:
                logger.warning(f"Position side mismatch: expected {side}, actual {current_position['side']}")
                side = current_position['side']  # Use actual side
            
            close_side = "Sell" if side == "Buy" else "Buy"
            
            logger.info(f"Closing position: {close_side} {qty} {symbol}")
            
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=str(qty),
                timeInForce="GTC",
                reduceOnly=True
            )
            
            if response['retCode'] == 0:
                logger.info(f"Position closed: {close_side} {qty} {symbol}")
                return True
            else:
                logger.error(f"Failed to close position: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def _update_ledger(self, trade_data):
        """Update ledger with trade data in database"""
        try:
            # Insert new trade data into database
            query = text(f"""
                INSERT INTO execution.ledger_{self.strategy_name} 
                (datetime, predicted_direction, action, buy_price, sell_price, balance, pnl, pnl_sum)
                VALUES (:datetime, :predicted_direction, :action, :buy_price, :sell_price, :balance, :pnl, :pnl_sum)
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, trade_data)
                conn.commit()
            
            logger.info(f"Ledger updated with live data: {trade_data}")
        except Exception as e:
            logger.error(f"Error updating ledger in database: {e}")
    
    def get_ledger_data(self):
        """Get all ledger data from database"""
        try:
            query = text(f"SELECT * FROM execution.ledger_{self.strategy_name} ORDER BY datetime")
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error getting ledger data: {e}")
            return pd.DataFrame()
    
    def export_ledger_to_csv(self, filename=None):
        """Export ledger data to CSV file"""
        try:
            df = self.get_ledger_data()
            if not df.empty:
                if filename is None:
                    filename = f"ledger_{self.strategy_name}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Ledger exported to {filename}")
                return filename
            else:
                logger.warning("No ledger data to export")
                return None
        except Exception as e:
            logger.error(f"Error exporting ledger: {e}")
            return None

    def take_trade(self, symbol, signal, tp_percent, sl_percent, position_size_usdt=1000):
        """
        Take a trade based on signal with TP/SL management
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            signal: 'long' or 'short'
            tp_percent: Take profit percentage (e.g., 0.05 for 5%)
            sl_percent: Stop loss percentage (e.g., 0.03 for 3%)
            position_size_usdt: Position size in USDT (this will be the contract value)
        """
        try:
            logger.info(f"Taking trade: {symbol} {signal}, TP: {tp_percent*100}%, SL: {sl_percent*100}%")
            
            # Get current price
            current_price = self._get_current_price(symbol)
            
            if not current_price:
                logger.error("Failed to get current price")
                return False
            
            # Get trading fees
            taker_fee, maker_fee = self._get_trading_fees(symbol)
            
            # Check if we have sufficient balance
            if self.current_balance < position_size_usdt:
                logger.warning(f"Insufficient balance: {self.current_balance} USDT < {position_size_usdt} USDT required")
                logger.info("Using available balance for position size")
                position_size_usdt = self.current_balance * 0.95  # Use 95% of available balance
            
            # Check for existing position and close it if different direction
            existing_position = self._get_position(symbol)
            if existing_position:
                logger.info(f"Closing existing position: {existing_position}")
                if not self._close_position(symbol, existing_position['side'], existing_position['size']):
                    logger.error("Failed to close existing position")
                    return False
            
            # Calculate position size for leverage trading
            # For leverage trading, position_size_usdt is the contract value
            # We need to calculate the quantity based on the contract value
            qty = position_size_usdt / current_price
            qty = round(qty, 3)  # Round to 3 decimal places
            
            # Ensure minimum order size
            if qty < 0.001:
                qty = 0.001
                logger.warning(f"Quantity adjusted to minimum: {qty}")
                # Recalculate position size based on minimum quantity
                position_size_usdt = qty * current_price
            
            # Determine order side
            side = "Buy" if signal == "long" else "Sell"
            
            # Calculate TP/SL prices based on entry price (current price)
            if signal == "long":
                tp_price = current_price * (1 + tp_percent)
                sl_price = current_price * (1 - sl_percent)
            else:  # short
                tp_price = current_price * (1 - tp_percent)
                sl_price = current_price * (1 + sl_percent)
            
            logger.info(f"Entry price: {current_price}")
            logger.info(f"Intended position size: {position_size_usdt} USDT")
            logger.info(f"Quantity: {qty} {symbol}")
            logger.info(f"TP price: {tp_price} ({tp_percent*100}%)")
            logger.info(f"SL price: {sl_price} ({sl_percent*100}%)")
            logger.info(f"Bybit will automatically close position at TP/SL levels")
            
            # Place the order with TP/SL - Bybit will handle the TP/SL automatically
            order_id, success = self._place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                tp_price=tp_price,
                sl_price=sl_price
            )
            
            if success:
                # Get the actual position details from Bybit
                time.sleep(2)  # Wait a moment for position to be created
                actual_position = self._get_position(symbol)
                
                if actual_position:
                    # Use actual position value from Bybit
                    actual_position_value = actual_position['position_value']
                    actual_qty = actual_position['size']
                    logger.info(f"Actual position created: {actual_qty} {symbol} = ${actual_position_value:.2f} USDT")
                else:
                    # Fallback to intended values
                    actual_position_value = position_size_usdt
                    actual_qty = qty
                    logger.warning("Could not get actual position details, using intended values")
                
                # Calculate fee cost (based on actual position value)
                fee_cost = actual_position_value * taker_fee
                new_balance = round(self.current_balance - fee_cost, 2)
                
                # Update current balance
                self.current_balance = new_balance
                
                # Record trade in ledger - BUY action for opening position
                trade_data = {
                    'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_direction': signal,
                    'action': 'buy',  # Always 'buy' when opening position
                    'buy_price': current_price,  # Entry price
                    'sell_price': 0.0,  # Will be filled when position closes
                    'balance': new_balance,
                    'pnl': -taker_fee * 100,  # Entry fee as negative PnL
                    'pnl_sum': self.cumulative_pnl - (taker_fee * 100)  # Cumulative PnL
                }
                
                self._update_ledger(trade_data)
                
                # Store active position info with actual values
                self.active_positions[symbol] = {
                    'entry_price': current_price,
                    'qty': actual_qty,
                    'side': side,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'signal': signal,
                    'order_id': order_id,
                    'position_size_usdt': actual_position_value,  # Actual contract value from Bybit
                    'fee_cost': fee_cost
                }
                
                logger.info(f"Trade opened successfully: {symbol} {signal} at {current_price}")
                logger.info(f"Actual contract value: ${actual_position_value:.2f} USDT")
                logger.info(f"Actual quantity: {actual_qty} {symbol}")
                logger.info(f"Fee cost: ${fee_cost:.2f} USDT")
                logger.info(f"New balance: ${new_balance:.2f} USDT")
                return True
            else:
                logger.error("Failed to place order")
                return False
                
        except Exception as e:
            logger.error(f"Error taking trade: {e}")
            return False
    
    def check_positions(self):
        """Check and monitor active positions for manual closures only"""
        try:
            for symbol, position in list(self.active_positions.items()):
                # Get live position from Bybit
                actual_position = self._get_position(symbol)
                
                if not actual_position or actual_position['size'] <= 0:
                    # Position was closed manually or by TP/SL
                    logger.info(f"Position for {symbol} was closed (manually or by TP/SL)")
                    
                    # Get live balance
                    live_balance = self._get_account_balance()
                    
                    # Update ledger with closure
                    trade_data = {
                        'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'predicted_direction': position['signal'],
                        'action': 'sell - position_closed',
                        'buy_price': position['entry_price'],
                        'sell_price': self._get_current_price(symbol) or position['entry_price'],
                        'balance': round(live_balance, 2),
                        'pnl': 0.0,  # Will be calculated from actual PnL
                        'pnl_sum': self.cumulative_pnl
                    }
                    
                    self._update_ledger(trade_data)
                    
                    # Remove from active positions
                    del self.active_positions[symbol]
                    logger.info(f"Removed {symbol} from active positions tracking")
                    continue
                
                # Position still exists - no action needed, let Bybit handle TP/SL
                logger.debug(f"Position still active: {symbol} {position['signal']}")
                        
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    def _close_position_by_tp_sl(self, symbol, reason, exit_price):
        """This method is no longer needed - Bybit handles TP/SL automatically"""
        logger.info(f"TP/SL handled by Bybit automatically - no manual intervention needed")
        return True
    
    def force_close_position(self, symbol):
        """Force close position at current market price (for signal change)"""
        try:
            position = self._get_position(symbol)
            if position:
                logger.info(f"Force closing position due to signal change: {position}")
                
                # Check if position size is valid
                if position['size'] <= 0:
                    logger.info(f"Position size is zero for {symbol} - nothing to close")
                    # Remove from active positions
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
                        logger.info(f"Removed {symbol} from active positions tracking")
                    return True
                
                # Close the position at market price
                if self._close_position(symbol, position['side'], position['size']):
                    # Get current price for ledger
                    current_price = self._get_current_price(symbol)
                    if current_price:
                        # Get actual position details from active positions
                        active_position = self.active_positions.get(symbol)
                        if active_position:
                            # Use stored position data for accurate calculations
                            entry_price = active_position['entry_price']
                            position_size_usdt = active_position['position_size_usdt']  # Actual contract value
                        else:
                            # Fallback to current position data
                            entry_price = position['entry_price']
                            position_size_usdt = position['position_value']  # Contract value
                        
                        if position['side'] == 'Buy':  # Long position
                            gross_pnl_percent = (current_price - entry_price) / entry_price
                        else:  # Short position
                            gross_pnl_percent = (entry_price - current_price) / entry_price
                        
                        # Get trading fees for exit
                        taker_fee, _ = self._get_trading_fees(symbol)
                        exit_fee = position_size_usdt * taker_fee
                        net_pnl_usdt = (position_size_usdt * gross_pnl_percent) - exit_fee
                        net_pnl_percent = net_pnl_usdt / position_size_usdt
                        
                        # Update balance and cumulative PnL
                        self.current_balance = round(self.current_balance + net_pnl_usdt, 2)
                        self.cumulative_pnl += net_pnl_percent * 100
                        
                        # Update ledger - SELL action for closing position
                        trade_data = {
                            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'predicted_direction': 'long' if position['side'] == 'Buy' else 'short',
                            'action': 'sell - direction_change',  # SELL when closing position
                            'buy_price': entry_price,  # Original entry price
                            'sell_price': current_price,  # Exit price
                            'balance': self.current_balance,
                            'pnl': net_pnl_percent * 100,
                            'pnl_sum': self.cumulative_pnl
                        }
                        
                        self._update_ledger(trade_data)
                        
                        # Remove from active positions
                        if symbol in self.active_positions:
                            del self.active_positions[symbol]
                        
                        logger.info(f"Position force closed: {symbol} at {current_price}")
                        logger.info(f"Actual contract value: ${position_size_usdt:.2f} USDT")
                        logger.info(f"Net PnL: {net_pnl_percent*100:.2f}% (${net_pnl_usdt:.2f})")
                        logger.info(f"Exit fee: ${exit_fee:.2f}")
                        logger.info(f"New balance: ${self.current_balance:.2f}")
                        logger.info(f"Cumulative PnL: {self.cumulative_pnl:.2f}%")
                        return True
                
                logger.error(f"Failed to force close position: {symbol}")
                return False
            else:
                logger.info(f"No position to close for {symbol}")
                # Remove from active positions if it was tracked but doesn't exist
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
                    logger.info(f"Removed {symbol} from active positions tracking")
                return True
                
        except Exception as e:
            logger.error(f"Error force closing position: {e}")
            return False

    def add_same_direction_entry(self, symbol, signal):
        """Add a 'same direction' entry to ledger when signal is the same"""
        try:
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                
                # Get current price
                current_price = self._get_current_price(symbol)
                if not current_price:
                    return
                
                # Add same direction entry to ledger
                trade_data = {
                    'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_direction': signal,
                    'action': 'same direction',
                    'buy_price': position['entry_price'],  # Original entry price
                    'sell_price': 0.0,  # No sell yet
                    'balance': round(self.current_balance, 2),
                    'pnl': 0.0,  # No PnL change
                    'pnl_sum': self.cumulative_pnl
                }
                
                self._update_ledger(trade_data)
                logger.info(f"Added same direction entry for {symbol} {signal}")
                
        except Exception as e:
            logger.error(f"Error adding same direction entry: {e}")

    def get_current_position_signal(self, symbol):
        """Get current position signal by checking actual position in Bybit"""
        try:
            # Get live position from Bybit
            actual_position = self._get_position(symbol)
            
            if actual_position and actual_position['size'] > 0:
                # Position exists in Bybit
                signal = 'long' if actual_position['side'] == 'Buy' else 'short'
                logger.info(f"Live position found: {signal} {actual_position['size']} {symbol}")
                
                # Update our tracking with live data
                self.active_positions[symbol] = {
                    'entry_price': actual_position['entry_price'],
                    'qty': actual_position['size'],
                    'side': actual_position['side'],
                    'tp_price': 0,  # Will be updated if we have TP/SL info
                    'sl_price': 0,  # Will be updated if we have TP/SL info
                    'signal': signal,
                    'order_id': 'live_position',
                    'position_size_usdt': actual_position['position_value'],
                    'fee_cost': 0  # Will be calculated when needed
                }
                
                return signal
            else:
                # No position in Bybit, remove from tracking if exists
                if symbol in self.active_positions:
                    logger.info(f"No live position found, removing from tracking")
                    del self.active_positions[symbol]
                return None
            
        except Exception as e:
            logger.error(f"Error getting current position signal: {e}")
            return None
    
    def _get_trade_history(self, symbol, limit=10):
        """Get recent trade history from Bybit - REMOVED - not needed"""
        # This method is no longer needed - we only track new trades
        return []
    
    def _update_ledger_from_bybit_trades(self, symbol):
        """Update ledger with actual trade data from Bybit - REMOVED - not needed"""
        # This method is no longer needed - we only track new trades
        logger.info(f"Not fetching old trades from Bybit - only tracking new trades")
        return
    
    def should_generate_signal(self):
        """Check if it's time to generate signal (every 4 hours)"""
        current_time = datetime.datetime.now()
        # Generate signal every 4 hours (0, 4, 8, 12, 16, 20)
        return current_time.hour % 4 == 0 and current_time.minute == 0

    def get_signal_from_strategy(self):
        """Get signal from strategy generator"""
        try:
            generator = StrategySignalGenerator()
            
            signal = generator.generate_latest_signal(self.strategy_name)
            
            # Convert signal to trading direction
            if signal == 1:
                return "long"
            elif signal == -1:
                return "short"
            elif signal == 0:
                return "hold"  # Special signal to hold current position
            else:
                logger.warning(f"Unexpected signal value: {signal} (type: {type(signal)})")
                return None
                
        except Exception as e:
            logger.error(f"Error getting signal from strategy: {e}")
            return None

    def main(self):
        """Main function with position management logic"""
        try:
            # ============================================
            # CONFIGURE YOUR TRADING PARAMETERS HERE
            # ============================================
            symbol = "BTCUSDT"          # Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
            tp_percent = 0.05          # Take profit: 0.05 = 5%
            sl_percent = 0.03          # Stop loss: 0.03 = 3%
            position_size_usdt = 1000  # Position size in USDT (contract value)
            leverage = 1               # Leverage (1 = no leverage, 10 = 10x, etc.)
            # ============================================
            
            logger.info("=== Starting Bybit Trading Bot with Database Ledger ===")
            logger.info(f"Strategy: {self.strategy_name}")
            logger.info(f"Trading parameters: {symbol}, TP: {tp_percent*100}%, SL: {sl_percent*100}%")
            logger.info(f"Position size: ${position_size_usdt} USDT (contract value)")
            logger.info(f"Leverage: {leverage}x")
            logger.info("Signal generation: Every 4 hours (0:00, 4:00, 8:00, 12:00, 16:00, 20:00)")
            logger.info(f"Starting balance: ${self.initial_balance:.2f} USDT")
            logger.info(f"Current balance: ${self.current_balance:.2f} USDT")
            logger.info(f"Current PnL: {self.cumulative_pnl:.2f}%")
            logger.info(f"Ledger table: execution.ledger_{self.strategy_name}")
            
            # Set leverage for the symbol
            logger.info(f"Setting leverage to {leverage}x for {symbol}")
            self._set_leverage(symbol, leverage)
            
            last_signal = None
            last_signal_time = None
            first_run = True  # Flag to track first run
            
            while True:
                try:
                    current_time = datetime.datetime.now()
                    
                    # Check if we need to generate a new signal (every 4 hours OR first run)
                    if self.should_generate_signal() or first_run:
                        if first_run:
                            logger.info(f"First run detected - generating signal immediately at {current_time.strftime('%H:%M')}")
                            first_run = False
                        else:
                            logger.info(f"Signal generation time: {current_time.strftime('%H:%M')}")
                        
                        # Get current position signal
                        current_position_signal = self.get_current_position_signal(symbol)
                        
                        # Generate new signal
                        new_signal = self.get_signal_from_strategy()
                        
                        if new_signal is not None:
                            logger.info(f"New signal generated: {new_signal}")
                            
                            # Get current position signal (check if position exists in Bybit)
                            current_position_signal = self.get_current_position_signal(symbol)
                            
                            # Position management logic
                            if current_position_signal is None:
                                # No current position exists in Bybit
                                if new_signal == "hold":
                                    logger.info("No current position and signal is HOLD - no action needed")
                                    # Add a neutral entry to ledger
                                    trade_data = {
                                        'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'predicted_direction': 'neutral',
                                        'action': 'no_position_hold',
                                        'buy_price': 0.0,
                                        'sell_price': 0.0,
                                        'balance': round(self._get_account_balance(), 2),
                                        'pnl': 0.0,
                                        'pnl_sum': self.cumulative_pnl
                                    }
                                    self._update_ledger(trade_data)
                                else:
                                    # No position exists - open new position
                                    logger.info(f"No current position - opening new {new_signal} position")
                                    success = self.take_trade(symbol, new_signal, tp_percent, sl_percent, position_size_usdt)
                                    if success:
                                        last_signal = new_signal
                                        last_signal_time = current_time
                                    else:
                                        logger.error("Failed to open new position")
                                    
                            elif new_signal == "hold":
                                # Signal is HOLD - keep current position (let Bybit handle TP/SL)
                                logger.info(f"HOLD signal - keeping current {current_position_signal} position")
                                logger.info(f"   Current position: {current_position_signal}")
                                logger.info(f"   New signal: {new_signal}")
                                logger.info(f"   Action: HOLD position (let Bybit handle TP/SL)")
                                last_signal = new_signal
                                last_signal_time = current_time
                                    
                            elif current_position_signal != new_signal:
                                # Signal change - close current position and open new one
                                logger.info(f"Signal change detected!")
                                logger.info(f"   Current position: {current_position_signal}")
                                logger.info(f"   New signal: {new_signal}")
                                logger.info(f"   Action: Close current position and open new {new_signal} position")
                                
                                # Force close current position
                                if self.force_close_position(symbol):
                                    # Open new position
                                    success = self.take_trade(symbol, new_signal, tp_percent, sl_percent, position_size_usdt)
                                    if success:
                                        last_signal = new_signal
                                        last_signal_time = current_time
                                        logger.info(f"Successfully switched to {new_signal} position")
                                    else:
                                        logger.error("Failed to open new position after signal change")
                                else:
                                    logger.error("Failed to close current position")
                                    
                            else:
                                # Same signal - add same direction entry to ledger
                                logger.info(f"Same signal - holding {current_position_signal} position")
                                logger.info(f"   Current position: {current_position_signal}")
                                logger.info(f"   New signal: {new_signal}")
                                logger.info(f"   Action: HOLD position (adding same direction entry)")
                                
                                # Add same direction entry to ledger
                                self.add_same_direction_entry(symbol, new_signal)
                                
                                last_signal = new_signal
                                last_signal_time = current_time
                        else:
                            logger.info("No new signal generated (signal generator returned None)")
                            
                    else:
                        # Not signal generation time - just monitor positions
                        if current_time.minute % 30 == 0:  # Log every 30 minutes
                            logger.info(f"Monitoring positions... {current_time.strftime('%H:%M')}")
                            
                            # Get live balance and position data
                            live_balance = self._get_account_balance()
                            current_position = self.get_current_position_signal(symbol)
                            
                            logger.info(f"Live balance: ${round(live_balance, 2):.2f} USDT")
                            logger.info(f"Current position: {current_position}")
                            logger.info(f"Current PnL: {self.cumulative_pnl:.2f}%")
                            
                            # Check for manual closures and update ledger
                            if current_position is None and symbol in self.active_positions:
                                logger.info("Position was closed manually - updating ledger")
                                # Position was closed - just remove from tracking
                                del self.active_positions[symbol]
                                logger.info(f"Removed {symbol} from active positions tracking")
                    
                    # Always check positions for TP/SL and manual closures
                    self.check_positions()
                    
                    # Sync with Bybit data every 5 minutes
                    if current_time.minute % 5 == 0:
                        logger.info("Syncing with Bybit data...")
                        # Update balance
                        self._get_account_balance()
                        # Check positions
                        self.get_current_position_signal(symbol)
                        # No need to fetch old trades
                        logger.info("Position and balance synced with Bybit")
                    
                    # Wait before next iteration
                    time.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.info("Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            logger.error(f"Error in main: {e}")


def list_strategy_ledgers():
    """List all available strategy ledgers in the database"""
    try:
        engine = get_pg_engine()
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'execution' 
            AND table_name LIKE 'ledger_%'
            ORDER BY table_name
        """)
        
        result = pd.read_sql_query(query, engine)
        
        if result.empty:
            print("No strategy ledgers found in database")
            return []
        
        print("Available Strategy Ledgers:")
        print("=" * 50)
        for _, row in result.iterrows():
            strategy_name = row['table_name'].replace('ledger_', '')
            print(f"  - {strategy_name}")
        
        return result['table_name'].tolist()
        
    except Exception as e:
        print(f"Error listing strategy ledgers: {e}")
        return []


def get_strategy_ledger_data(strategy_name):
    """Get ledger data for a specific strategy"""
    try:
        engine = get_pg_engine()
        query = text(f"SELECT * FROM execution.ledger_{strategy_name} ORDER BY datetime")
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        print(f"Error getting ledger data for {strategy_name}: {e}")
        return pd.DataFrame()


def export_strategy_ledger(strategy_name, filename=None):
    """Export strategy ledger to CSV"""
    try:
        df = get_strategy_ledger_data(strategy_name)
        if not df.empty:
            if filename is None:
                filename = f"ledger_{strategy_name}.csv"
            df.to_csv(filename, index=False)
            print(f"Ledger exported to {filename}")
            return filename
        else:
            print(f"No ledger data found for {strategy_name}")
            return None
    except Exception as e:
        print(f"Error exporting ledger: {e}")
        return None


def main(strategy_name="strategy_02"):
    """Main function to start the trading bot"""
    try:
        # Initialize trader with strategy name
        trader = BybitTrader(strategy_name)
        
        # Start the main trading loop
        trader.main()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use strategy name from command line argument
        strategy_name = sys.argv[1]
        print(f"Starting trading bot for strategy: {strategy_name}")
        main(strategy_name)
    else:
        # Use default strategy
        print("Starting trading bot with default strategy: strategy_02")
        print("Usage: python main.py <strategy_name>")
        print("Example: python main.py strategy_01")
        main() 