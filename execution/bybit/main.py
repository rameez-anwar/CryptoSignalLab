import os
import pandas as pd
import datetime
import time
import logging
import sys
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
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
    def __init__(self):
        """Initialize Bybit trader with demo account"""
        self.api_key = os.getenv('bybit_api')
        self.api_secret = os.getenv('bybit_secret')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Please set bybit_api and bybit_secret environment variables")
        
        # Initialize client with demo account and proper settings
        self.client = HTTP(
            demo=True,  # Use demo account
            api_key=self.api_key,
            api_secret=self.api_secret,
            recv_window=20000  # Large recv_window to handle time differences
        )
        
        self.ledger_file = "ledger_bybit_real.csv"
        self.active_positions = {}
        self.initial_balance = 10000.0  # Start with $10,000
        self._init_ledger()
        logger.info("BybitTrader initialized successfully")
    
    def _init_ledger(self):
        """Initialize ledger file with proper structure like backtesting"""
        if not os.path.exists(self.ledger_file):
            ledger_columns = [
                'datetime', 'predicted_direction', 'action', 'buy_price', 
                'sell_price', 'balance', 'pnl', 'pnl_sum'
            ]
            df = pd.DataFrame(columns=ledger_columns)
            df.to_csv(self.ledger_file, index=False)
            logger.info(f"Created new ledger file: {self.ledger_file}")
    
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
                        balance = float(coin['walletBalance'])
                        logger.info(f"Current USDT balance: {balance}")
                        return balance
            
            # Try alternative balance endpoint
            try:
                account_data = self.client.get_account_info(accountType="UNIFIED")
                if account_data['retCode'] == 0 and account_data['result']['list']:
                    balance = float(account_data['result']['list'][0]['totalWalletBalance'])
                    logger.info(f"Current balance (alternative): {balance} USDT")
                    return balance
            except Exception as e:
                logger.warning(f"Alternative balance check failed: {e}")
            
            logger.warning("Could not fetch balance from Bybit, using default")
            return self.initial_balance  # Use initial balance as fallback
            
        except Exception as e:
            logger.error(f"Error getting balance from Bybit: {e}")
            return self.initial_balance
    
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
                logger.info(f"Current price for {symbol}: {price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
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
                        logger.info(f"Position details: {position_info}")
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
        """Update ledger with trade data"""
        try:
            df = pd.read_csv(self.ledger_file)
            new_row = pd.DataFrame([trade_data])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.ledger_file, index=False)
            logger.info(f"Ledger updated: {trade_data}")
        except Exception as e:
            logger.error(f"Error updating ledger: {e}")
    
    def take_trade(self, symbol, signal, tp_percent, sl_percent, position_size_usdt=1000):
        """
        Take a trade based on signal with TP/SL management
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            signal: 'long' or 'short'
            tp_percent: Take profit percentage (e.g., 0.05 for 5%)
            sl_percent: Stop loss percentage (e.g., 0.03 for 3%)
            position_size_usdt: Position size in USDT
        """
        try:
            logger.info(f"Taking trade: {symbol} {signal}, TP: {tp_percent*100}%, SL: {sl_percent*100}%")
            
            # Get current balance and price
            balance = self._get_account_balance()
            current_price = self._get_current_price(symbol)
            
            if not current_price:
                logger.error("Failed to get current price")
                return False
            
            # Get trading fees
            taker_fee, maker_fee = self._get_trading_fees(symbol)
            
            # Check if we have sufficient balance
            if balance < position_size_usdt:
                logger.warning(f"Insufficient balance: {balance} USDT < {position_size_usdt} USDT required")
                logger.info("Using available balance for position size")
                position_size_usdt = balance * 0.95  # Use 95% of available balance
            
            # Check for existing position and close it if different direction
            existing_position = self._get_position(symbol)
            if existing_position:
                logger.info(f"Closing existing position: {existing_position}")
                if not self._close_position(symbol, existing_position['side'], existing_position['size']):
                    logger.error("Failed to close existing position")
                    return False
            
            # Calculate position size
            qty = position_size_usdt / current_price
            qty = round(qty, 3)  # Round to 3 decimal places
            
            # Ensure minimum order size
            if qty < 0.001:
                qty = 0.001
                logger.warning(f"Quantity adjusted to minimum: {qty}")
            
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
            logger.info(f"TP price: {tp_price} ({tp_percent*100}%)")
            logger.info(f"SL price: {sl_price} ({sl_percent*100}%)")
            
            # Place the order
            order_id, success = self._place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                tp_price=tp_price,
                sl_price=sl_price
            )
            
            if success:
                # Calculate fee cost
                fee_cost = position_size_usdt * taker_fee
                new_balance = balance - fee_cost
                
                # Record trade in ledger - BUY action for opening position
                trade_data = {
                    'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_direction': signal,
                    'action': 'buy',  # Always 'buy' when opening position
                    'buy_price': current_price,  # Entry price
                    'sell_price': 0.0,  # Will be filled when position closes
                    'balance': new_balance,
                    'pnl': -taker_fee * 100,  # Entry fee as negative PnL
                    'pnl_sum': -taker_fee * 100
                }
                
                self._update_ledger(trade_data)
                
                # Store active position info
                self.active_positions[symbol] = {
                    'entry_price': current_price,
                    'qty': qty,
                    'side': side,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'signal': signal,
                    'order_id': order_id,
                    'position_size_usdt': position_size_usdt,
                    'fee_cost': fee_cost
                }
                
                logger.info(f"Trade opened successfully: {symbol} {signal} at {current_price}")
                logger.info(f"Position size: {qty} {symbol} = ${position_size_usdt:.2f} USDT")
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
        """Check and monitor active positions for TP/SL"""
        try:
            for symbol, position in list(self.active_positions.items()):
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue
                
                entry_price = position['entry_price']
                tp_price = position['tp_price']
                sl_price = position['sl_price']
                signal = position['signal']
                
                # Check if TP/SL hit
                if signal == 'long':
                    if current_price >= tp_price:
                        self._close_position_by_tp_sl(symbol, 'take_profit', current_price)
                    elif current_price <= sl_price:
                        self._close_position_by_tp_sl(symbol, 'stop_loss', current_price)
                else:  # short
                    if current_price <= tp_price:
                        self._close_position_by_tp_sl(symbol, 'take_profit', current_price)
                    elif current_price >= sl_price:
                        self._close_position_by_tp_sl(symbol, 'stop_loss', current_price)
                        
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    def _close_position_by_tp_sl(self, symbol, reason, exit_price):
        """Close position due to TP/SL hit"""
        try:
            position = self.active_positions.get(symbol)
            if not position:
                return
            
            # Close the position
            if self._close_position(symbol, position['side'], position['qty']):
                # Calculate PnL
                entry_price = position['entry_price']
                position_size_usdt = position['position_size_usdt']
                fee_cost = position['fee_cost']
                
                if position['signal'] == 'long':
                    gross_pnl_percent = (exit_price - entry_price) / entry_price
                else:
                    gross_pnl_percent = (entry_price - exit_price) / entry_price
                
                # Calculate net PnL (including fees)
                gross_pnl_usdt = position_size_usdt * gross_pnl_percent
                net_pnl_usdt = gross_pnl_usdt - fee_cost  # Subtract entry fee
                net_pnl_percent = net_pnl_usdt / position_size_usdt
                
                # Get new balance
                new_balance = self._get_account_balance()
                
                # Update ledger - SELL action for closing position
                trade_data = {
                    'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_direction': position['signal'],
                    'action': f'sell - {reason}',  # SELL when closing position
                    'buy_price': entry_price,  # Original entry price
                    'sell_price': exit_price,  # Exit price
                    'balance': new_balance,
                    'pnl': net_pnl_percent * 100,
                    'pnl_sum': net_pnl_percent * 100  # This should be cumulative
                }
                
                self._update_ledger(trade_data)
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                logger.info(f"Position closed by {reason}: {symbol} at {exit_price}")
                logger.info(f"Entry price: {entry_price}, Exit price: {exit_price}")
                logger.info(f"Gross PnL: {gross_pnl_percent*100:.2f}% (${gross_pnl_usdt:.2f})")
                logger.info(f"Net PnL: {net_pnl_percent*100:.2f}% (${net_pnl_usdt:.2f})")
                logger.info(f"Fee cost: ${fee_cost:.2f}")
                logger.info(f"New balance: ${new_balance:.2f}")
                
        except Exception as e:
            logger.error(f"Error closing position by TP/SL: {e}")
    
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
                        # Calculate PnL
                        entry_price = position['entry_price']
                        position_size_usdt = position['position_value']
                        
                        if position['side'] == 'Buy':  # Long position
                            gross_pnl_percent = (current_price - entry_price) / entry_price
                        else:  # Short position
                            gross_pnl_percent = (entry_price - current_price) / entry_price
                        
                        # Get trading fees for exit
                        taker_fee, _ = self._get_trading_fees(symbol)
                        exit_fee = position_size_usdt * taker_fee
                        net_pnl_usdt = (position_size_usdt * gross_pnl_percent) - exit_fee
                        net_pnl_percent = net_pnl_usdt / position_size_usdt
                        
                        # Get new balance
                        new_balance = self._get_account_balance()
                        
                        # Update ledger - SELL action for closing position
                        trade_data = {
                            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'predicted_direction': 'long' if position['side'] == 'Buy' else 'short',
                            'action': 'sell - signal_change',  # SELL when closing position
                            'buy_price': entry_price,  # Original entry price
                            'sell_price': current_price,  # Exit price
                            'balance': new_balance,
                            'pnl': net_pnl_percent * 100,
                            'pnl_sum': net_pnl_percent * 100
                        }
                        
                        self._update_ledger(trade_data)
                        
                        # Remove from active positions
                        if symbol in self.active_positions:
                            del self.active_positions[symbol]
                        
                        logger.info(f"Position force closed: {symbol} at {current_price}")
                        logger.info(f"Net PnL: {net_pnl_percent*100:.2f}% (${net_pnl_usdt:.2f})")
                        logger.info(f"Exit fee: ${exit_fee:.2f}")
                        logger.info(f"New balance: ${new_balance:.2f}")
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


def get_signal_from_strategy():
    """Get signal from strategy generator"""
    try:
        strategy_name = "strategy_02"  # You can change this strategy name
        generator = StrategySignalGenerator()
        
        signal = generator.generate_latest_signal(strategy_name)
        
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


def should_generate_signal():
    """Check if it's time to generate signal (every hour at :01)"""
    current_time = datetime.datetime.now()
    return current_time.minute == 1


def get_current_position_signal(symbol):
    """Get current position signal from active positions"""
    try:
        # Check if we have an active position for this symbol
        if symbol in trader.active_positions:
            return trader.active_positions[symbol]['signal']
        return None
    except Exception as e:
        logger.error(f"Error getting current position signal: {e}")
        return None


def main():
    """Main function with position management logic"""
    try:
        # Initialize trader
        global trader
        trader = BybitTrader()
        
        # ============================================
        # CONFIGURE YOUR TRADING PARAMETERS HERE
        # ============================================
        symbol = "BTCUSDT"          # Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
        tp_percent = 0.02          # Take profit: 0.02 = 2%
        sl_percent = 0.01          # Stop loss: 0.01 = 1%
        position_size_usdt = 1000  # Position size in USDT
        # ============================================
        
        logger.info("=== Starting Bybit Trading Bot with Position Management ===")
        logger.info(f"Trading parameters: {symbol}, TP: {tp_percent*100}%, SL: {sl_percent*100}%")
        logger.info("Signal generation: Every hour at XX:01 (3:01, 4:01, 5:01, etc.)")
        logger.info(f"Starting balance: ${trader.initial_balance:.2f} USDT")
        
        # Get initial account balance
        initial_balance = trader._get_account_balance()
        logger.info(f"Current account balance: ${initial_balance:.2f} USDT")
        
        last_signal = None
        last_signal_time = None
        first_run = True  # Flag to track first run
        
        while True:
            try:
                current_time = datetime.datetime.now()
                
                # Check if we need to generate a new signal (every hour at :01 OR first run)
                if should_generate_signal() or first_run:
                    if first_run:
                        logger.info(f"First run detected - generating signal immediately at {current_time.strftime('%H:%M')}")
                        first_run = False
                    else:
                        logger.info(f"Signal generation time: {current_time.strftime('%H:%M')}")
                    
                    # Get current position signal
                    current_position_signal = get_current_position_signal(symbol)
                    
                    # Generate new signal
                    new_signal = get_signal_from_strategy()
                    
                    if new_signal is not None:
                        logger.info(f"New signal generated: {new_signal}")
                        
                        # Position management logic
                        if current_position_signal is None:
                            # No current position
                            if new_signal == "hold":
                                logger.info("No current position and signal is HOLD - no action needed")
                            else:
                                # Open new position
                                logger.info(f"No current position - opening new {new_signal} position")
                                success = trader.take_trade(symbol, new_signal, tp_percent, sl_percent, position_size_usdt)
                                if success:
                                    last_signal = new_signal
                                    last_signal_time = current_time
                                else:
                                    logger.error("Failed to open new position")
                                
                        elif new_signal == "hold":
                            # Signal is HOLD - keep current position
                            logger.info(f"HOLD signal - keeping current {current_position_signal} position")
                            logger.info(f"   Current position: {current_position_signal}")
                            logger.info(f"   New signal: {new_signal}")
                            logger.info(f"   Action: HOLD position (no action needed)")
                            last_signal = new_signal
                            last_signal_time = current_time
                                
                        elif current_position_signal != new_signal:
                            # Signal change - close current position and open new one
                            logger.info(f"Signal change detected!")
                            logger.info(f"   Current position: {current_position_signal}")
                            logger.info(f"   New signal: {new_signal}")
                            logger.info(f"   Action: Close current position and open new {new_signal} position")
                            
                            # Force close current position
                            if trader.force_close_position(symbol):
                                # Open new position
                                success = trader.take_trade(symbol, new_signal, tp_percent, sl_percent, position_size_usdt)
                                if success:
                                    last_signal = new_signal
                                    last_signal_time = current_time
                                    logger.info(f"Successfully switched to {new_signal} position")
                                else:
                                    logger.error("Failed to open new position after signal change")
                            else:
                                logger.error("Failed to close current position")
                                
                        else:
                            # Same signal - hold position (don't open duplicate)
                            logger.info(f"Same signal - holding {current_position_signal} position")
                            logger.info(f"   Current position: {current_position_signal}")
                            logger.info(f"   New signal: {new_signal}")
                            logger.info(f"   Action: HOLD position (no duplicate trade)")
                            last_signal = new_signal
                            last_signal_time = current_time
                    else:
                        logger.info("No new signal generated (signal generator returned None)")
                        
                else:
                    # Not signal generation time - just monitor positions
                    if current_time.minute % 5 == 0:  # Log every 5 minutes
                        logger.info(f"Monitoring positions... {current_time.strftime('%H:%M')}")
                
                # Always check positions for TP/SL
                trader.check_positions()
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
                
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main() 