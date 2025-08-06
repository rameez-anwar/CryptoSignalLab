import os
import pandas as pd
import datetime
import configparser
from sqlalchemy import text
from dotenv import load_dotenv
from data.utils.db_utils import get_pg_engine
from data.downloaders.DataDownloader import DataDownloader
from indicators.technical_indicator import IndicatorCalculator
from signals.technical_indicator_signal.signal_generator import SignalGenerator
from strategies.strategy_pipeline.utils.indicator_utils import get_all_indicator_configs

# Load environment variables
load_dotenv()

class StrategySignalGenerator:
    def __init__(self):
        self.engine = get_pg_engine()
        self.indicator_catalog = get_all_indicator_configs()
        
    def get_strategy_config(self, strategy_name):
        """Get strategy configuration from database"""
        try:
            query = text("SELECT * FROM public.config_strategies WHERE name = :strategy_name")
            strategy_df = pd.read_sql_query(query, self.engine, params={'strategy_name': strategy_name})
            
            if strategy_df.empty:
                raise ValueError(f"Strategy '{strategy_name}' not found in database")
            
            return strategy_df.iloc[0].to_dict()
        except Exception as e:
            print(f"Error getting strategy config: {e}")
            return None
    
    def get_highest_window_size(self, strategy_config):
        """Get the highest window size from enabled indicators"""
        max_window = 0
        enabled_indicators = []
        
        for col in strategy_config.keys():
            if col.endswith('_window_size') and strategy_config[col] > 0:
                window_size = strategy_config[col]
                indicator_name = col.replace('_window_size', '')
                if strategy_config.get(indicator_name, False):  # Check if indicator is enabled
                    max_window = max(max_window, window_size)
                    enabled_indicators.append((indicator_name, window_size))
        
        return max_window, enabled_indicators
    
    def get_latest_data(self, exchange, symbol, time_horizon, required_window_size):
        """Get latest data with enough history for window size"""
        try:
            # Calculate how much data we need (window_size + some buffer)
            required_minutes = required_window_size * self._parse_time_horizon_minutes(time_horizon) + 100
            
            # Get current time
            now = datetime.datetime.now()
            start_time = now - datetime.timedelta(minutes=required_minutes)
            
            # Download data
            downloader = DataDownloader(exchange=exchange, symbol=symbol, time_horizon=time_horizon)
            df_1min, df_horizon = downloader.fetch_data(start_time=start_time, end_time=now)
            
            if df_horizon.empty:
                raise ValueError(f"No data available for {exchange} {symbol} {time_horizon}")
            
            return df_horizon
            
        except Exception as e:
            print(f"Error getting latest data: {e}")
            return None
    
    def _parse_time_horizon_minutes(self, time_horizon):
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
                return 60  # Default to 1 hour
    
    def calculate_indicators(self, df, strategy_config):
        """Calculate indicators based on strategy configuration"""
        try:
            # Ensure dataframe has datetime column
            if 'datetime' not in df.columns:
                df = df.reset_index()
                if 'index' in df.columns and 'datetime' not in df.columns:
                    df = df.rename(columns={'index': 'datetime'})
            
            # Initialize calculator with dataframe
            calculator = IndicatorCalculator(df)
            calculated_indicators = []
            
            print(f"   Available indicators in strategy config:")
            enabled_indicators = []
            for col in strategy_config.keys():
                if col.endswith('_window_size') and strategy_config[col] > 0:
                    indicator_name = col.replace('_window_size', '')
                    if strategy_config.get(indicator_name, False):  # Check if indicator is enabled
                        enabled_indicators.append(indicator_name)
            
            print(f"   Enabled indicators: {enabled_indicators}")
            
            for col in strategy_config.keys():
                if col.endswith('_window_size') and strategy_config[col] > 0:
                    indicator_name = col.replace('_window_size', '')
                    if strategy_config.get(indicator_name, False):  # Check if indicator is enabled
                        window_size = strategy_config[col]
                        
                        # Get the method name for the indicator
                        method_name = f"add_{indicator_name}"
                        if hasattr(calculator, method_name):
                            try:
                                if window_size != 0:
                                    getattr(calculator, method_name)(window_size)
                                else:
                                    getattr(calculator, method_name)()
                                calculated_indicators.append(indicator_name)
                                print(f"   ✓ Calculated {indicator_name} with window {window_size}")
                            except Exception as e:
                                print(f"   ✗ Could not calculate {indicator_name}: {e}")
                                continue
                        else:
                            print(f"   ✗ Method {method_name} not found for {indicator_name}")
            
            # Ensure the final dataframe has the required columns
            df_with_indicators = calculator.df
            if 'datetime' not in df_with_indicators.columns:
                df_with_indicators = df_with_indicators.reset_index()
                if 'index' in df_with_indicators.columns and 'datetime' not in df_with_indicators.columns:
                    df_with_indicators = df_with_indicators.rename(columns={'index': 'datetime'})
            
            print(f"   Final dataframe columns: {list(df_with_indicators.columns)}")
            return df_with_indicators, calculated_indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df, []
    
    def generate_signal(self, df_with_indicators, calculated_indicators):
        """Generate trading signal based on indicator data with improved logic"""
        try:
            # Create signal generator with the dataframe and indicator names
            signal_generator = SignalGenerator(df_with_indicators, calculated_indicators)
            signals_df = signal_generator.generate_signals()
            
            if signals_df.empty:
                print("   No signals generated - empty dataframe")
                return 0  # No signal
            
            # Get the latest signal by combining all indicator signals
            signal_columns = [col for col in signals_df.columns if col.startswith('signal_')]
            if not signal_columns:
                print("   No signal columns found")
                return 0
            
            # Get the latest row
            latest_row = signals_df.iloc[-1]
            
            # Combine signals with improved logic
            latest_signals = [latest_row[col] for col in signal_columns]
            
            # Count signals with weights
            long_signals = sum(1 for s in latest_signals if s == 1)
            short_signals = sum(1 for s in latest_signals if s == -1)
            neutral_signals = sum(1 for s in latest_signals if s == 0)
            
            print(f"   Signal breakdown: LONG={long_signals}, SHORT={short_signals}, NEUTRAL={neutral_signals}")
            
            # Improved decision logic
            total_signals = len(latest_signals)
            if total_signals == 0:
                print("   No valid signals found")
                return 0
            
            # Calculate signal strength
            long_strength = long_signals / total_signals
            short_strength = short_signals / total_signals
            
            print(f"   Signal strength: LONG={long_strength:.2f}, SHORT={short_strength:.2f}")
            
            # Decision logic with minimum threshold
            min_threshold = 0.3  # At least 30% of signals must agree
            
            if long_strength > min_threshold and long_strength > short_strength:
                print(f"   Decision: LONG (strength: {long_strength:.2f})")
                return 1  # Long
            elif short_strength > min_threshold and short_strength > long_strength:
                print(f"   Decision: SHORT (strength: {short_strength:.2f})")
                return -1  # Short
            else:
                print(f"   Decision: NEUTRAL (no clear signal)")
                return 0  # Neutral
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return 0
    
    def should_generate_signal(self, time_horizon, last_signal_time=None):
        """Check if enough time has passed to generate a new signal"""
        if last_signal_time is None:
            return True
        
        horizon_minutes = self._parse_time_horizon_minutes(time_horizon)
        time_since_last = datetime.datetime.now() - last_signal_time
        minutes_since_last = time_since_last.total_seconds() / 60
        
        return minutes_since_last >= horizon_minutes
    
    def get_latest_signal_from_db(self, strategy_name):
        """Get the latest signal from database for a strategy"""
        try:
            query = text(f"SELECT datetime, signal FROM signals.{strategy_name} ORDER BY datetime DESC LIMIT 1")
            result = pd.read_sql_query(query, self.engine)
            
            if result.empty:
                return None, None
            
            latest_time = pd.to_datetime(result.iloc[0]['datetime'])
            latest_signal = result.iloc[0]['signal']
            
            return latest_time, latest_signal
            
        except Exception as e:
            print(f"Error getting latest signal from DB: {e}")
            return None, None
    
    def generate_latest_signal(self, strategy_name):
        """Main method to generate the latest signal for a strategy"""
        print(f"🔍 Generating signal for strategy: {strategy_name}")
        print("=" * 60)
        
        # Get strategy configuration
        strategy_config = self.get_strategy_config(strategy_name)
        if strategy_config is None:
            print("❌ Failed to get strategy configuration")
            return None
        
        print(f"📊 Strategy Configuration:")
        print(f"   Exchange: {strategy_config['exchange']}")
        print(f"   Symbol: {strategy_config['symbol']}")
        print(f"   Time Horizon: {strategy_config['time_horizon']}")
        print(f"   Take Profit: {strategy_config['take_profit']:.2%}")
        print(f"   Stop Loss: {strategy_config['stop_loss']:.2%}")
        
        # Get highest window size
        max_window, enabled_indicators = self.get_highest_window_size(strategy_config)
        print(f"   Max Window Size: {max_window}")
        print(f"   Enabled Indicators: {len(enabled_indicators)}")
        
        # Check if we should generate a new signal
        last_signal_time, last_signal = self.get_latest_signal_from_db(strategy_name)
        
        if last_signal_time is not None:
            print(f"   Last Signal Time: {last_signal_time}")
            print(f"   Last Signal: {'LONG' if last_signal == 1 else 'SHORT' if last_signal == -1 else 'NEUTRAL'}")
            
            if not self.should_generate_signal(strategy_config['time_horizon'], last_signal_time):
                horizon_minutes = self._parse_time_horizon_minutes(strategy_config['time_horizon'])
                time_since_last = datetime.datetime.now() - last_signal_time
                minutes_since_last = time_since_last.total_seconds() / 60
                remaining_minutes = horizon_minutes - minutes_since_last
                
                print(f"\n⏰ Signal not ready yet!")
                print(f"   Time since last signal: {minutes_since_last:.1f} minutes")
                print(f"   Time horizon: {horizon_minutes} minutes")
                print(f"   Remaining time: {remaining_minutes:.1f} minutes")
                print(f"   Next signal in: {remaining_minutes/60:.1f} hours")
                return last_signal  # Return the last signal instead of None
        
        print(f"\n📈 Getting latest market data...")
        
        # Get latest data
        df = self.get_latest_data(
            strategy_config['exchange'],
            strategy_config['symbol'],
            strategy_config['time_horizon'],
            max_window
        )
        
        if df is None:
            print("❌ Failed to get market data")
            return None
        
        print(f"   Data points: {len(df)}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        # Calculate indicators
        print(f"\n🔧 Calculating indicators...")
        df_with_indicators, calculated_indicators = self.calculate_indicators(df, strategy_config)
        print(f"   Calculated {len(calculated_indicators)} indicators: {calculated_indicators}")
        
        # Generate signal
        print(f"\n🎯 Generating signal...")
        signal = self.generate_signal(df_with_indicators, calculated_indicators)
        
        # Display result
        print(f"\n" + "=" * 60)
        print(f"📊 SIGNAL RESULT")
        print(f"=" * 60)
        
        if signal == 1:
            print(f"🟢 LONG SIGNAL")
            print(f"   Action: BUY {strategy_config['symbol'].upper()}")
            print(f"   Take Profit: +{strategy_config['take_profit']:.2%}")
            print(f"   Stop Loss: -{strategy_config['stop_loss']:.2%}")
        elif signal == -1:
            print(f"🔴 SHORT SIGNAL")
            print(f"   Action: SELL {strategy_config['symbol'].upper()}")
            print(f"   Take Profit: +{strategy_config['take_profit']:.2%}")
            print(f"   Stop Loss: -{strategy_config['stop_loss']:.2%}")
        else:
            # Determine what position to hold based on last signal
            if last_signal == 1:
                print(f"🟡 HOLD SIGNAL (LONG)")
                print(f"   Action: HOLD LONG {strategy_config['symbol'].upper()}")
                print(f"   Take Profit: +{strategy_config['take_profit']:.2%}")
                print(f"   Stop Loss: -{strategy_config['stop_loss']:.2%}")
            elif last_signal == -1:
                print(f"🟡 HOLD SIGNAL (SHORT)")
                print(f"   Action: HOLD SHORT {strategy_config['symbol'].upper()}")
                print(f"   Take Profit: +{strategy_config['take_profit']:.2%}")
                print(f"   Stop Loss: -{strategy_config['stop_loss']:.2%}")
            else:
                print(f"🟡 NEUTRAL SIGNAL")
                print(f"   Action: NO POSITION")
        
        print(f"\n⏰ Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Next signal in: {strategy_config['time_horizon']}")
        print("=" * 60)
        
        return signal

def main(strategy_name="strategy_07"):
    """Main function to run signal generator"""
    # You can change the strategy name here or pass it as parameter
    # strategy_name = "strategy_02"  # Change this to test different strategies
    
    generator = StrategySignalGenerator()
    signal = generator.generate_latest_signal(strategy_name)
    
    if signal is not None:
        print(f"\n✅ Signal generation completed for {strategy_name}")
        return signal
    else:
        print(f"\n❌ Signal generation failed for {strategy_name}")
        return None

if __name__ == "__main__":
    # You can change the strategy name here
    strategy_name = "strategy_07"  # Change this to test different strategies
    main(strategy_name)
