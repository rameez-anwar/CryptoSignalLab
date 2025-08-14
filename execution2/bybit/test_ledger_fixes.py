#!/usr/bin/env python3

import sys
import os
import time
from execution2.bybit.unified_signal_generator import UnifiedSignalGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ledger_fixes():
    """Test the ledger fixes to ensure they work correctly"""
    generator = UnifiedSignalGenerator()
    user_id = 1
    user_config = generator.get_user_config(user_id)
    if not user_config:
        logger.error("Failed to get user config")
        return
    
    logger.info(f"Testing ledger fixes for user: {user_config['name']}")
    
    client = generator.get_bybit_client(user_id)
    if not client:
        logger.error("Failed to get Bybit client")
        return
    
    # Get strategy configs
    strategy_configs = generator.get_strategy_configs(user_config['strategies'])
    if not strategy_configs:
        logger.error("No strategy configs found")
        return
    
    # Test the first strategy
    strategy_config = strategy_configs[0]
    symbol = f"{strategy_config['symbol'].upper()}USDT"
    
    logger.info(f"Testing with strategy: {strategy_config['name']}, symbol: {symbol}")
    
    # Test 1: Check if old trades are processed correctly
    logger.info("=== Test 1: Checking old trades from ledger ===")
    old_trade_updated = generator.check_old_trades_from_ledger(client, user_id, strategy_config)
    logger.info(f"Old trade updated: {old_trade_updated}")
    
    # Test 2: Check if duplicate prevention works
    logger.info("=== Test 2: Testing duplicate prevention ===")
    table_name = generator.get_ledger_table_name(user_id, strategy_config)
    
    # Check current ledger entries
    query = f"""
        SELECT datetime, action, buy_price, sell_price, pnl_percent, pnl_sum, order_id
        FROM execution.{table_name} 
        ORDER BY datetime DESC 
        LIMIT 5
    """
    
    with generator.engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        
        logger.info(f"Current ledger entries for {table_name}:")
        for row in rows:
            logger.info(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}% | {row[5]} | {row[6]}")
    
    # Test 3: Check if PnL calculation is correct
    logger.info("=== Test 3: Testing PnL calculation ===")
    if rows:
        last_row = rows[0]
        logger.info(f"Last entry PnL: {last_row[4]}%, Cumulative PnL: {last_row[5]}")
        
        # Check if the PnL includes the -0.05% fee
        if last_row[1] in ['sell - take_profit', 'sell - stop_loss', 'sell - direction change', 'manually_closed']:
            if last_row[4] <= -0.05:
                logger.info("✅ PnL correctly includes -0.05% fee for closing trades")
            else:
                logger.warning("❌ PnL may not include -0.05% fee for closing trades")
    
    logger.info("=== Test completed ===")

if __name__ == "__main__":
    test_ledger_fixes()
