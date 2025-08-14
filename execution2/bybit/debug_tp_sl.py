#!/usr/bin/env python3

import sys
import os
import time
from execution2.bybit.unified_signal_generator import UnifiedSignalGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_tp_sl_detection():
    """Debug TP/SL detection for the specific XRP trade"""
    generator = UnifiedSignalGenerator()
    user_id = 1
    user_config = generator.get_user_config(user_id)
    if not user_config:
        logger.error("Failed to get user config")
        return
    logger.info(f"Debugging TP/SL detection for user: {user_config['name']}")

    client = generator.get_bybit_client(user_id)
    if not client:
        logger.error("Failed to get Bybit client")
        return

    symbol = "XRPUSDT"
    test_order_id = "839f9317-f7a9-4172-a4b8-f745bd77e25b"
    
    # Simulate the exact position data that was stored
    position_key = f"{user_id}_strategy_08"
    generator.active_positions[position_key] = {
        'order_id': test_order_id,
        'entry_price': 3.11,  # Entry price when opening long
        'side': 'Buy',
        'tp_price': 3.2346,  # 5% above entry
        'sl_price': 3.048    # 2% below entry
    }
    
    logger.info(f"=== DEBUGGING TP/SL DETECTION ===")
    logger.info(f"Order ID: {test_order_id}")
    logger.info(f"Entry Price: 3.11")
    logger.info(f"TP Price: 3.2346")
    logger.info(f"SL Price: 3.048")
    logger.info(f"Position Side: Buy (Long)")
    
    # Test the current TP/SL detection method
    should_close, close_reason = generator.check_tp_sl_hit(client, symbol, "Buy", position_key)
    logger.info(f"Current method result: should_close={should_close}, reason={close_reason}")
    
    # Now let's manually check the order history to see what's happening
    logger.info(f"\n=== MANUAL ORDER HISTORY CHECK ===")
    try:
        order_resp = client.get_order_history(
            category="linear",
            symbol=symbol,
            limit=50
        )
        
        if order_resp.get("retCode") == 0 and order_resp["result"]["list"]:
            orders = order_resp["result"]["list"]
            logger.info(f"Found {len(orders)} orders in history")
            
            # Find the opening order
            opening_order = None
            opening_order_time = None
            for order in orders:
                if order.get('orderId') == test_order_id:
                    opening_order = order
                    opening_order_time = int(order.get('createdTime', 0))
                    logger.info(f"Found opening order: {order.get('orderId')}")
                    logger.info(f"Opening order time: {opening_order_time}")
                    logger.info(f"Opening order side: {order.get('side')}")
                    logger.info(f"Opening order status: {order.get('orderStatus')}")
                    logger.info(f"Opening order avgPrice: {order.get('avgPrice')}")
                    break
            
            if opening_order_time:
                # Look for closing orders (Sell orders) created after the opening order
                closing_orders = []
                for order in orders:
                    order_side = order.get('side', '')
                    order_time = int(order.get('createdTime', 0))
                    order_status = order.get('orderStatus', '')
                    
                    if (order_side == 'Sell' and 
                        order_time > opening_order_time and 
                        order_status == 'Filled'):
                        closing_orders.append(order)
                        logger.info(f"Found closing order: {order.get('orderId')}")
                        logger.info(f"Closing order time: {order_time}")
                        logger.info(f"Closing order side: {order_side}")
                        logger.info(f"Closing order status: {order_status}")
                        logger.info(f"Closing order avgPrice: {order.get('avgPrice')}")
                
                if closing_orders:
                    # Sort by creation time and take the one closest to the opening order
                    closing_orders.sort(key=lambda x: int(x.get('createdTime', 0)))
                    closing_order = closing_orders[0]  # Take the earliest closing order
                    
                    # Additional check: Verify this is the correct closing order by checking the time difference
                    time_diff = int(closing_order.get('createdTime', 0)) - opening_order_time
                    if time_diff > 60000:  # More than 1 minute difference
                        logger.warning(f"Closing order {closing_order.get('orderId')} was created {time_diff}ms after opening order - this might not be the correct closing order")
                        # Look for a closer closing order
                        for order in closing_orders:
                            order_time = int(order.get('createdTime', 0))
                            time_diff = order_time - opening_order_time
                            if time_diff <= 60000:  # Within 1 minute
                                closing_order = order
                                logger.info(f"Found closer closing order: {closing_order.get('orderId')} (time diff: {time_diff}ms)")
                                break
                    
                    exit_price = float(closing_order.get('avgPrice', 0))
                    
                    logger.info(f"\n=== ANALYSIS ===")
                    logger.info(f"Exit Price: {exit_price}")
                    logger.info(f"SL Price: 3.048")
                    logger.info(f"TP Price: 3.2346")
                    
                    if exit_price <= 3.048:
                        logger.info(f"✅ SL HIT: {exit_price} <= 3.048")
                        expected_reason = "sl_hit"
                    elif exit_price >= 3.2346:
                        logger.info(f"✅ TP HIT: {exit_price} >= 3.2346")
                        expected_reason = "tp_hit"
                    else:
                        logger.info(f"❌ Manual Close: {exit_price} is between SL (3.048) and TP (3.2346)")
                        expected_reason = "manually_closed"
                    
                    logger.info(f"Expected reason: {expected_reason}")
                    logger.info(f"Actual reason: {close_reason}")
                    
                    if expected_reason != close_reason:
                        logger.error(f"❌ MISMATCH: Expected {expected_reason}, got {close_reason}")
                    else:
                        logger.info(f"✅ MATCH: Expected {expected_reason}, got {close_reason}")
                else:
                    logger.warning("No closing orders found")
            else:
                logger.warning("Opening order not found")
        else:
            logger.error(f"Failed to get order history: {order_resp}")
            
    except Exception as e:
        logger.error(f"Error in manual check: {e}")

if __name__ == "__main__":
    debug_tp_sl_detection()
