#!/usr/bin/env python3

import sys
import os
import time
from execution2.bybit.unified_signal_generator import UnifiedSignalGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_order_history():
    generator = UnifiedSignalGenerator()
    user_id = 1
    user_config = generator.get_user_config(user_id)
    if not user_config:
        logger.error("Failed to get user config")
        return

    client = generator.get_bybit_client(user_id)
    if not client:
        logger.error("Failed to get Bybit client")
        return

    symbol = "XRPUSDT"
    test_order_id = "839f9317-f7a9-4172-a4b8-f745bd77e25b"
    logger.info(f"Debugging order history for order ID: {test_order_id}")

    # Get order history
    order_resp = client.get_order_history(
        category="linear",
        symbol=symbol,
        limit=50
    )
    
    if order_resp.get("retCode") == 0 and order_resp["result"]["list"]:
        orders = order_resp["result"]["list"]
        logger.info(f"Found {len(orders)} orders in history")
        
        # Find the opening order time
        opening_order_time = None
        opening_order = None
        for order in orders:
            if order.get('orderId') == test_order_id:
                opening_order_time = int(order.get('createdTime', 0))
                opening_order = order
                logger.info(f"Opening order found:")
                logger.info(f"  Order ID: {order.get('orderId')}")
                logger.info(f"  Side: {order.get('side')}")
                logger.info(f"  Created Time: {order.get('createdTime')}")
                logger.info(f"  Status: {order.get('orderStatus')}")
                logger.info(f"  Avg Price: {order.get('avgPrice')}")
                logger.info(f"  Take Profit: {order.get('takeProfit')}")
                logger.info(f"  Stop Loss: {order.get('stopLoss')}")
                break
        
        if opening_order_time:
            logger.info(f"\nLooking for closing orders after time: {opening_order_time}")
            
            # Look for all closing orders (opposite side) that were created after the opening order
            closing_orders = []
            for order in orders:
                order_side = order.get('side', '')
                order_time = int(order.get('createdTime', 0))
                order_status = order.get('orderStatus', '')
                
                # For long position (Buy), look for Sell orders
                # For short position (Sell), look for Buy orders
                expected_closing_side = "Sell" if opening_order.get('side') == "Buy" else "Buy"
                
                if (order_side == expected_closing_side and 
                    order_time > opening_order_time and 
                    order_status == 'Filled'):
                    closing_orders.append(order)
                    logger.info(f"Closing order found:")
                    logger.info(f"  Order ID: {order.get('orderId')}")
                    logger.info(f"  Side: {order.get('side')}")
                    logger.info(f"  Created Time: {order.get('createdTime')}")
                    logger.info(f"  Status: {order.get('orderStatus')}")
                    logger.info(f"  Avg Price: {order.get('avgPrice')}")
                    logger.info(f"  Exec Qty: {order.get('cumExecQty')}")
                    logger.info(f"  Exec Fee: {order.get('cumExecFee')}")
            
            if closing_orders:
                logger.info(f"\nFound {len(closing_orders)} closing orders")
                
                # Sort by time to get the most recent
                closing_orders.sort(key=lambda x: int(x.get('createdTime', 0)))
                
                for i, order in enumerate(closing_orders):
                    logger.info(f"\nClosing order {i+1}:")
                    logger.info(f"  Order ID: {order.get('orderId')}")
                    logger.info(f"  Exit Price: {order.get('avgPrice')}")
                    logger.info(f"  Time: {order.get('createdTime')}")
                    
                    # Check if this was TP/SL hit
                    exit_price = float(order.get('avgPrice', 0))
                    tp_price = 3.2346
                    sl_price = 3.048
                    
                    if opening_order.get('side') == "Buy":  # Long position
                        if exit_price >= tp_price:
                            logger.info(f"  ✅ TP HIT: {exit_price} >= {tp_price}")
                        elif exit_price <= sl_price:
                            logger.info(f"  ✅ SL HIT: {exit_price} <= {sl_price}")
                        else:
                            logger.info(f"  ❌ Manual Close: {exit_price} is between SL ({sl_price}) and TP ({tp_price})")
                    else:  # Short position
                        if exit_price <= tp_price:
                            logger.info(f"  ✅ TP HIT: {exit_price} <= {tp_price}")
                        elif exit_price >= sl_price:
                            logger.info(f"  ✅ SL HIT: {exit_price} >= {sl_price}")
                        else:
                            logger.info(f"  ❌ Manual Close: {exit_price} is between TP ({tp_price}) and SL ({sl_price})")
            else:
                logger.info("No closing orders found")
        else:
            logger.error("Opening order not found")
    else:
        logger.error("Failed to get order history")

if __name__ == "__main__":
    debug_order_history()
