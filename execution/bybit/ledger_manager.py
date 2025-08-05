#!/usr/bin/env python3
"""
Ledger Manager for Bybit Trading Bot
This script helps manage multiple strategy ledgers stored in the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from sqlalchemy import text
from data.utils.db_utils import get_pg_engine
from execution.bybit.main import list_strategy_ledgers, get_strategy_ledger_data, export_strategy_ledger

def show_ledger_summary(strategy_name):
    """Show summary of a strategy's ledger"""
    try:
        df = get_strategy_ledger_data(strategy_name)
        
        if df.empty:
            print(f"No ledger data found for {strategy_name}")
            return
        
        print(f"\n📊 LEDGER SUMMARY: {strategy_name}")
        print("=" * 60)
        
        # Basic stats
        total_trades = len(df[df['action'].isin(['buy', 'sell - take_profit', 'sell - stop_loss', 'sell - direction_change'])])
        total_entries = len(df[df['action'] == 'buy'])
        total_exits = len(df[df['action'].str.contains('sell', na=False)])
        
        print(f"Total Entries: {total_entries}")
        print(f"Total Exits: {total_exits}")
        print(f"Total Trades: {total_trades}")
        
        # Balance info
        if not df.empty:
            initial_balance = df.iloc[0]['balance'] + abs(df.iloc[0]['pnl']) if df.iloc[0]['pnl'] < 0 else df.iloc[0]['balance']
            final_balance = df.iloc[-1]['balance']
            total_pnl = df.iloc[-1]['pnl_sum']
            
            print(f"Initial Balance: ${initial_balance:.2f}")
            print(f"Final Balance: ${final_balance:.2f}")
            print(f"Total PnL: {total_pnl:.2f}%")
            print(f"Net Profit/Loss: ${final_balance - initial_balance:.2f}")
        
        # Recent trades
        print(f"\n📈 RECENT TRADES:")
        print("-" * 60)
        recent_trades = df[df['action'].isin(['buy', 'sell - take_profit', 'sell - stop_loss', 'sell - direction_change'])].tail(10)
        
        for _, trade in recent_trades.iterrows():
            action_emoji = "🟢" if trade['action'] == 'buy' else "🔴"
            print(f"{action_emoji} {trade['datetime']} | {trade['action']} | "
                  f"Price: {trade['buy_price']:.2f} → {trade['sell_price']:.2f} | "
                  f"PnL: {trade['pnl']:.2f}% | Balance: ${trade['balance']:.2f}")
        
    except Exception as e:
        print(f"Error showing ledger summary: {e}")

def show_all_ledgers():
    """Show summary of all strategy ledgers"""
    try:
        ledgers = list_strategy_ledgers()
        
        if not ledgers:
            print("No strategy ledgers found")
            return
        
        print("\n📋 ALL STRATEGY LEDGERS SUMMARY")
        print("=" * 80)
        
        summary_data = []
        
        for ledger in ledgers:
            strategy_name = ledger.replace('ledger_', '')
            df = get_strategy_ledger_data(strategy_name)
            
            if not df.empty:
                total_trades = len(df[df['action'].isin(['buy', 'sell - take_profit', 'sell - stop_loss', 'sell - direction_change'])])
                final_balance = df.iloc[-1]['balance']
                total_pnl = df.iloc[-1]['pnl_sum']
                
                summary_data.append({
                    'strategy': strategy_name,
                    'trades': total_trades,
                    'balance': final_balance,
                    'pnl': total_pnl
                })
        
        # Sort by PnL
        summary_data.sort(key=lambda x: x['pnl'], reverse=True)
        
        print(f"{'Strategy':<20} {'Trades':<8} {'Balance':<12} {'PnL':<8}")
        print("-" * 80)
        
        for data in summary_data:
            pnl_color = "🟢" if data['pnl'] > 0 else "🔴" if data['pnl'] < 0 else "🟡"
            print(f"{data['strategy']:<20} {data['trades']:<8} ${data['balance']:<11.2f} {pnl_color} {data['pnl']:<6.2f}%")
        
    except Exception as e:
        print(f"Error showing all ledgers: {e}")

def export_all_ledgers():
    """Export all strategy ledgers to CSV files"""
    try:
        ledgers = list_strategy_ledgers()
        
        if not ledgers:
            print("No strategy ledgers found to export")
            return
        
        print(f"\n📤 EXPORTING ALL LEDGERS")
        print("=" * 40)
        
        exported_files = []
        
        for ledger in ledgers:
            strategy_name = ledger.replace('ledger_', '')
            filename = export_strategy_ledger(strategy_name)
            if filename:
                exported_files.append(filename)
        
        print(f"\n✅ Exported {len(exported_files)} ledger files:")
        for filename in exported_files:
            print(f"  - {filename}")
        
    except Exception as e:
        print(f"Error exporting ledgers: {e}")

def show_ledger_details(strategy_name):
    """Show detailed ledger data for a strategy"""
    try:
        df = get_strategy_ledger_data(strategy_name)
        
        if df.empty:
            print(f"No ledger data found for {strategy_name}")
            return
        
        print(f"\n📊 DETAILED LEDGER: {strategy_name}")
        print("=" * 80)
        print(f"{'DateTime':<20} {'Action':<20} {'Buy':<10} {'Sell':<10} {'Balance':<10} {'PnL':<8}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            action_emoji = "🟢" if row['action'] == 'buy' else "🔴" if 'sell' in row['action'] else "🟡"
            buy_price = f"${row['buy_price']:.2f}" if row['buy_price'] > 0 else "-"
            sell_price = f"${row['sell_price']:.2f}" if row['sell_price'] > 0 else "-"
            
            print(f"{row['datetime']:<20} {action_emoji} {row['action']:<17} {buy_price:<10} {sell_price:<10} "
                  f"${row['balance']:<9.2f} {row['pnl']:<7.2f}%")
        
    except Exception as e:
        print(f"Error showing ledger details: {e}")

def main():
    """Main function for ledger manager"""
    print("🔍 Bybit Trading Bot - Ledger Manager")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python ledger_manager.py list                    - List all strategy ledgers")
        print("  python ledger_manager.py summary <strategy>     - Show ledger summary")
        print("  python ledger_manager.py details <strategy>     - Show detailed ledger")
        print("  python ledger_manager.py export                  - Export all ledgers to CSV")
        print("  python ledger_manager.py export <strategy>      - Export specific ledger")
        print("\nExamples:")
        print("  python ledger_manager.py list")
        print("  python ledger_manager.py summary strategy_01")
        print("  python ledger_manager.py details strategy_02")
        print("  python ledger_manager.py export")
        print("  python ledger_manager.py export strategy_01")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        show_all_ledgers()
        
    elif command == "summary":
        if len(sys.argv) < 3:
            print("Error: Please specify strategy name")
            print("Usage: python ledger_manager.py summary <strategy_name>")
            return
        strategy_name = sys.argv[2]
        show_ledger_summary(strategy_name)
        
    elif command == "details":
        if len(sys.argv) < 3:
            print("Error: Please specify strategy name")
            print("Usage: python ledger_manager.py details <strategy_name>")
            return
        strategy_name = sys.argv[2]
        show_ledger_details(strategy_name)
        
    elif command == "export":
        if len(sys.argv) >= 3:
            # Export specific strategy
            strategy_name = sys.argv[2]
            filename = export_strategy_ledger(strategy_name)
            if filename:
                print(f"✅ Exported {strategy_name} ledger to {filename}")
        else:
            # Export all ledgers
            export_all_ledgers()
            
    else:
        print(f"Unknown command: {command}")
        print("Use 'list', 'summary', 'details', or 'export'")

if __name__ == "__main__":
    main() 