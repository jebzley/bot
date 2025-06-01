"""
Trade Analysis Tool - Analyze your trading performance from logs
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import argparse

class TradeAnalyzer:
    def __init__(self, log_dir="trade_logs"):
        self.log_dir = log_dir
        self.trades = []
        
    def load_trades(self, days=None):
        """Load trades from JSON log files"""
        if not os.path.exists(self.log_dir):
            print(f"Log directory {self.log_dir} not found")
            return
        
        # Get all trade log files
        log_files = sorted([f for f in os.listdir(self.log_dir) if f.startswith("trades_")])
        
        if not log_files:
            print("No trade logs found")
            return
        
        # Load trades from files
        cutoff_date = None
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in log_files:
            with open(os.path.join(self.log_dir, log_file), 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            trade = json.loads(line)
                            trade_time = datetime.fromisoformat(trade['timestamp'])
                            
                            if cutoff_date and trade_time < cutoff_date:
                                continue
                                
                            self.trades.append(trade)
                        except:
                            continue
        
        print(f"Loaded {len(self.trades)} trades")
    
    def analyze_performance(self):
        """Analyze overall trading performance"""
        if not self.trades:
            print("No trades to analyze")
            return
        
        # Separate completed trades
        completed_trades = [t for t in self.trades if 'pnl' in t and 'exit_time' in t]
        
        if not completed_trades:
            print("No completed trades found")
            return
        
        # Calculate metrics
        total_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in completed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Print results
        print("\n" + "="*50)
        print("TRADING PERFORMANCE ANALYSIS")
        print("="*50)
        print(f"\nTotal Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1%})")
        print(f"Losing Trades: {len(losing_trades)} ({(1-win_rate):.1%})")
        print(f"\nTotal P&L: ${total_pnl:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Best and worst trades
        if completed_trades:
            best_trade = max(completed_trades, key=lambda x: x['pnl'])
            worst_trade = min(completed_trades, key=lambda x: x['pnl'])
            print(f"\nBest Trade: ${best_trade['pnl']:.2f}")
            print(f"Worst Trade: ${worst_trade['pnl']:.2f}")
    
    def analyze_by_day(self):
        """Analyze performance by day"""
        if not self.trades:
            return
        
        completed_trades = [t for t in self.trades if 'pnl' in t and 'exit_time' in t]
        if not completed_trades:
            return
        
        # Group by day
        daily_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'wins': 0})
        
        for trade in completed_trades:
            exit_date = datetime.fromisoformat(trade['exit_time']).date()
            daily_stats[exit_date]['trades'] += 1
            daily_stats[exit_date]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                daily_stats[exit_date]['wins'] += 1
        
        print("\n" + "="*50)
        print("DAILY PERFORMANCE")
        print("="*50)
        print(f"{'Date':<12} {'Trades':<8} {'Wins':<8} {'P&L':<10} {'Win Rate':<10}")
        print("-"*50)
        
        for date in sorted(daily_stats.keys()):
            stats = daily_stats[date]
            win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            print(f"{date} {stats['trades']:<8} {stats['wins']:<8} "
                  f"${stats['pnl']:<9.2f} {win_rate:<10.1%}")
        
        # Summary
        total_days = len(daily_stats)
        profitable_days = len([d for d in daily_stats.values() if d['pnl'] > 0])
        print(f"\nTotal Days: {total_days}")
        print(f"Profitable Days: {profitable_days} ({profitable_days/total_days*100:.1f}%)")
    
    def analyze_by_exit_reason(self):
        """Analyze trades by exit reason"""
        if not self.trades:
            return
        
        completed_trades = [t for t in self.trades if 'pnl' in t and 'reason' in t]
        if not completed_trades:
            return
        
        # Group by exit reason
        reason_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
        
        for trade in completed_trades:
            reason = trade['reason']
            reason_stats[reason]['count'] += 1
            reason_stats[reason]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                reason_stats[reason]['wins'] += 1
        
        print("\n" + "="*50)
        print("PERFORMANCE BY EXIT REASON")
        print("="*50)
        print(f"{'Exit Reason':<25} {'Count':<8} {'Wins':<8} {'P&L':<10} {'Win Rate':<10}")
        print("-"*60)
        
        for reason, stats in sorted(reason_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            win_rate = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            print(f"{reason:<25} {stats['count']:<8} {stats['wins']:<8} "
                  f"${stats['pnl']:<9.2f} {win_rate:<10.1%}")
    
    def analyze_trade_duration(self):
        """Analyze trade durations"""
        if not self.trades:
            return
        
        completed_trades = [t for t in self.trades if 'entry_time' in t and 'exit_time' in t]
        if not completed_trades:
            return
        
        durations = []
        duration_pnl = defaultdict(list)
        
        for trade in completed_trades:
            entry = datetime.fromisoformat(trade['entry_time'])
            exit = datetime.fromisoformat(trade['exit_time'])
            duration = (exit - entry).total_seconds() / 3600  # Hours
            durations.append(duration)
            
            # Bucket by duration
            if duration < 1:
                bucket = "< 1 hour"
            elif duration < 4:
                bucket = "1-4 hours"
            elif duration < 12:
                bucket = "4-12 hours"
            else:
                bucket = "> 12 hours"
            
            if 'pnl' in trade:
                duration_pnl[bucket].append(trade['pnl'])
        
        print("\n" + "="*50)
        print("TRADE DURATION ANALYSIS")
        print("="*50)
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"Average Duration: {avg_duration:.1f} hours")
            print(f"Shortest Trade: {min(durations):.1f} hours")
            print(f"Longest Trade: {max(durations):.1f} hours")
            
            print(f"\n{'Duration':<15} {'Trades':<10} {'Avg P&L':<10} {'Total P&L':<10}")
            print("-"*45)
            
            for bucket in ["< 1 hour", "1-4 hours", "4-12 hours", "> 12 hours"]:
                if bucket in duration_pnl:
                    trades = duration_pnl[bucket]
                    avg_pnl = sum(trades) / len(trades)
                    total_pnl = sum(trades)
                    print(f"{bucket:<15} {len(trades):<10} ${avg_pnl:<9.2f} ${total_pnl:<9.2f}")
    
    def export_to_csv(self, filename="trade_analysis.csv"):
        """Export completed trades to CSV"""
        if not self.trades:
            print("No trades to export")
            return
        
        completed_trades = [t for t in self.trades if 'pnl' in t and 'exit_time' in t]
        if not completed_trades:
            print("No completed trades to export")
            return
        
        # Prepare data for DataFrame
        data = []
        for trade in completed_trades:
            data.append({
                'entry_time': trade.get('entry_time', ''),
                'exit_time': trade.get('exit_time', ''),
                'position_type': trade.get('position_type', ''),
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'pnl': trade.get('pnl', 0),
                'reason': trade.get('reason', ''),
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\nExported {len(data)} trades to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Analyze trading performance from logs')
    parser.add_argument('--days', type=int, help='Analyze last N days only')
    parser.add_argument('--export', action='store_true', help='Export trades to CSV')
    parser.add_argument('--log-dir', default='trade_logs', help='Directory containing trade logs')
    
    args = parser.parse_args()
    
    analyzer = TradeAnalyzer(log_dir=args.log_dir)
    
    print("Loading trades...")
    analyzer.load_trades(days=args.days)
    
    if analyzer.trades:
        analyzer.analyze_performance()
        analyzer.analyze_by_day()
        analyzer.analyze_by_exit_reason()
        analyzer.analyze_trade_duration()
        
        if args.export:
            analyzer.export_to_csv()
    else:
        print("No trades found to analyze")

if __name__ == "__main__":
    main()