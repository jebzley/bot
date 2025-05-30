import pandas as pd
import time
import datetime
import math
from typing import Optional, Dict

from config import TradingConfig
from portfolio import Portfolio
from telegram import TelegramNotifier
from ta import TechnicalAnalysis
from exchange import HyperliquidExchange
from logger import logger

class TradingBot:
    """Main trading bot class"""
    
    def __init__(self, config: TradingConfig, exchange: HyperliquidExchange):
        self.config = config
        self.portfolio = Portfolio(config.INITIAL_CASH)
        self.notifier = TelegramNotifier(bot_instance=self)  
        self.ta = TechnicalAnalysis(config, exchange)
        self.exchange = None
        self.is_running = False
        self.trade_history = []
        self.last_price = None 
        
    def calculate_position_size(self, price: float, signal: str, atr: float, signal_score: int = 0) -> float:
        """Calculate position size using hybrid approach based on config"""
        try:
            base_size = 0.0
            
            # Simple sizing (original method)
            if self.config.POSITION_SIZING_MODE == "simple":
                return self.calculate_simple_position_size(price, signal)
            
            # ATR-based sizing only
            elif self.config.POSITION_SIZING_MODE == "atr" and atr > 0:
                risk_per_trade = self.portfolio.cash * 0.015
                stop_distance = self.config.ATR_MULTIPLIER * atr
                base_size = risk_per_trade / stop_distance
            
            # Signal strength sizing only
            elif self.config.POSITION_SIZING_MODE == "signal":
                base_allocation = self.portfolio.cash * self.config.TRADE_SIZE_PCT
                multiplier = self.get_signal_strength_multiplier(signal_score)
                base_size = (base_allocation * multiplier) / price
            
            # Hybrid sizing (best of both)
            elif self.config.POSITION_SIZING_MODE == "hybrid":
                # Start with ATR-based calculation
                if atr > 0:
                    risk_per_trade = self.portfolio.cash * 0.015
                    stop_distance = self.config.ATR_MULTIPLIER * atr
                    atr_size = risk_per_trade / stop_distance
                else:
                    # Fallback to simple if no ATR
                    atr_size = (self.portfolio.cash * self.config.TRADE_SIZE_PCT) / price
                
                # Apply signal strength multiplier
                multiplier = self.get_signal_strength_multiplier(signal_score)
                base_size = atr_size * multiplier
            
            else:
                # Default to simple
                return self.calculate_simple_position_size(price, signal)
            
            # Apply maximum position size limit (30% of capital)
            max_position_value = self.portfolio.cash * 0.3
            max_position_size = max_position_value / price
            final_size = min(base_size, max_position_size)
            
            # Apply drawdown reduction if needed
            if self.portfolio.in_drawdown:
                final_size *= self.config.POSITION_REDUCTION_FACTOR
                logger.info(f"Position size reduced by {(1-self.config.POSITION_REDUCTION_FACTOR)*100:.0f}% due to drawdown")
            
            # Round and return with correct sign
            if signal == 'buy':
                return math.floor(final_size * 100) / 100
            else:  # sell
                return -math.floor(final_size * 100) / 100
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.calculate_simple_position_size(price, signal)
    
    def get_signal_strength_multiplier(self, score: int) -> float:
        """Get position size multiplier based on signal strength"""
        abs_score = abs(score)
        
        if abs_score >= self.config.SIGNAL_VERY_STRONG:
            return self.config.MULTIPLIER_VERY_STRONG
        elif abs_score >= self.config.SIGNAL_STRONG:
            return self.config.MULTIPLIER_STRONG
        else:
            return self.config.MULTIPLIER_NORMAL
    
    def calculate_simple_position_size(self, price: float, signal: str) -> float:
        """Simple position size calculation as fallback"""
        try:
            if signal == 'buy':
                available_cash = self.portfolio.cash * self.config.TRADE_SIZE_PCT
                max_qty = available_cash / price
                return math.floor(max_qty * 100) / 100
            elif signal == 'sell':
                available_cash = self.portfolio.cash * self.config.TRADE_SIZE_PCT
                max_qty = available_cash / price
                return -math.floor(max_qty * 100) / 100
            return 0.0
        except Exception as e:
            logger.error(f"Error in simple position size calculation: {e}")
            return 0.0
    
    def execute_trade(self, signal: str, price: float, regime: str, atr: float, signal_desc: str, signal_score: int) -> bool:
        """Execute trade based on signal with dynamic stops"""
        try:
            # Check if we've hit daily loss limit
            if self.check_daily_loss_limit():
                logger.warning("Daily loss limit reached - no new trades")
                return False
            
            qty = self.calculate_position_size(price, signal, atr, signal_score)
            
            if abs(qty) < 0.01:
                logger.warning(f"Position size too small: {qty}")
                return False
            
            cost = abs(qty) * price
            
            if cost > self.portfolio.cash:
                logger.warning(f"Insufficient funds: need {cost}, have {self.portfolio.cash}")
                return False
            
            # Execute the trade
            self.portfolio.position = qty
            self.portfolio.cash -= cost
            self.portfolio.entry_price = price
            self.portfolio.entry_time = datetime.datetime.now()
            self.portfolio.position_type = 'long' if qty > 0 else 'short'
            
            # Update last trade date
            self.portfolio.last_trade_date = datetime.datetime.now().date()
            
            # Get position size info for logging
            multiplier = self.get_signal_strength_multiplier(signal_score)
            size_info = f"Size: {abs(qty):.2f} (x{multiplier:.1f})"
            
            # Set stops - use hybrid approach: tighter of ATR or percentage-based
            if signal == 'buy':
                self.portfolio.highest_price = price
                self.portfolio.lowest_price = None
                
                # Calculate both stop types and use the tighter one
                atr_stop = price - (self.config.ATR_MULTIPLIER * atr)
                pct_stop = price * (1 - self.config.STOP_LOSS_PCT)
                self.portfolio.stop_loss_price = max(atr_stop, pct_stop)
                
                # Take profit uses the original percentage
                self.portfolio.take_profit_price = price * (1 + self.config.TAKE_PROFIT_PCT)
                
                msg = f"📈 LONG @ {price:.4f} | {size_info} | SL: {self.portfolio.stop_loss_price:.4f} | TP: {self.portfolio.take_profit_price:.4f} | {regime.upper()} | {signal_desc}"
            else:
                self.portfolio.lowest_price = price
                self.portfolio.highest_price = None
                
                # Calculate both stop types and use the tighter one
                atr_stop = price + (self.config.ATR_MULTIPLIER * atr)
                pct_stop = price * (1 + self.config.STOP_LOSS_PCT)
                self.portfolio.stop_loss_price = min(atr_stop, pct_stop)
                
                # Take profit uses the original percentage
                self.portfolio.take_profit_price = price * (1 - self.config.TAKE_PROFIT_PCT)
                
                msg = f"📉 SHORT @ {price:.4f} | {size_info} | SL: {self.portfolio.stop_loss_price:.4f} | TP: {self.portfolio.take_profit_price:.4f} | {regime.upper()} | {signal_desc}"
            
            logger.info(msg)
            self.notifier.send_message(msg)
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return False
            # Set dynamic stops based on ATR
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return False
    
    def close_position(self, price: float, reason: str) -> bool:
        """Close current position and record trade"""
        try:
            if self.portfolio.position == 0:
                return False
            
            position_value = self.portfolio.get_position_value(price)
            pnl = self.portfolio.get_unrealized_pnl(price)
            
            # Update daily PnL
            self.portfolio.daily_pnl += pnl
            
            # Update consecutive losses/wins
            if pnl < 0:
                self.portfolio.consecutive_losses += 1
            else:
                self.portfolio.consecutive_losses = 0
            
            # Record trade for metrics
            trade_data = {
                'entry_price': self.portfolio.entry_price,
                'exit_price': price,
                'pnl': pnl,
                'position_type': self.portfolio.position_type,
                'entry_time': self.portfolio.entry_time,
                'exit_time': datetime.datetime.now(),
                'reason': reason
            }
            self.trade_history.append(trade_data)
            self.portfolio.daily_trades.append(trade_data)
            
            self.portfolio.cash += position_value
            self.portfolio.position = 0
            self.portfolio.entry_price = None
            self.portfolio.entry_time = None
            self.portfolio.highest_price = None
            self.portfolio.lowest_price = None
            self.portfolio.position_type = None
            self.portfolio.stop_loss_price = None
            self.portfolio.take_profit_price = None
            
            # Update drawdown status after trade
            self.update_drawdown_status()
            
            msg = f"{reason} @ {price:.4f} | PNL: {pnl:.2f} | Total: {self.portfolio.cash:.2f}"
            logger.info(msg)
            self.notifier.send_message(msg)
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def check_exit_conditions(self, price: float, signal: Optional[str], regime: str, df: pd.DataFrame) -> bool:
        """Check various exit conditions including time-based stops"""
        if self.portfolio.position == 0:
            return False
        
        try:
            # Check maximum loss
            if self.portfolio.is_max_loss_exceeded(price, self.config.MAX_DAILY_LOSS_PCT):
                self.close_position(price, "🚨 MAX LOSS STOP")
                return True
            
            # Time-based stop loss - only for positions open too long
            if self.portfolio.entry_time:
                time_in_position = datetime.datetime.now() - self.portfolio.entry_time
                if time_in_position > datetime.timedelta(hours=self.config.MAX_POSITION_DURATION_HOURS):
                    if self.portfolio.get_unrealized_pnl(price) < -self.portfolio.cash * 0.01:  # Only if losing more than 1%
                        self.close_position(price, "⏰ TIME STOP")
                        return True
            
            # Get current ATR for dynamic stops
            current_atr = df.iloc[-1].get('atr', None)
            
            # Long position management
            if self.portfolio.position_type == 'long':
                # Update highest price
                if not self.portfolio.highest_price or price > self.portfolio.highest_price:
                    self.portfolio.highest_price = price
                
                # Dynamic take profit (ATR-based)
                if self.portfolio.take_profit_price and price >= self.portfolio.take_profit_price:
                    self.close_position(price, "💰 TAKE PROFIT (LONG)")
                    return True
                
                # Dynamic stop loss (ATR-based)
                if self.portfolio.stop_loss_price and price <= self.portfolio.stop_loss_price:
                    self.close_position(price, "🛑 STOP LOSS (LONG)")
                    return True
                
                # Trailing stop
                if self.portfolio.highest_price and current_atr:
                    trailing_stop = self.portfolio.highest_price - (self.config.ATR_MULTIPLIER * current_atr)
                    if price <= trailing_stop:
                        self.close_position(price, "🛑 TRAILING STOP (LONG)")
                        return True
                
                # Signal-based exits - exit on opposite signal if we're in profit
                if signal == 'sell' and self.portfolio.get_unrealized_pnl(price) > 0:
                    self.close_position(price, f"📉 SIGNAL EXIT (LONG) - {regime.upper()}")
                    return True
                
                # Also exit if momentum is clearly shifting (MACD histogram declining)
                if len(df) >= 3 and self.portfolio.entry_time <= datetime.datetime.now() - datetime.timedelta(minutes=15):
                    macd_hist_current = df.iloc[-1].get('macd_hist', 0)
                    macd_hist_prev = df.iloc[-2].get('macd_hist', 0)
                    macd_hist_prev2 = df.iloc[-3].get('macd_hist', 0)
                    
                    if (macd_hist_current < macd_hist_prev < macd_hist_prev2 and 
                        macd_hist_current < 0 and 
                        self.portfolio.get_unrealized_pnl(price) > 0 ):
                        self.close_position(price, "📊 MOMENTUM EXIT (LONG)")
                        return True
            
            # Short position management
            elif self.portfolio.position_type == 'short':
                # Update lowest price
                if not self.portfolio.lowest_price or price < self.portfolio.lowest_price:
                    self.portfolio.lowest_price = price
                
                # Dynamic take profit (ATR-based)
                if self.portfolio.take_profit_price and price <= self.portfolio.take_profit_price:
                    self.close_position(price, "💰 TAKE PROFIT (SHORT)")
                    return True
                
                # Dynamic stop loss (ATR-based)
                if self.portfolio.stop_loss_price and price >= self.portfolio.stop_loss_price:
                    self.close_position(price, "🛑 STOP LOSS (SHORT)")
                    return True
                
                # Trailing stop
                if self.portfolio.lowest_price and current_atr:
                    trailing_stop = self.portfolio.lowest_price + (self.config.ATR_MULTIPLIER * current_atr)
                    if price >= trailing_stop:
                        self.close_position(price, "🛑 TRAILING STOP (SHORT)")
                        return True
                
                # Signal-based exits - exit on opposite signal if we're in profit
                if signal == 'buy' and self.portfolio.get_unrealized_pnl(price) > 0:
                    self.close_position(price, f"📈 SIGNAL EXIT (SHORT) - {regime.upper()}")
                    return True
                
                # Also exit if momentum is clearly shifting (MACD histogram rising)
                if len(df) >= 3 and self.portfolio.entry_time <= datetime.datetime.now() - datetime.timedelta(minutes=15):
                    macd_hist_current = df.iloc[-1].get('macd_hist', 0)
                    macd_hist_prev = df.iloc[-2].get('macd_hist', 0)
                    macd_hist_prev2 = df.iloc[-3].get('macd_hist', 0)
                    
                    if (macd_hist_current > macd_hist_prev > macd_hist_prev2 and 
                        macd_hist_current > 0 and 
                        self.portfolio.get_unrealized_pnl(price) > 0):
                        self.close_position(price, "📊 MOMENTUM EXIT (SHORT)")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def log_portfolio_status(self, price: float, regime: str):
        """Log current portfolio status"""
        try:
            position_value = self.portfolio.get_position_value(price)
            unrealized_pnl = self.portfolio.get_unrealized_pnl(price)
            total_value = self.portfolio.get_total_value(price)
            return_pct = self.portfolio.get_total_return_pct(price)
            
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = (f"{now} | Price: {price:.4f} | Cash: {self.portfolio.cash:.2f} | "
                     f"Pos: {position_value:.2f} | UPNL: {unrealized_pnl:.2f} | "
                     f"Total: {total_value:.2f} | Return: {return_pct:.2f}% | Regime: {regime.upper()}")
            
            logger.info(status)
            
        except Exception as e:
            logger.error(f"Error logging portfolio status: {e}")
    
    def simulate_trade(self, signal: Optional[str], price: float, regime: str, df: pd.DataFrame, signal_desc: str, signal_score: int):
        """Main trading logic simulation"""
        try:
            self.log_portfolio_status(price, regime)
            
            # Check exit conditions first
            if self.check_exit_conditions(price, signal, regime, df):
                return
            
            # Entry logic - only enter if no position
            if signal and self.portfolio.position == 0:
                current_atr = df.iloc[-1].get('atr', None)
                if current_atr and signal in ['buy', 'sell']:
                    # Additional entry filter: Don't trade against strong trends
                    current_adx = df.iloc[-1].get('adx', 0)
                    if current_adx > 40:  # Strong trend
                        # Only take trades in direction of trend
                        ema_20 = df.iloc[-1].get('ema_20', price)
                        if (signal == 'buy' and price < ema_20) or (signal == 'sell' and price > ema_20):
                            logger.info(f"Skipping {signal} signal against strong trend (ADX: {current_adx:.1f})")
                            return
                    
                    self.execute_trade(signal, price, regime, current_atr, signal_desc, signal_score)
                    
        except Exception as e:
            logger.error(f"Error in trade simulation: {e}")
    
    def update_drawdown_status(self):
        """Update drawdown status based on recent performance"""
        try:
            # Reset daily stats if new day
            current_date = datetime.datetime.now().date()
            if self.portfolio.last_trade_date and current_date != self.portfolio.last_trade_date:
                self.portfolio.daily_pnl = 0.0
                self.portfolio.daily_trades = []
                self.portfolio.last_trade_date = current_date
            
            # Check daily loss limit
            if self.portfolio.daily_pnl < -self.portfolio.initial_cash * self.config.MAX_DAILY_LOSS_PCT:
                self.portfolio.in_drawdown = True
                logger.warning(f"Daily loss limit reached: ${self.portfolio.daily_pnl:.2f}")
            
            # Check consecutive losses
            if self.portfolio.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
                self.portfolio.in_drawdown = True
                logger.warning(f"Consecutive losses limit reached: {self.portfolio.consecutive_losses}")
            
            # Reset drawdown status if we have a winning streak
            if self.portfolio.consecutive_losses == 0 and self.portfolio.daily_pnl > 0:
                if self.portfolio.in_drawdown:
                    logger.info("Drawdown status cleared - back to normal position sizing")
                self.portfolio.in_drawdown = False
                
        except Exception as e:
            logger.error(f"Error updating drawdown status: {e}")
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        if self.portfolio.daily_pnl < -self.portfolio.initial_cash * self.config.MAX_DAILY_LOSS_PCT:
            return True
        return False
    
    def get_current_price(self) -> Optional[float]:
        try:
            df = self.get_historical_ohlcv(
                self.config.SYMBOL,
                self.config.INTERVAL,
                5
            )
            if df is not None and len(df) > 0:
                price = df.iloc[-1]['close']
                self.last_price = price  # Store for status updates
                return price
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
        
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0
                }
            
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            losing_trades = [t for t in self.trade_history if t['pnl'] < 0]
            
            total_trades = len(self.trade_history)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = gross_profit / win_count if win_count > 0 else 0
            avg_loss = gross_loss / loss_count if loss_count > 0 else 0
            
            largest_win = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def backtest(self):
        """Run backtest on historical data"""
        logger.info("Starting backtest...")
        
        try:
            df = self.get_historical_ohlcv(
                self.config.SYMBOL, 
                self.config.INTERVAL, 
                self.config.BACKTEST_LIMIT
            )
            
            if df is None or len(df) < 100:
                logger.error("Insufficient data for backtest")
                return
            
            df = self.apply_indicators(df)
            if df is None:
                logger.error("Failed to apply indicators for backtest")
                return
            
            # Start backtest from a point where indicators are stable
            start_idx = 100
            
            for i in range(start_idx, len(df)):
                sub_df = df.iloc[:i+1]
                
                # Detect market regime
                regime = self.ta.detect_ta(sub_df)
                
                # Get signal with scoring system
                signal, signal_desc, score = self.get_signal_score(sub_df)
                
                current_price = sub_df.iloc[-1]['close']
                self.simulate_trade(signal, current_price, regime, sub_df, signal_desc, score)
            
            # Final results
            final_price = df.iloc[-1]['close']
            final_value = self.portfolio.get_total_value(final_price)
            total_return = self.portfolio.get_total_return_pct(final_price)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics()
            
            # Calculate regime statistics
            regime_stats = {}
            for entry in self.ta.regime_history:
                regime = entry['regime']
                if regime not in regime_stats:
                    regime_stats[regime] = 0
                regime_stats[regime] += 1
            
            regime_summary = " | ".join([f"{k}: {v}" for k, v in regime_stats.items()])
            
            results = (f"✅ BACKTEST COMPLETE\n"
                      f"Initial: ${self.config.INITIAL_CASH:.2f}\n"
                      f"Final: ${final_value:.2f}\n"
                      f"Return: {total_return:.2f}%\n"
                      f"Regime Distribution: {regime_summary}\n\n"
                      f"📊 PERFORMANCE METRICS:\n"
                      f"Total Trades: {metrics.get('total_trades', 0)}\n"
                      f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
                      f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
                      f"Avg Win: ${metrics.get('avg_win', 0):.2f}\n"
                      f"Avg Loss: ${metrics.get('avg_loss', 0):.2f}\n"
                      f"Largest Win: ${metrics.get('largest_win', 0):.2f}\n"
                      f"Largest Loss: ${metrics.get('largest_loss', 0):.2f}")
            
            logger.info(results)
            self.notifier.send_message(results)
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
    
    def paper_trader(self):
        logger.info("Starting paper trader...")
        self.is_running = True
        
        # Start Telegram command polling
        self.notifier.start_command_polling()
        
        try:
            while self.is_running:
                try:
                    df = self.get_historical_ohlcv(
                        self.config.SYMBOL, 
                        self.config.INTERVAL, 
                        self.config.HISTORICAL_LIMIT
                    )
                    
                    if df is None or len(df) < 100:
                        logger.warning("Insufficient data, skipping iteration")
                        time.sleep(self.config.SLEEP_INTERVAL)
                        continue
                    
                    df = self.apply_indicators(df)
                    if df is None:
                        logger.warning("Failed to apply indicators, skipping iteration")
                        time.sleep(self.config.SLEEP_INTERVAL)
                        continue
                    
                    # Store current price for status updates
                    self.last_price = df.iloc[-1]['close']
                    
                    # Detect market regime
                    regime = self.ta.detect_market_regime(df)
                    
                    # Get signal with scoring system
                    signal, signal_desc, score = self.get_signal_score(df)
                    
                    current_price = df.iloc[-1]['close']
                    
                    self.simulate_trade(signal, current_price, regime, df, signal_desc, score)
                    
                    # Log performance metrics periodically
                    if len(self.trade_history) > 0 and len(self.trade_history) % 10 == 0:
                        metrics = self.calculate_performance_metrics()
                        logger.info(f"Performance Update - Trades: {metrics['total_trades']}, "
                                f"Win Rate: {metrics['win_rate']:.2%}, "
                                f"Profit Factor: {metrics['profit_factor']:.2f}")
                    
                    time.sleep(self.config.SLEEP_INTERVAL)
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal, stopping...")
                    break
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(self.config.SLEEP_INTERVAL)
                    
        except Exception as e:
            logger.error(f"Paper trader failed: {e}")
        finally:
            self.is_running = False
            
            self.notifier.stop_command_polling()
            
            # Final performance report
            if self.trade_history:
                metrics = self.calculate_performance_metrics()
                final_report = (f"📊 FINAL PERFORMANCE REPORT:\n"
                            f"Total Trades: {metrics.get('total_trades', 0)}\n"
                            f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
                            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                logger.info(final_report)
                self.notifier.send_message(final_report)
            
            logger.info("Paper trader stopped")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.notifier.stop_command_polling()
        logger.info("Stop signal sent to trading bot")