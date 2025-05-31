from logger import logger
from config import TradingConfig
from exchange import HyperliquidExchange
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
import pandas as pd
from typing import Tuple, Optional

class TechnicalAnalysis:
    def __init__(self, config: TradingConfig, exchange: HyperliquidExchange):
        self.config = config
        self.exchange = exchange.hyperliquid
        self.current_regime = "neutral"
        self.regime_history = []

    def get_historical_ohlcv(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            if not self.exchange:
                logger.error("Exchange not initialized")
                return None
            
            fetch_limit = min(limit + 100, 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=interval, limit=fetch_limit)
            
            if not ohlcv:
                logger.error("No OHLCV data received")
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            initial_len = len(df)
            logger.debug(f"Fetched {initial_len} candles")
            
            ohlcv_nan = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum()
            if ohlcv_nan > 0:
                logger.warning(f"OHLCV data contains {ohlcv_nan} NaN values")
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')
                df = df.dropna()
                logger.info(f"After OHLCV cleanup: {len(df)} rows remaining")
            
            duplicates = df.duplicated(subset=['timestamp']).sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate timestamps, removing...")
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            result = df.tail(limit)
            logger.debug(f"Returning {len(result)} clean candles")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}")
            return None
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        try:
            if len(df) < max(self.config.ADX_PERIOD, self.config.EMA_PERIOD, self.config.BB_PERIOD) + 50:
                logger.warning("Insufficient data for market regime detection")
                return "neutral"
            
            df_copy = df.copy()
            
            # Calculate ADX
            adx_indicator = ADXIndicator(
                high=df_copy['high'], 
                low=df_copy['low'], 
                close=df_copy['close'], 
                window=self.config.ADX_PERIOD
            )
            df_copy.loc[:, 'adx'] = adx_indicator.adx()
            current_adx = df_copy['adx'].iloc[-1]
            
            if pd.isna(current_adx):
                logger.warning("ADX calculation returned NaN")
                return "neutral"
            
            bb_zscore = self.calculate_bollinger_band_width_zscore(df_copy)
            is_trending, trend_direction = self.check_ema_trend_persistence(df_copy)
            
            if (current_adx < self.config.ADX_RANGING_THRESHOLD and 
                bb_zscore < self.config.BB_ZSCORE_THRESHOLD):
                regime = "ranging"
            elif (current_adx > self.config.ADX_TRENDING_THRESHOLD and is_trending):
                regime = "trending"
            else:
                regime = "neutral"
            
            if regime != self.current_regime:
                logger.info(f"Market regime changed: {self.current_regime} -> {regime}")
                logger.info(f"ADX: {current_adx:.2f}, BB Z-score: {bb_zscore:.2f}, "
                           f"Trend persistent: {is_trending} ({trend_direction})")
                self.current_regime = regime
            
            self.regime_history.append({
                'timestamp': df_copy.iloc[-1]['timestamp'],
                'regime': regime,
                'adx': current_adx,
                'bb_zscore': bb_zscore,
                'trend_persistent': is_trending,
                'trend_direction': trend_direction
            })
            
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-500:]
            
            return regime
            
        except Exception as e:
            logger.error(f"Error in market regime detection: {e}")
            return "neutral"
        
    def calculate_bollinger_band_width_zscore(self, df: pd.DataFrame, lookback: int = 50) -> float:
        try:
            df_copy = df.copy()
            
            bb = BollingerBands(close=df_copy['close'], window=self.config.BB_PERIOD, window_dev=self.config.BB_STD)
            df_copy.loc[:, 'bb_upper'] = bb.bollinger_hband()
            df_copy.loc[:, 'bb_lower'] = bb.bollinger_lband()
            df_copy.loc[:, 'bb_width'] = (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['close']
            
            if len(df_copy) < lookback + self.config.BB_PERIOD:
                return 0.0
            
            recent_widths = df_copy['bb_width'].dropna().tail(lookback)
            if len(recent_widths) < 2:
                return 0.0
            
            current_width = recent_widths.iloc[-1]
            mean_width = recent_widths.mean()
            std_width = recent_widths.std()
            
            if std_width == 0:
                return 0.0
            
            zscore = (current_width - mean_width) / std_width
            return zscore
            
        except Exception as e:
            logger.error(f"Error calculating BB width Z-score: {e}")
            return 0.0
    
    def detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            df_copy['rsi_divergence'] = 0
            
            if len(df_copy) < lookback + 5:
                return df_copy
            
            for i in range(lookback + 5, len(df_copy)):
                recent_data = df_copy.iloc[i-lookback:i]
                
                # Find price peaks and troughs
                price_highs = recent_data['high'].rolling(window=3).max()
                price_lows = recent_data['low'].rolling(window=3).min()
                
                # Find RSI peaks and troughs
                rsi_highs = recent_data['rsi'].rolling(window=3).max()
                rsi_lows = recent_data['rsi'].rolling(window=3).min()
                
                # Check for bullish divergence (price lower low, RSI higher low)
                if (recent_data['low'].iloc[-1] < price_lows.iloc[-5] and 
                    recent_data['rsi'].iloc[-1] > rsi_lows.iloc[-5] and
                    recent_data['rsi'].iloc[-1] < self.config.RSI_OVERSOLD + 10):
                    df_copy.loc[df_copy.index[i], 'rsi_divergence'] = 1
                
                # Check for bearish divergence (price higher high, RSI lower high)
                elif (recent_data['high'].iloc[-1] > price_highs.iloc[-5] and 
                      recent_data['rsi'].iloc[-1] < rsi_highs.iloc[-5] and
                      recent_data['rsi'].iloc[-1] > self.config.RSI_OVERBOUGHT - 10):
                    df_copy.loc[df_copy.index[i], 'rsi_divergence'] = -1
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
            df['rsi_divergence'] = 0
            return df
        
    def check_ema_trend_persistence(self, df: pd.DataFrame) -> Tuple[bool, str]:
        try:
            df_copy = df.copy()
            
            ema = EMAIndicator(close=df_copy['close'], window=self.config.EMA_PERIOD)
            df_copy.loc[:, 'ema_20'] = ema.ema_indicator()
            
            if len(df_copy) < self.config.EMA_TREND_CANDLES + self.config.EMA_PERIOD:
                return False, "insufficient_data"
            
            recent_closes = df_copy['close'].tail(self.config.EMA_TREND_CANDLES)
            recent_ema = df_copy['ema_20'].tail(self.config.EMA_TREND_CANDLES)
            
            above_ema = (recent_closes > recent_ema).all()
            below_ema = (recent_closes < recent_ema).all()
            
            if above_ema:
                return True, "uptrend"
            elif below_ema:
                return True, "downtrend"
            else:
                return False, "sideways"
                
        except Exception as e:
            logger.error(f"Error checking EMA trend persistence: {e}")
            return False, "error"
    
    def apply_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            required_periods = max(
                self.config.MACD_SLOW + self.config.MACD_SIGNAL,
                self.config.ADX_PERIOD,
                self.config.EMA_PERIOD,
                self.config.BB_PERIOD,
                self.config.RSI_PERIOD,
                self.config.ATR_PERIOD,
                self.config.VOLUME_SMA_PERIOD
            ) + 50
            
            if len(df) < required_periods:
                logger.error(f"Insufficient data for indicators: need {required_periods}, have {len(df)}")
                return None
            
            df_result = df.copy()
            
            # MACD
            macd = MACD(
                close=df_result['close'], 
                window_slow=self.config.MACD_SLOW, 
                window_fast=self.config.MACD_FAST, 
                window_sign=self.config.MACD_SIGNAL
            )
            df_result.loc[:, 'macd'] = macd.macd()
            df_result.loc[:, 'macd_signal'] = macd.macd_signal()
            df_result.loc[:, 'macd_hist'] = macd.macd_diff()
            
            # RSI
            rsi = RSIIndicator(close=df_result['close'], window=self.config.RSI_PERIOD)
            df_result.loc[:, 'rsi'] = rsi.rsi()
            
            # ATR
            atr = AverageTrueRange(
                high=df_result['high'], 
                low=df_result['low'], 
                close=df_result['close'],
                window=self.config.ATR_PERIOD
            )
            df_result.loc[:, 'atr'] = atr.average_true_range()
            
            # Bollinger Bands
            bb = BollingerBands(
                close=df_result['close'], 
                window=self.config.BB_PERIOD, 
                window_dev=self.config.BB_STD
            )
            df_result.loc[:, 'bb_upper'] = bb.bollinger_hband()
            df_result.loc[:, 'bb_middle'] = bb.bollinger_mavg()
            df_result.loc[:, 'bb_lower'] = bb.bollinger_lband()
            
            # EMA
            ema = EMAIndicator(close=df_result['close'], window=self.config.EMA_PERIOD)
            df_result.loc[:, 'ema_20'] = ema.ema_indicator()
            
            # ADX
            adx_indicator = ADXIndicator(
                high=df_result['high'], 
                low=df_result['low'], 
                close=df_result['close'], 
                window=self.config.ADX_PERIOD
            )
            df_result.loc[:, 'adx'] = adx_indicator.adx()
            
            # Volume indicators
            df_result.loc[:, 'volume_sma'] = df_result['volume'].rolling(window=self.config.VOLUME_SMA_PERIOD).mean()
            df_result.loc[:, 'volume_ratio'] = df_result['volume'] / df_result['volume_sma']
            
            # RSI Divergence Detection
            df_result = self.detect_rsi_divergence(df_result)
            
            # Clean NaN values
            initial_len = len(df_result)
            df_result = df_result.dropna()
            dropped_rows = initial_len - len(df_result)
            
            if dropped_rows > 0:
                logger.debug(f"Dropped {dropped_rows} indicator initialization rows, {len(df_result)} trading rows remaining")
            
            min_required = 50
            if len(df_result) < min_required:
                logger.error(f"Insufficient data after indicator calculation: {len(df_result)} < {min_required}")
                return None
            
            logger.debug(f"Indicator calculation successful: {len(df_result)} valid rows")
            return df_result
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return None
    
    def get_signal_score(self, df: pd.DataFrame) -> Tuple[Optional[str], str, int]:
        try:
            if len(df) < 2:
                return None, "Insufficient data", 0
            
            score = 0
            signals = []
            
            # Get market regime for context
            regime = self.market_regime.detect_market_regime(df)
            
            # MACD signal (+/- 20 points)
            current_macd = df.iloc[-1]['macd']
            current_signal = df.iloc[-1]['macd_signal']
            prev_macd = df.iloc[-2]['macd']
            prev_signal = df.iloc[-2]['macd_signal']
            
            if not pd.isna(current_macd) and not pd.isna(current_signal):
                if current_macd > current_signal and prev_macd <= prev_signal:
                    score += 20
                    signals.append("MACD_BUY")
                elif current_macd < current_signal and prev_macd >= prev_signal:
                    score -= 20
                    signals.append("MACD_SELL")
            
            # RSI signal (+/- 15 points) - only in ranging markets
            current_rsi = df.iloc[-1]['rsi']
            if not pd.isna(current_rsi):
                if current_rsi < self.config.RSI_OVERSOLD:
                    score += 15
                    signals.append("RSI_OVERSOLD")
                elif current_rsi > self.config.RSI_OVERBOUGHT:
                    score -= 15
                    signals.append("RSI_OVERBOUGHT")
            
            # RSI Divergence (+/- 25 points - stronger signal)
            rsi_divergence = df.iloc[-1].get('rsi_divergence', 0)
            if rsi_divergence == 1:
                score += 25
                signals.append("RSI_BULL_DIV")
            elif rsi_divergence == -1:
                score -= 25
                signals.append("RSI_BEAR_DIV")
            
            # Volume confirmation (+/- 10 points)
            volume_ratio = df.iloc[-1]['volume_ratio']
            if not pd.isna(volume_ratio) and volume_ratio > self.config.VOLUME_SPIKE_THRESHOLD:
                if score > 0:
                    score += 10
                    signals.append("VOL_CONFIRM_BUY")
                elif score < 0:
                    score -= 10
                    signals.append("VOL_CONFIRM_SELL")
            
            # Bollinger Band signals - stronger in ranging markets
            current_close = df.iloc[-1]['close']
            current_upper = df.iloc[-1]['bb_upper']
            current_lower = df.iloc[-1]['bb_lower']
            prev_close = df.iloc[-2]['close']
            
            if not pd.isna(current_upper) and not pd.isna(current_lower):
                if regime == "ranging":
                    # Strong BB signals in ranging markets
                    if current_close <= current_lower and prev_close > df.iloc[-2]['bb_lower']:
                        score += 20  # Increased from 15
                        signals.append("BB_OVERSOLD_RANGE")
                    elif current_close >= current_upper and prev_close < df.iloc[-2]['bb_upper']:
                        score -= 20  # Increased from 15
                        signals.append("BB_OVERBOUGHT_RANGE")
                else:
                    # Weaker BB signals in trending markets
                    if current_close <= current_lower and prev_close > df.iloc[-2]['bb_lower']:
                        score += 10
                        signals.append("BB_OVERSOLD")
                    elif current_close >= current_upper and prev_close < df.iloc[-2]['bb_upper']:
                        score -= 10
                        signals.append("BB_OVERBOUGHT")
            
            # EMA trend confirmation bonus (for trending markets)
            if regime == "trending":
                current_close = df.iloc[-1]['close']
                ema_20 = df.iloc[-1].get('ema_20', current_close)
                if not pd.isna(ema_20):
                    if current_close > ema_20 and score > 0:
                        score += 5
                        signals.append("EMA_TREND_CONFIRM")
                    elif current_close < ema_20 and score < 0:
                        score -= 5
                        signals.append("EMA_TREND_CONFIRM")
            signal_description = f"Score: {score} [{', '.join(signals)}]"
            
            if score >= self.config.BUY_SIGNAL_THRESHOLD:
                return 'buy', signal_description, score
            elif score <= self.config.SELL_SIGNAL_THRESHOLD:
                return 'sell', signal_description, score
            
            return None, signal_description, score
            
        except Exception as e:
            logger.error(f"Error generating signal score: {e}")
            return None, "Error", 0
    
    def get_signal_strength_multiplier(self, score: int) -> float:
        abs_score = abs(score)
        
        if abs_score >= self.config.SIGNAL_VERY_STRONG:
            return self.config.MULTIPLIER_VERY_STRONG
        elif abs_score >= self.config.SIGNAL_STRONG:
            return self.config.MULTIPLIER_STRONG
        else:
            return self.config.MULTIPLIER_NORMAL