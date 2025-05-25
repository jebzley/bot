from logger import logger
from config import TradingConfig
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands
import pandas as pd
from typing import Tuple

class MarketRegime:
    """Market regime detection and strategy selection"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.current_regime = "neutral"
        self.regime_history = []
    
    def calculate_bollinger_band_width_zscore(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Calculate Z-score of Bollinger Band width"""
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
    
    def check_ema_trend_persistence(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if price has been above/below EMA for specified candles"""
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
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime: trending, ranging, or neutral"""
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
            
            # Calculate Bollinger Band width Z-score
            bb_zscore = self.calculate_bollinger_band_width_zscore(df_copy)
            
            # Check EMA trend persistence
            is_trending, trend_direction = self.check_ema_trend_persistence(df_copy)
            
            # Market regime logic
            if (current_adx < self.config.ADX_RANGING_THRESHOLD and 
                bb_zscore < self.config.BB_ZSCORE_THRESHOLD):
                regime = "ranging"
            elif (current_adx > self.config.ADX_TRENDING_THRESHOLD and is_trending):
                regime = "trending"
            else:
                regime = "neutral"
            
            # Log regime changes
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