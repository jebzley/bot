from dataclasses import dataclass

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    SYMBOL: str = "FARTCOIN/USDC:USDC"
    INTERVAL: str = "5m"
    TRADE_SIZE_PCT: float = 0.3  
    INITIAL_CASH: float = 1000
    TRAILING_STOP_PCT: float = 0.02
    TAKE_PROFIT_PCT: float = 0.03 
    STOP_LOSS_PCT: float = 0.02 
    
    MAX_POSITION_DURATION_HOURS: int = 4
    
    # Drawdown management
    MAX_DAILY_LOSS_PCT: float = 0.05  
    MAX_CONSECUTIVE_LOSSES: int = 4  
    POSITION_REDUCTION_FACTOR: float = 0.5 
    
    # Position sizing mode
    POSITION_SIZING_MODE: str = "hybrid"  # "simple", "atr", "signal", or "hybrid"
    
    # Signal strength position multipliers
    SIGNAL_VERY_STRONG: int = 40
    SIGNAL_STRONG: int = 25
    MULTIPLIER_VERY_STRONG: float = 1.5
    MULTIPLIER_STRONG: float = 1.0
    MULTIPLIER_NORMAL: float = 0.7
    
    # Signal scoring thresholds
    BUY_SIGNAL_THRESHOLD: int = 15  
    SELL_SIGNAL_THRESHOLD: int = -15  
    
    # MACD Strategy Parameters
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    # RSI Parameters
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 35  
    RSI_OVERBOUGHT: float = 65  
    
    # ATR Parameters
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER: float = 1.5  
    
    # Volume Parameters
    VOLUME_SMA_PERIOD: int = 20
    VOLUME_SPIKE_THRESHOLD: float = 1.5
    
    # Market Regime Detection Parameters
    ADX_PERIOD: int = 14
    ADX_TRENDING_THRESHOLD: float = 25
    ADX_RANGING_THRESHOLD: float = 20
    EMA_PERIOD: int = 20
    EMA_TREND_CANDLES: int = 10
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    BB_ZSCORE_THRESHOLD: float = -1.0
    
    SLEEP_INTERVAL: int = 20
    HISTORICAL_LIMIT: int = 300
    BACKTEST_LIMIT: int = 21600