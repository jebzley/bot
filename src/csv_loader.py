import pandas as pd
import os
from typing import Optional
from logger import logger

class CSVDataLoader:
    """Load and parse historical data from CSV files for backtesting"""
    
    @staticmethod
    def load_csv_data(filepath: str) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data from CSV file
        
        CSV columns expected:
        0: Open time (microseconds)
        1: Open price
        2: High price
        3: Low price
        4: Close price
        5: Volume
        6-11: Other data (ignored for our purposes)
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"CSV file not found: {filepath}")
                return None
            
            logger.info(f"Loading CSV data from {filepath}")
            # Define column names based on the documentation
            column_names = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
            
            # Read CSV with specific columns
            df = pd.read_csv(
                filepath,
                names=column_names,
                usecols=['open_time', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp from microseconds to datetime
            # The timestamps appear to be in microseconds (16 digits)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='us')
            
            # Drop the original time column
            df = df.drop('open_time', axis=1)
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values
            initial_len = len(df)
            df = df.dropna()
            if len(df) < initial_len:
                logger.warning(f"Dropped {initial_len - len(df)} rows with invalid data")
            
            # Reorder columns to match expected format
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Log info about loaded data
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"Price range: {df['close'].min():.8f} to {df['close'].max():.8f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            return None
    
    @staticmethod
    def resample_to_interval(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe if needed
        
        Args:
            df: DataFrame with OHLCV data
            target_interval: Target interval (e.g., '1h', '15m', '5m')
        
        Returns:
            Resampled DataFrame
        """
        try:
            # Set timestamp as index for resampling
            df_resampled = df.set_index('timestamp')
            
            # Define aggregation rules for OHLCV
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Convert interval to pandas frequency
            interval_map = {
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D'
            }
            
            if target_interval not in interval_map:
                logger.warning(f"Unknown interval {target_interval}, returning original data")
                return df
            
            freq = interval_map[target_interval]
            
            # Resample the data
            df_resampled = df_resampled.resample(freq).agg(agg_rules)
            
            # Drop any rows with NaN values
            df_resampled = df_resampled.dropna()
            
            # Reset index to get timestamp back as column
            df_resampled = df_resampled.reset_index()
            
            logger.info(f"Resampled data to {target_interval}: {len(df)} -> {len(df_resampled)} rows")
            
            return df_resampled
            
        except Exception as e:
            logger.error(f"Failed to resample data: {e}")
            return df