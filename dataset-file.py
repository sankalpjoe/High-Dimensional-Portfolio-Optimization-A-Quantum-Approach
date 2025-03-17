"""
Dataset handling and preprocessing for financial time series data.
"""

import os
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
import logging
import yfinance as yf

# Configure logging
logger = logging.getLogger(__name__)

class FinancialDataset:
    """
    Dataset class for financial time series data, with preprocessing
    capabilities and time series cross-validation features.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the financial dataset.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to the configuration file. If None, default config is used.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize attributes
        self.tickers = self.config['data']['default_tickers']
        self.start_date = self.config['data']['default_start_date']
        self.end_date = self.config['data']['default_end_date']
        self.returns_type = self.config['data']['returns_type']
        self.frequency = self.config['data']['frequency']
        
        # Data containers
        self.prices = None
        self.returns = None
        self.features = None
        self.scaler = StandardScaler()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from YAML file.
        
        Parameters:
        -----------
        config_path : str or None
            Path to the configuration file.
            
        Returns:
        --------
        dict
            Configuration dictionary.
        """
        if config_path is None:
            # Use default config path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'default.yaml')
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def fetch_data(self, 
                   tickers: Optional[List[str]] = None, 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical price data from Yahoo Finance.
        
        Parameters:
        -----------
        tickers : list of str, optional
            List of ticker symbols. If None, uses the list from config.
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. If None, uses date from config.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. If None, uses date from config.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of adjusted close prices.
        """
        # Use provided parameters or fall back to defaults
        tickers = tickers or self.tickers
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        try:
            # Download data
            data = yf.download(tickers, start=start_date, end=end_date)
            
            # Extract adjusted close prices
            if len(tickers) == 1:
                # Handle single ticker case
                prices = data['Adj Close'].to_frame(tickers[0])
            else:
                prices = data['Adj Close']
            
            # Resample if needed
            if self.frequency != 'daily':
                if self.frequency == 'weekly':
                    prices = prices.resample('W').last()
                elif self.frequency == 'monthly':
                    prices = prices.resample('M').last()
            
            # Store prices
            self.prices = prices
            
            logger.info(f"Successfully fetched data: {prices.shape}")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def calculate_returns(self, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Parameters:
        -----------
        prices : pd.DataFrame, optional
            DataFrame of asset prices. If None, uses self.prices.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of returns (log or simple).
        """
        prices = prices if prices is not None else self.prices
        
        if prices is None:
            logger.error("No price data available. Fetch data first.")
            raise ValueError("No price data available. Fetch data first.")
        
        try:
            if self.returns_type == 'log':
                # Calculate logarithmic returns
                returns = np.log(prices / prices.shift(1))
            else:
                # Calculate simple returns
                returns = prices.pct_change()
            
            # Drop missing values (first row)
            returns = returns.dropna()
            
            # Store returns
            self.returns = returns
            
            logger.info(f"Calculated {self.returns_type} returns: {returns.shape}")
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise
    
    def engineer_features(self, returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Engineer features from return data for improved portfolio optimization.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            DataFrame of asset returns. If None, uses self.returns.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of engineered features.
        """
        returns = returns if returns is not None else self.returns
        
        if returns is None:
            logger.error("No return data available. Calculate returns first.")
            raise ValueError("No return data available. Calculate returns first.")
        
        try:
            # Initialize features DataFrame
            features = pd.DataFrame(index=returns.index)
            
            # Add rolling statistics for each asset
            for column in returns.columns:
                # 20-day rolling volatility
                features[f"{column}_vol_20d"] = returns[column].rolling(20).std()
                
                # 60-day rolling volatility
                features[f"{column}_vol_60d"] = returns[column].rolling(60).std()
                
                # Momentum features (20-day and 60-day returns)
                features[f"{column}_mom_20d"] = returns[column].rolling(20).sum()
                features[f"{column}_mom_60d"] = returns[column].rolling(60).sum()
                
                # Downside deviation (negative returns only)
                neg_returns = returns[column].copy()
                neg_returns[neg_returns > 0] = 0
                features[f"{column}_downside_20d"] = neg_returns.rolling(20).std()
                
                # Skewness and kurtosis
                features[f"{column}_skew_60d"] = returns[column].rolling(60).skew()
                features[f"{column}_kurt_60d"] = returns[column].rolling(60).kurt()
            
            # Drop missing values from the rolling calculations
            features = features.dropna()
            
            # Store features
            self.features = features
            
            logger.info(f"Engineered features: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    def normalize_features(self, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Normalize features for model training.
        
        Parameters:
        -----------
        features : pd.DataFrame, optional
            DataFrame of features. If None, uses self.features.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of normalized features.
        """
        features = features if features is not None else self.features
        
        if features is None:
            logger.error("No feature data available. Engineer features first.")
            raise ValueError("No feature data available. Engineer features first.")
        
        try:
            # Fit and transform the data
            normalized_data = self.scaler.fit_transform(features)
            
            # Convert back to DataFrame with original index and columns
            normalized_features = pd.DataFrame(
                normalized_data,
                index=features.index,
                columns=features.columns
            )
            
            logger.info(f"Normalized features: {normalized_features.shape}")
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            raise
    
    def create_time_series_splits(self, 
                                 data: Optional[pd.DataFrame] = None,
                                 method: Optional[str] = None,
                                 n_splits: Optional[int] = None,
                                 test_size: Optional[int] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time series cross-validation splits.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame to split. If None, uses self.returns.
        method : str, optional
            Split method ('expanding_window' or 'sliding_window').
            If None, uses the method from config.
        n_splits : int, optional
            Number of splits. If None, uses the value from config.
        test_size : int, optional
            Size of test set (in time units). If None, uses the value from config.
            
        Returns:
        --------
        list of tuples
            List of (train, test) DataFrames for each split.
        """
        data = data if data is not None else self.returns
        
        if data is None:
            logger.error("No data available to split.")
            raise ValueError("No data available to split.")
        
        # Use provided parameters or fall back to defaults from config
        method = method or self.config['training']['cv_method']
        n_splits = n_splits or self.config['training']['n_splits']
        test_size = test_size or self.config['training']['test_size']
        
        try:
            splits = []
            
            if method == 'expanding_window':
                # Expanding window method: train set grows with each iteration
                train_start = 0
                for i in range(n_splits):
                    # Calculate indices
                    train_end = len(data) - (n_splits - i) * test_size
                    test_start = train_end
                    test_end = test_start + test_size
                    
                    # Ensure valid indices
                    if train_end <= train_start or test_end > len(data):
                        continue
                    
                    # Create splits
                    train = data.iloc[train_start:train_end]
                    test = data.iloc[test_start:test_end]
                    
                    splits.append((train, test))
            
            elif method == 'sliding_window':
                # Sliding window method: fixed-size train set that moves forward
                window_size = len(data) - n_splits * test_size
                
                for i in range(n_splits):
                    # Calculate indices
                    train_start = i * test_size
                    train_end = train_start + window_size
                    test_start = train_end
                    test_end = test_start + test_size
                    
                    # Ensure valid indices
                    if test_end > len(data):
                        continue
                    
                    # Create splits
                    train = data.iloc[train_start:train_end]
                    test = data.iloc[test_start:test_end]
                    
                    splits.append((train, test))
            
            else:
                raise ValueError(f"Unknown cross-validation method: {method}")
            
            logger.info(f"Created {len(splits)} time series splits using {method} method")
            return splits
            
        except Exception as e:
            logger.error(f"Error creating time series splits: {e}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset.
        
        Returns:
        --------
        dict
            Dictionary with summary statistics.
        """
        summary = {}
        
        if self.prices is not None:
            summary['prices'] = {
                'shape': self.prices.shape,
                'date_range': (self.prices.index.min(), self.prices.index.max()),
                'missing_values': self.prices.isna().sum().sum()
            }
        
        if self.returns is not None:
            summary['returns'] = {
                'shape': self.returns.shape,
                'mean': self.returns.mean().to_dict(),
                'std': self.returns.std().to_dict(),
                'min': self.returns.min().to_dict(),
                'max': self.returns.max().to_dict(),
                'skew': self.returns.skew().to_dict(),
                'kurt': self.returns.kurtosis().to_dict()
            }
        
        if self.features is not None:
            summary['features'] = {
                'shape': self.features.shape,
                'columns': list(self.features.columns)
            }
        
        return summary

    def get_benchmark_returns(self, benchmark_ticker: Optional[str] = None) -> pd.Series:
        """
        Get returns for a benchmark asset.
        
        Parameters:
        -----------
        benchmark_ticker : str, optional
            Ticker symbol for the benchmark. If None, uses the one from config.
            
        Returns:
        --------
        pd.Series
            Series of benchmark returns.
        """
        benchmark_ticker = benchmark_ticker or self.config['evaluation']['benchmark']
        
        try:
            # Fetch benchmark data
            benchmark_prices = yf.download(benchmark_ticker, 
                                         start=self.start_date, 
                                         end=self.end_date)['Adj Close']
            
            # Calculate returns
            if self.returns_type == 'log':
                benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
            else:
                benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Resample if needed
            if self.frequency != 'daily':
                if self.frequency == 'weekly':
                    benchmark_returns = benchmark_returns.resample('W').sum()
                elif self.frequency == 'monthly':
                    benchmark_returns = benchmark_returns.resample('M').sum()
            
            logger.info(f"Fetched benchmark returns for {benchmark_ticker}")
            return benchmark_returns
            
        except Exception as e:
            logger.error(f"Error fetching benchmark returns: {e}")
            raise


# Example usage (when run as a script)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize dataset
    dataset = FinancialDataset()
    
    # Fetch data
    prices = dataset.fetch_data()
    
    # Calculate returns
    returns = dataset.calculate_returns()
    
    # Engineer features
    features = dataset.engineer_features()
    
    # Normalize features
    normalized_features = dataset.normalize_features()
    
    # Create time series splits
    splits = dataset.create_time_series_splits()
    
    # Get data summary
    summary = dataset.get_data_summary()
    
    # Print summary
    import json
    print(json.dumps(summary, indent=2, default=str))
