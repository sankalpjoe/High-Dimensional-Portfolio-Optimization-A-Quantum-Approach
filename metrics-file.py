"""
Financial metrics calculation for portfolio evaluation.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioMetrics:
    """
    Class for calculating financial metrics for portfolio evaluation.
    """
    
    @staticmethod
    def calculate_returns(weights: Dict[str, float], returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns given weights and asset returns.
        
        Parameters:
        -----------
        weights : dict
            Dictionary mapping asset names to weights.
        returns : pd.DataFrame
            DataFrame of asset returns.
            
        Returns:
        --------
        pd.Series
            Series of portfolio returns.
        """
        # Create weight vector aligned with returns columns
        weight_vector = np.zeros(len(returns.columns))
        for i, asset in enumerate(returns.columns):
            if asset in weights:
                weight_vector[i] = weights[asset]
        
        # Normalize weights to sum to 1
        if np.sum(weight_vector) > 0:
            weight_vector = weight_vector / np.sum(weight_vector)
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weight_vector)
        
        return portfolio_returns
    
    @staticmethod
    def calculate_basic_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate basic portfolio performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of portfolio returns.
        risk_free_rate : float, optional
            Annualized risk-free rate.
            
        Returns:
        --------
        dict
            Dictionary of performance metrics.
        """
        # Convert annualized risk-free rate to match return frequency
        freq_factor = 252  # Assuming daily returns
        daily_rf = (1 + risk_free_rate) ** (1 / freq_factor) - 1
        
        # Calculate basic metrics
        mean_return = returns.mean() * freq_factor  # Annualized
        volatility = returns.std() * np.sqrt(freq_factor)  # Annualized
        
        # Calculate Sharpe ratio
        excess_return = mean_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility != 0 else 0
        
        # Calculate drawdowns
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        
        # Calculate Sortino ratio (downside risk)
        neg_returns = returns.copy()
        neg_returns[neg_returns > 0] = 0
        downside_deviation = neg_returns.std() * np.sqrt(freq_factor)
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        # Calculate Calmar ratio (return / max drawdown)
        calmar_ratio = -mean_return / max_drawdown if max_drawdown != 0 else 0
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Compile results
        metrics = {
            'expected_return': mean_return,
            'expected_risk': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
        
        return metrics
    
    @staticmethod
    def calculate_relative_metrics(returns: pd.Series, 
                                 benchmark_returns: pd.Series,
                                 risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate benchmark-relative performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of portfolio returns.
        benchmark_returns : pd.Series
            Series of benchmark returns.
        risk_free_rate : float, optional
            Annualized risk-free rate.
            
        Returns:
        --------
        dict
            Dictionary of relative performance metrics.
        """
        # Align returns with benchmark
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            logger.warning("No common dates between portfolio and benchmark returns")
            return {}
        
        port_returns = returns.loc[common_index]
        bench_returns = benchmark_returns.loc[common_index]
        
        # Convert annualized risk-free rate to match return frequency
        freq_factor = 252  # Assuming daily returns
        daily_rf = (1 + risk_free_rate) ** (1 / freq_factor) - 1
        
        # Calculate CAPM regression: excess_return = alpha + beta * excess_benchmark_return + epsilon
        excess_port_returns = port_returns - daily_rf
        excess_bench_returns = bench_returns - daily_rf
        
        # Calculate beta and alpha using covariance and variance
        covariance = np.cov(excess_port_returns, excess_bench_returns)[0, 1]
        benchmark_variance = np.var(excess_bench_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Calculate alpha (annualized)
        alpha = (port_returns.mean() - daily_rf - beta * (bench_returns.mean() - daily_rf)) * freq_factor
        
        # Calculate tracking error (annualized)
        tracking_error = (port_returns - bench_returns).std() * np.sqrt(freq_factor)
        
        # Calculate information ratio
        excess_return = (port_returns.mean() - bench_returns.mean()) * freq_factor
        information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
        
        # Calculate up/down capture ratios
        up_months = bench_returns > 0
        down_months = bench_returns < 0
        
        if sum(up_months) > 0:
            up_capture = (port_returns[up_months].mean() / bench_returns[up_months].mean()) * 100
        else:
            up_capture = 0
            
        if sum(down_months) > 0:
            down_capture = (port_returns[down_months].mean() / bench_returns[down_months].mean()) * 100
        else:
            down_capture = 0
        
        # Calculate win rate (percentage of months outperforming benchmark)
        win_rate = (port_returns > bench_returns).mean() * 100
        
        # Calculate relative outperformance (annualized)
        relative_return = (port_returns.mean() - bench_returns.mean()) * freq_factor
        
        # Compile results
        metrics = {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'win_rate': win_rate,
            'relative_return': relative_return
        }
        
        return metrics
    
    @staticmethod
    def calculate_risk_adjusted_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate additional risk-adjusted performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of portfolio returns.
        risk_free_rate : float, optional
            Annualized risk-free rate.
            
        Returns:
        --------
        dict
            Dictionary of risk-adjusted performance metrics.
        """
        # Convert annualized risk-free rate to match return frequency
        freq_factor = 252  # Assuming daily returns
        daily_rf = (1 + risk_free_rate) ** (1 / freq_factor) - 1
        
        # Calculate Treynor ratio (needs beta, which requires a benchmark)
        # This is a placeholder; the actual calculation happens in calculate_relative_metrics
        
        # Calculate Omega ratio (probability-weighted ratio of gains versus losses)
        threshold = daily_rf  # Or use 0
        omega_numerator = np.mean(np.maximum(returns - threshold, 0))
        omega_denominator = np.mean(np.maximum(threshold - returns, 0))
        omega_ratio = omega_numerator / omega_denominator if omega_denominator != 0 else 0
        
        # Calculate Kappa ratio (similar to Omega, but using different moment)
        kappa_numerator = returns.mean() - threshold
        kappa_denominator = np.sqrt(np.mean(np.maximum(threshold - returns, 0) ** 2))
        kappa_ratio = kappa_numerator / kappa_denominator if kappa_denominator != 0 else 0
        
        # Calculate Gain-Loss ratio
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        gain_loss_ratio = gains.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
        
        # Calculate Upside Potential ratio
        upside_returns = np.maximum(returns - threshold, 0)
        downside_deviation = np.sqrt(np.mean(np.minimum(returns - threshold, 0) ** 2))
        upside_potential = upside_returns.mean() / downside_deviation if downside_deviation != 0 else 0
        
        # Compile results
        metrics = {
            'omega_ratio': omega_ratio,
            'kappa_ratio': kappa_ratio,
            'gain_loss_ratio': gain_loss_ratio,
            'upside_potential': upside_potential
        }
        
        return metrics
    
    @staticmethod
    def calculate_drawdown_metrics(returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate drawdown-related metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of portfolio returns.
            
        Returns:
        --------
        dict
            Dictionary of drawdown metrics and data.
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cum_returns / running_max) - 1
        
        # Calculate max drawdown
        max_drawdown = drawdowns.min()
        max_drawdown_date = drawdowns.idxmin()
        
        # Find drawdown start date (last peak before max drawdown)
        temp = pd.Series(running_max, index=drawdowns.index)
        temp = temp.loc[:max_drawdown_date]
        max_peak_date = temp.idxmax()
        
        # Find drawdown end date (next recovery after max drawdown)
        temp = drawdowns.loc[max_drawdown_date:]
        try:
            recovery_date = temp[temp >= 0].index[0] if len(temp[temp >= 0]) > 0 else None
        except:
            recovery_date = None
        
        # Calculate drawdown duration
        if recovery_date:
            drawdown_duration = (recovery_date - max_peak_date).days
            recovery_duration = (recovery_date - max_drawdown_date).days
        else:
            drawdown_duration = (drawdowns.index[-1] - max_peak_date).days
            recovery_duration = None
        
        # Calculate average drawdown
        avg_drawdown = drawdowns[drawdowns < 0].mean()
        
        # Calculate average recovery time
        recover_durations = []
        in_drawdown = False
        drawdown_start = None
        
        for i, (date, value) in enumerate(drawdowns.items()):
            if not in_drawdown and value < 0:
                # Start of a drawdown
                in_drawdown = True
                drawdown_start = date
            elif in_drawdown and value >= 0:
                # End of a drawdown
                in_drawdown = False
                duration = (date - drawdown_start).days
                recover_durations.append(duration)
        
        avg_recovery_time = np.mean(recover_durations) if recover_durations else None
        
        # Count drawdowns
        drawdown_count = len(recover_durations)
        
        # Calculate time underwater (% of time in drawdown)
        time_underwater = (drawdowns < 0).mean() * 100
        
        # Compile results
        metrics = {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'drawdown_start_date': max_peak_date,
            'drawdown_end_date': recovery_date,
            'drawdown_duration': drawdown_duration,
            'recovery_duration': recovery_duration,
            'avg_drawdown': avg_drawdown,
            'avg_recovery_time': avg_recovery_time,
            'drawdown_count': drawdown_count,
            'time_underwater': time_underwater,
            'drawdowns': drawdowns
        }
        
        return metrics
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, 
                                window: int = 252,
                                risk_free_rate: float = 0.0) -> Dict[str, pd.Series]:
        """
        Calculate rolling performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of portfolio returns.
        window : int, optional
            Rolling window size.
        risk_free_rate : float, optional
            Annualized risk-free rate.
            
        Returns:
        --------
        dict
            Dictionary of rolling performance metrics.
        """
        # Convert annualized risk-free rate to match return frequency
        freq_factor = 252  # Assuming daily returns
        daily_rf = (1 + risk_free_rate) ** (1 / freq_factor) - 1
        
        # Calculate rolling return
        rolling_return = returns.rolling(window=window).mean() * freq_factor
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(freq_factor)
        
        # Calculate rolling Sharpe ratio
        rolling_excess_return = rolling_return - risk_free_rate
        rolling_sharpe = rolling_excess_return / rolling_vol
        
        # Calculate rolling Sortino ratio
        def downside_deviation(x):
            neg_returns = x.copy()
            neg_returns[neg_returns > 0] = 0
            return neg_returns.std() * np.sqrt(freq_factor)
        
        rolling_downside = returns.rolling(window=window).apply(downside_deviation, raw=True)
        rolling_sortino = rolling_excess_return / rolling_downside
        
        # Calculate rolling max drawdown
        def rolling_max_drawdown(x):
            cum_returns = (1 + x).cumprod()
            running_max = cum_returns.cummax()
            drawdowns = (cum_returns / running_max) - 1
            return drawdowns.min()
        
        rolling_drawdown = returns.rolling(window=window).apply(rolling_max_drawdown, raw=False)
        
        # Calculate rolling skewness and kurtosis
        rolling_skew = returns.rolling(window=window).skew()
        rolling_kurt = returns.rolling(window=window).kurt()
        
        # Calculate rolling VaR and CVaR
        rolling_var_95 = returns.rolling(window=window).quantile(0.05)
        
        def rolling_cvar(x):
            var_95 = np.percentile(x, 5)
            return x[x <= var_95].mean()
        
        rolling_cvar_95 = returns.rolling(window=window).apply(rolling_cvar, raw=True)
        
        # Compile results
        metrics = {
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_sortino': rolling_sortino,
            'rolling_max_drawdown': rolling_drawdown,
            'rolling_skewness': rolling_skew,
            'rolling_kurtosis': rolling_kurt,
            'rolling_var_95': rolling_var_95,
            'rolling_cvar_95': rolling_cvar_95
        }
        
        return metrics
    
    @staticmethod
    def calculate_all_metrics(weights: Dict[str, float], 
                            returns: pd.DataFrame,
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: float = 0.0,
                            calculate_rolling: bool = False) -> Dict[str, Any]:
        """
        Calculate all portfolio metrics.
        
        Parameters:
        -----------
        weights : dict
            Dictionary mapping asset names to weights.
        returns : pd.DataFrame
            DataFrame of asset returns.
        benchmark_returns : pd.Series, optional
            Series of benchmark returns.
        risk_free_rate : float, optional
            Annualized risk-free rate.
        calculate_rolling : bool, optional
            Whether to calculate rolling metrics.
            
        Returns:
        --------
        dict
            Dictionary of all performance metrics.
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = PortfolioMetrics.calculate_returns(weights, returns)
            
            # Calculate basic metrics
            basic_metrics = PortfolioMetrics.calculate_basic_metrics(portfolio_returns, risk_free_rate)
            
            # Calculate risk-adjusted metrics
            risk_adjusted_metrics = PortfolioMetrics.calculate_risk_adjusted_metrics(portfolio_returns, risk_free_rate)
            
            # Calculate drawdown metrics
            drawdown_metrics = PortfolioMetrics.calculate_drawdown_metrics(portfolio_returns)
            
            # Combine metrics
            all_metrics = {**basic_metrics, **risk_adjusted_metrics}
            all_metrics.update({k: v for k, v in drawdown_metrics.items() if k != 'drawdowns'})
            
            # Add benchmark-relative metrics if benchmark provided
            if benchmark_returns is not None:
                relative_metrics = PortfolioMetrics.calculate_relative_metrics(portfolio_returns, benchmark_returns, risk_free_rate)
                all_metrics.update(relative_metrics)
            
            # Add rolling metrics if requested
            if calculate_rolling:
                rolling_metrics = PortfolioMetrics.calculate_rolling_metrics(portfolio_returns, risk_free_rate=risk_free_rate)
                all_metrics['rolling'] = rolling_metrics
            
            # Add drawdown series
            all_metrics['drawdowns'] = drawdown_metrics['drawdowns']
            
            # Add portfolio returns
            all_metrics['portfolio_returns'] = portfolio_returns
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error calculating all metrics: {e}")
            raise


# Example usage (when run as a script)
if __name__ == "__main__":
    import yaml
    import matplotlib.pyplot as plt
    import numpy as np
    from data.dataset import FinancialDataset
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open('../config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize dataset
    dataset = FinancialDataset(config_path='../config/default.yaml')
    
    # Fetch data
    prices = dataset.fetch_data()
    returns = dataset.calculate_returns()
    
    # Get benchmark returns
    benchmark_returns = dataset.get_benchmark_returns()
    
    # Generate random portfolio weights
    np.random.seed(42)
    assets = returns.columns.tolist()
    weights = np.random.random(len(assets))
    weights = weights / np.sum(weights)
    portfolio_weights = {asset: weight for asset, weight in zip(assets, weights)}
    
    # Calculate all metrics
    metrics = PortfolioMetrics.calculate_all_metrics(
        portfolio_weights, returns, benchmark_returns, 
        risk_free_rate=0.02, calculate_rolling=True
    )
    
    # Print basic metrics
    print("\nBasic Metrics:")
    for key in ['expected_return', 'expected_risk', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown']:
        print(f"  {key}: {metrics[key]:.4f}")
    
    # Print relative metrics if benchmark provided
    if benchmark_returns is not None:
        print("\nRelative Metrics:")
        for key in ['alpha', 'beta', 'information_ratio', 'tracking_error']:
            print(f"  {key}: {metrics[key]:.4f}")
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative returns
    portfolio_cum_returns = (1 + metrics['portfolio_returns']).cumprod()
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    
    # Align benchmark returns with portfolio returns
    common_index = portfolio_cum_returns.index.intersection(benchmark_cum_returns.index)
    port_cum_aligned = portfolio_cum_returns.loc[common_index]
    bench_cum_aligned = benchmark_cum_returns.loc[common_index]
    
    # Plot
    plt.plot(port_cum_aligned.index, port_cum_aligned, label='Portfolio')
    plt.plot(bench_cum_aligned.index, bench_cum_aligned, label='Benchmark')
    
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    
    plt.plot(metrics['drawdowns'].index, metrics['drawdowns'] * 100)
    plt.fill_between(metrics['drawdowns'].index, metrics['drawdowns'] * 100, 0, color='crimson', alpha=0.3)
    
    plt.title('Portfolio Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
