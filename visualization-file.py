"""
Visualization utilities for portfolio analysis and evaluation.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import os

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioVisualizer:
    """
    Visualization class for portfolio analysis and evaluation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the portfolio visualizer.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary.
        """
        self.config = config
        self.visualization_config = config['visualization']
        
        # Configure style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Create output directory if it doesn't exist
        if self.visualization_config['save_plots']:
            os.makedirs(self.visualization_config['plots_dir'], exist_ok=True)
    
    def plot_portfolio_weights(self, 
                              weights: Dict[str, float], 
                              top_n: int = 15,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot portfolio weights.
        
        Parameters:
        -----------
        weights : dict
            Dictionary of asset weights.
        top_n : int, optional
            Number of top assets to show.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with weight plot.
        """
        try:
            # Create sorted weights
            sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Limit to top N assets
            if len(sorted_weights) > top_n:
                # Extract top N assets
                top_weights = sorted_weights[:top_n]
                
                # Calculate sum of remaining weights
                other_sum = sum(w for _, w in sorted_weights[top_n:])
                
                # Add "Other" category if needed
                if other_sum > 0:
                    top_weights.append(('Other', other_sum))
            else:
                top_weights = sorted_weights
            
            # Extract data for plotting
            assets, weight_values = zip(*top_weights)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Bar plot
            bars = ax1.bar(range(len(assets)), weight_values, color='skyblue')
            
            # Add labels and title
            ax1.set_xticks(range(len(assets)))
            ax1.set_xticklabels(assets, rotation=45, ha='right')
            ax1.set_ylabel('Weight')
            ax1.set_title('Portfolio Weights (Bar Chart)')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height == 0:
                    continue
                ax1.annotate(f'{height:.2%}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
            
            # Pie chart
            ax2.pie(
                weight_values,
                labels=assets,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                textprops={'fontsize': 10}
            )
            ax2.set_title('Portfolio Weights (Pie Chart)')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if configured
            if self.visualization_config['save_plots']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(
                    self.visualization_config['plots_dir'],
                    f'portfolio_weights_{timestamp}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Portfolio weights plot saved to {filename}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting portfolio weights: {e}")
            raise
    
    def plot_performance_metrics(self, 
                                metrics: Dict[str, float],
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot portfolio performance metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of performance metrics.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with metrics plot.
        """
        try:
            # Define metric groups
            return_metrics = ['expected_return', 'alpha']
            risk_metrics = ['expected_risk', 'max_drawdown', 'var_95', 'cvar_95', 'tracking_error']
            ratio_metrics = ['sharpe_ratio', 'sortino_ratio', 'information_ratio']
            other_metrics = ['beta', 'skewness', 'kurtosis']
            
            # Filter metrics to only include available ones
            return_metrics = [m for m in return_metrics if m in metrics]
            risk_metrics = [m for m in risk_metrics if m in metrics]
            ratio_metrics = [m for m in ratio_metrics if m in metrics]
            other_metrics = [m for m in other_metrics if m in metrics]
            
            # Create figure with grid layout
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, 2, figure=fig)
            
            # Return metrics (top left)
            if return_metrics:
                ax1 = fig.add_subplot(gs[0, 0])
                values = [metrics[m] * 100 for m in return_metrics]  # Convert to percentage
                bars = ax1.bar(return_metrics, values, color='forestgreen')
                ax1.set_title('Return Metrics (%)')
                ax1.set_ylabel('Percentage (%)')
                ax1.grid(axis='y', alpha=0.3)
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax1.annotate(f'{height:.2f}%',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom')
            
            # Risk metrics (top right)
            if risk_metrics:
                ax2 = fig.add_subplot(gs[0, 1])
                values = [abs(metrics[m] * 100) for m in risk_metrics]  # Convert to percentage and use absolute value
                bars = ax2.bar(risk_metrics, values, color='firebrick')
                ax2.set_title('Risk Metrics (%)')
                ax2.set_ylabel('Percentage (%)')
                ax2.grid(axis='y', alpha=0.3)
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax2.annotate(f'{height:.2f}%',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom')
            
            # Ratio metrics (bottom left)
            if ratio_metrics:
                ax3 = fig.add_subplot(gs[1, 0])
                values = [metrics[m] for m in ratio_metrics]
                bars = ax3.bar(ratio_metrics, values, color='royalblue')
                ax3.set_title('Ratio Metrics')
                ax3.set_ylabel('Ratio')
                ax3.grid(axis='y', alpha=0.3)
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax3.annotate(f'{height:.2f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom')
            
            # Other metrics (bottom right)
            if other_metrics:
                ax4 = fig.add_subplot(gs[1, 1])
                values = [metrics[m] for m in other_metrics]
                bars = ax4.bar(other_metrics, values, color='purple')
                ax4.set_title('Other Metrics')
                ax4.grid(axis='y', alpha=0.3)
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax4.annotate(f'{height:.2f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom')
            
            # Add a title to the figure
            plt.suptitle('Portfolio Performance Metrics', fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save plot if configured
            if self.visualization_config['save_plots']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(
                    self.visualization_config['plots_dir'],
                    f'performance_metrics_{timestamp}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Performance metrics plot saved to {filename}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting performance metrics: {e}")
            raise
    
    def plot_efficient_frontier(self, 
                               returns: pd.DataFrame,
                               n_portfolios: int = 1000,
                               highlight_portfolios: Optional[Dict[str, Dict]] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot the efficient frontier along with selected portfolios.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
        n_portfolios : int, optional
            Number of random portfolios to generate.
        highlight_portfolios : dict, optional
            Dictionary of portfolios to highlight on the plot.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with efficient frontier plot.
        """
        try:
            if highlight_portfolios is None:
                highlight_portfolios = {}
                
            # Calculate mean returns and covariance matrix
            mu = returns.mean().values
            sigma = returns.cov().values
            
            # Number of assets
            n = len(mu)
            
            # Generate random portfolios
            weights = np.random.random((n_portfolios, n))
            weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
            
            # Calculate returns and risks
            port_returns = np.dot(weights, mu) * 252  # Annualized
            port_risks = np.array([np.sqrt(w.T @ sigma @ w) * np.sqrt(252) for w in weights])  # Annualized
            
            # Calculate Sharpe ratios
            sharpe_ratios = port_returns / port_risks
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot random portfolios
            scatter = ax.scatter(port_risks, port_returns, c=sharpe_ratios, 
                               cmap='viridis', alpha=0.5, s=10)
            
            # Highlight specified portfolios
            for name, portfolio in highlight_portfolios.items():
                if 'portfolio_weights' in portfolio and 'expected_return' in portfolio and 'expected_risk' in portfolio:
                    weights_array = np.array([portfolio['portfolio_weights'].get(asset, 0) for asset in returns.columns])
                    
                    risk = portfolio['expected_risk']
                    ret = portfolio['expected_return']
                    
                    ax.scatter(risk, ret, s=300, marker='*', label=name)
            
            # Add labels and title
            ax.set_xlabel('Expected Volatility (Annualized)')
            ax.set_ylabel('Expected Return (Annualized)')
            ax.set_title('Efficient Frontier with Highlighted Portfolios')
            
            # Add colorbar
            cbar = fig.colorbar(scatter)
            cbar.set_label('Sharpe Ratio')
            
            # Add legend
            if highlight_portfolios:
                ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save plot if configured
            if self.visualization_config['save_plots']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(
                    self.visualization_config['plots_dir'],
                    f'efficient_frontier_{timestamp}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Efficient frontier plot saved to {filename}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting efficient frontier: {e}")
            raise
    
    def plot_performance_over_time(self, 
                                 portfolio_returns: pd.Series,
                                 benchmark_returns: Optional[pd.Series] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot portfolio performance over time, compared to benchmark if provided.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series of portfolio returns.
        benchmark_returns : pd.Series, optional
            Series of benchmark returns.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with performance plot.
        """
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
            
            # Calculate cumulative returns
            cum_returns = (1 + portfolio_returns).cumprod()
            
            # Plot cumulative returns
            ax1.plot(cum_returns.index, cum_returns, label='Portfolio', linewidth=2)
            
            # Add benchmark if provided
            if benchmark_returns is not None:
                # Align benchmark returns with portfolio returns
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 0:
                    bench_returns_aligned = benchmark_returns.loc[common_index]
                    cum_bench_returns = (1 + bench_returns_aligned).cumprod()
                    ax1.plot(cum_bench_returns.index, cum_bench_returns, label='Benchmark', linewidth=2, alpha=0.7)
            
            # Add labels and title
            ax1.set_title('Cumulative Returns Over Time')
            ax1.set_ylabel('Cumulative Return')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Calculate drawdowns
            running_max = cum_returns.cummax()
            drawdowns = (cum_returns / running_max) - 1
            
            # Plot drawdowns
            ax2.fill_between(drawdowns.index, drawdowns, 0, color='crimson', alpha=0.3)
            ax2.plot(drawdowns.index, drawdowns, color='crimson', linewidth=1)
            
            # Add labels
            ax2.set_title('Drawdowns')
            ax2.set_ylabel('Drawdown')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if configured
            if self.visualization_config['save_plots']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(
                    self.visualization_config['plots_dir'],
                    f'performance_over_time_{timestamp}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Performance over time plot saved to {filename}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting performance over time: {e}")
            raise
    
    def plot_rolling_metrics(self, 
                           portfolio_returns: pd.Series,
                           benchmark_returns: Optional[pd.Series] = None,
                           window: int = 252,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series of portfolio returns.
        benchmark_returns : pd.Series, optional
            Series of benchmark returns.
        window : int, optional
            Rolling window size.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with rolling metrics.
        """
        try:
            # Create figure
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            
            # Calculate rolling return
            rolling_return = portfolio_returns.rolling(window=window).mean() * 252  # Annualize
            
            # Calculate rolling volatility
            rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)  # Annualize
            
            # Calculate rolling Sharpe ratio
            rolling_sharpe = rolling_return / rolling_vol
            
            # Calculate rolling drawdown
            cum_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cum_returns.rolling(window=window).max()
            rolling_drawdown = (cum_returns / rolling_max) - 1
            
            # Plot rolling return
            axs[0, 0].plot(rolling_return.index, rolling_return * 100, label='Portfolio', linewidth=2)  # As percentage
            axs[0, 0].set_title(f'Rolling {window}-Day Return (Annualized)')
            axs[0, 0].set_ylabel('Return (%)')
            axs[0, 0].grid(True, alpha=0.3)
            
            # Plot rolling volatility
            axs[0, 1].plot(rolling_vol.index, rolling_vol * 100, label='Portfolio', linewidth=2)  # As percentage
            axs[0, 1].set_title(f'Rolling {window}-Day Volatility (Annualized)')
            axs[0, 1].set_ylabel('Volatility (%)')
            axs[0, 1].grid(True, alpha=0.3)
            
            # Plot rolling Sharpe ratio
            axs[1, 0].plot(rolling_sharpe.index, rolling_sharpe, label='Portfolio', linewidth=2)
            axs[1, 0].set_title(f'Rolling {window}-Day Sharpe Ratio')
            axs[1, 0].set_ylabel('Sharpe Ratio')
            axs[1, 0].grid(True, alpha=0.3)
            
            # Plot rolling drawdown
            axs[1, 1].fill_between(rolling_drawdown.index, rolling_drawdown * 100, 0, color='crimson', alpha=0.3)  # As percentage
            axs[1, 1].plot(rolling_drawdown.index, rolling_drawdown * 100, color='crimson', linewidth=1)
            axs[1, 1].set_title(f'Rolling {window}-Day Drawdown')
            axs[1, 1].set_ylabel('Drawdown (%)')
            axs[1, 1].grid(True, alpha=0.3)
            
            # Add benchmark metrics if provided
            if benchmark_returns is not None:
                # Align benchmark returns with portfolio returns
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 0:
                    bench_returns_aligned = benchmark_returns.loc[common_index]
                    
                    # Calculate benchmark metrics
                    bench_rolling_return = bench_returns_aligned.rolling(window=window).mean() * 252
                    bench_rolling_vol = bench_returns_aligned.rolling(window=window).std() * np.sqrt(252)
                    bench_rolling_sharpe = bench_rolling_return / bench_rolling_vol
                    
                    # Plot benchmark metrics
                    axs[0, 0].plot(bench_rolling_return.index, bench_rolling_return * 100, 
                                 label='Benchmark', linewidth=2, alpha=0.7)
                    axs[0, 1].plot(bench_rolling_vol.index, bench_rolling_vol * 100, 
                                 label='Benchmark', linewidth=2, alpha=0.7)
                    axs[1, 0].plot(bench_rolling_sharpe.index, bench_rolling_sharpe, 
                                 label='Benchmark', linewidth=2, alpha=0.7)
                    
                    # Add legends
                    axs[0, 0].legend()
                    axs[0, 1].legend()
                    axs[1, 0].legend()
            
            # Add x-labels
            for ax in axs.flat:
                ax.set_xlabel('Date')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if configured
            if self.visualization_config['save_plots']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(
                    self.visualization_config['plots_dir'],
                    f'rolling_metrics_{timestamp}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Rolling metrics plot saved to {filename}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting rolling metrics: {e}")
            raise
    
    def plot_correlation_matrix(self, 
                              returns: pd.DataFrame,
                              figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot correlation matrix for asset returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with correlation matrix.
        """
        try:
            # Calculate correlation matrix
            corr = returns.corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot heatmap
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                       square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
            
            # Add labels and title
            ax.set_title('Asset Correlation Matrix')
            
            # Rotate tick labels
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if configured
            if self.visualization_config['save_plots']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(
                    self.visualization_config['plots_dir'],
                    f'correlation_matrix_{timestamp}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation matrix plot saved to {filename}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {e}")
            raise


# Example usage (when run as a script)
if __name__ == "__main__":
    import yaml
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
    
    # Initialize visualizer
    visualizer = PortfolioVisualizer(config)
    
    # Generate random portfolio weights
    np.random.seed(42)
    assets = returns.columns.tolist()
    weights = np.random.random(len(assets))
    weights = weights / np.sum(weights)
    portfolio_weights = {asset: weight for asset, weight in zip(assets, weights)}
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(list(portfolio_weights.values()))
    
    # Get benchmark returns
    benchmark_returns = dataset.get_benchmark_returns()
    
    # Create sample metrics
    metrics = {
        'expected_return': 0.12,
        'expected_risk': 0.15,
        'sharpe_ratio': 0.8,
        'sortino_ratio': 1.2,
        'max_drawdown': -0.25,
        'alpha': 0.03,
        'beta': 0.9,
        'tracking_error': 0.06,
        'information_ratio': 0.5,
        'skewness': -0.3,
        'kurtosis': 3.5,
        'var_95': -0.02,
        'cvar_95': -0.03
    }
    
    # Create dummy portfolios for efficient frontier
    portfolio1 = {
        'portfolio_weights': portfolio_weights,
        'expected_return': 0.12,
        'expected_risk': 0.15
    }
    
    portfolio2 = {
        'portfolio_weights': {asset: 1/len(assets) for asset in assets},
        'expected_return': 0.10,
        'expected_risk': 0.12
    }
    
    highlight_portfolios = {
        'Portfolio 1': portfolio1,
        'Equal Weight': portfolio2
    }
    
    # Generate plots
    fig1 = visualizer.plot_portfolio_weights(portfolio_weights)
    fig2 = visualizer.plot_performance_metrics(metrics)
    fig3 = visualizer.plot_efficient_frontier(returns, highlight_portfolios=highlight_portfolios)
    fig4 = visualizer.plot_performance_over_time(portfolio_returns, benchmark_returns)
    fig5 = visualizer.plot_rolling_metrics(portfolio_returns, benchmark_returns)
    fig6 = visualizer.plot_correlation_matrix(returns)
    
    plt.show()
