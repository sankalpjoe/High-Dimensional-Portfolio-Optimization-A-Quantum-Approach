"""
Classical portfolio optimization models for comparison with quantum approaches.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Configure logging
logger = logging.getLogger(__name__)

class ClassicalPortfolioModel:
    """
    Portfolio optimization model using classical methods.
    Implements various optimization approaches including:
    - Equal Weight
    - Minimum Variance
    - Maximum Sharpe Ratio
    - Risk Parity
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the classical portfolio model.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary.
        """
        self.config = config
        self.classical_config = config['classical']
        self.optimization_config = config['optimization']
        
        # Optimization settings
        self.methods = self.classical_config['methods']
        self.solver = self.classical_config['solver']
        self.max_iterations = self.classical_config['max_iterations']
        self.risk_aversion = self.optimization_config['risk_aversion_range'][0]  # Default to min
        self.max_weight = self.optimization_config['constraints']['max_weight']
        self.min_weight = self.optimization_config['constraints']['min_weight']
        
        # Results storage
        self.portfolio_weights = None
        self.cluster_weights = None
        self.best_method = None
    
    def configure(self, **kwargs):
        """
        Configure the model parameters.
        
        Parameters:
        -----------
        **kwargs
            Keyword arguments for model parameters.
        """
        valid_params = {
            'risk_aversion', 'max_weight', 'min_weight', 'solver', 'max_iterations'
        }
        
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)
                logger.info(f"Set {key} = {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")
    
    def _equal_weight(self, n: int) -> np.ndarray:
        """
        Create an equal weight portfolio.
        
        Parameters:
        -----------
        n : int
            Number of assets (clusters).
            
        Returns:
        --------
        np.ndarray
            Array of weights.
        """
        return np.ones(n) / n
    
    def _min_variance(self, 
                     cluster_cov: np.ndarray, 
                     initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a minimum variance portfolio.
        
        Parameters:
        -----------
        cluster_cov : np.ndarray
            Covariance matrix for clusters.
        initial_guess : np.ndarray, optional
            Initial guess for weights. If None, use equal weights.
            
        Returns:
        --------
        np.ndarray
            Array of weights.
        """
        n = cluster_cov.shape[0]
        
        # Initial guess
        w0 = initial_guess if initial_guess is not None else self._equal_weight(n)
        
        # Define objective function (portfolio variance)
        def portfolio_variance(w, sigma):
            return w.T @ sigma @ w
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        
        # Optimize
        result = minimize(
            portfolio_variance,
            w0,
            args=(cluster_cov,),
            method=self.solver,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if not result.success:
            logger.warning(f"Minimum variance optimization did not converge: {result.message}")
        
        return result.x
    
    def _max_sharpe(self, 
                   cluster_means: np.ndarray, 
                   cluster_cov: np.ndarray,
                   initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a maximum Sharpe ratio portfolio.
        
        Parameters:
        -----------
        cluster_means : np.ndarray
            Mean returns for each cluster.
        cluster_cov : np.ndarray
            Covariance matrix for clusters.
        initial_guess : np.ndarray, optional
            Initial guess for weights. If None, use equal weights.
            
        Returns:
        --------
        np.ndarray
            Array of weights.
        """
        n = len(cluster_means)
        
        # Initial guess
        w0 = initial_guess if initial_guess is not None else self._equal_weight(n)
        
        # Define objective function (negative Sharpe ratio)
        def negative_sharpe(w, mu, sigma):
            portfolio_return = w.T @ mu
            portfolio_risk = np.sqrt(w.T @ sigma @ w)
            if portfolio_risk == 0:
                return -1e6  # Arbitrary large negative number if risk is zero
            return -portfolio_return / portfolio_risk
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        
        # Optimize
        result = minimize(
            negative_sharpe,
            w0,
            args=(cluster_means, cluster_cov),
            method=self.solver,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if not result.success:
            logger.warning(f"Maximum Sharpe optimization did not converge: {result.message}")
        
        return result.x
    
    def _risk_parity(self, 
                    cluster_cov: np.ndarray,
                    initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a risk parity portfolio.
        
        Parameters:
        -----------
        cluster_cov : np.ndarray
            Covariance matrix for clusters.
        initial_guess : np.ndarray, optional
            Initial guess for weights. If None, use equal weights.
            
        Returns:
        --------
        np.ndarray
            Array of weights.
        """
        n = cluster_cov.shape[0]
        
        # Initial guess
        w0 = initial_guess if initial_guess is not None else self._equal_weight(n)
        
        # Define risk contribution function
        def risk_contribution(w, sigma):
            portfolio_risk = np.sqrt(w.T @ sigma @ w)
            marginal_risk = sigma @ w
            risk_contribution = w * marginal_risk / portfolio_risk
            return risk_contribution
        
        # Define objective function (variance of risk contributions)
        def risk_parity_objective(w, sigma):
            rc = risk_contribution(w, sigma)
            target_risk = 1.0 / n  # Equal risk contribution
            return np.sum((rc - target_risk)**2)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            w0,
            args=(cluster_cov,),
            method=self.solver,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization did not converge: {result.message}")
        
        return result.x
    
    def _mean_variance(self, 
                      cluster_means: np.ndarray, 
                      cluster_cov: np.ndarray,
                      initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a mean-variance portfolio.
        
        Parameters:
        -----------
        cluster_means : np.ndarray
            Mean returns for each cluster.
        cluster_cov : np.ndarray
            Covariance matrix for clusters.
        initial_guess : np.ndarray, optional
            Initial guess for weights. If None, use equal weights.
            
        Returns:
        --------
        np.ndarray
            Array of weights.
        """
        n = len(cluster_means)
        
        # Initial guess
        w0 = initial_guess if initial_guess is not None else self._equal_weight(n)
        
        # Define objective function (mean-variance utility)
        def mean_variance_objective(w, mu, sigma, lambda_):
            portfolio_return = w.T @ mu
            portfolio_risk = w.T @ sigma @ w
            return -portfolio_return + lambda_ * portfolio_risk
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        
        # Optimize
        result = minimize(
            mean_variance_objective,
            w0,
            args=(cluster_means, cluster_cov, self.risk_aversion),
            method=self.solver,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if not result.success:
            logger.warning(f"Mean-variance optimization did not converge: {result.message}")
        
        return result.x
    
    def optimize(self, 
                cluster_means: pd.Series, 
                cluster_cov: pd.DataFrame,
                clusters: List[List[str]]) -> Dict[str, Any]:
        """
        Perform portfolio optimization using classical methods.
        Tries multiple methods and returns the best one based on Sharpe ratio.
        
        Parameters:
        -----------
        cluster_means : pd.Series
            Mean returns for each cluster.
        cluster_cov : pd.DataFrame
            Covariance matrix for clusters.
        clusters : list of lists
            List of asset clusters, where each cluster is a list of ticker symbols.
            
        Returns:
        --------
        dict
            Dictionary with optimization results.
        """
        try:
            # Convert to numpy arrays
            mu = cluster_means.values
            sigma = cluster_cov.values
            
            # Number of assets (clusters in this case)
            n = len(mu)
            
            # Run all optimization methods
            methods_results = {}
            
            # 1. Equal Weight
            if 'equal_weight' in self.methods:
                logger.info("Running equal weight optimization")
                weights_ew = self._equal_weight(n)
                sharpe_ew = self._calculate_sharpe(weights_ew, mu, sigma)
                methods_results['equal_weight'] = {
                    'weights': weights_ew,
                    'sharpe': sharpe_ew
                }
            
            # 2. Minimum Variance
            if 'min_variance' in self.methods:
                logger.info("Running minimum variance optimization")
                weights_mv = self._min_variance(sigma)
                sharpe_mv = self._calculate_sharpe(weights_mv, mu, sigma)
                methods_results['min_variance'] = {
                    'weights': weights_mv,
                    'sharpe': sharpe_mv
                }
            
            # 3. Maximum Sharpe
            if 'max_sharpe' in self.methods:
                logger.info("Running maximum Sharpe optimization")
                weights_ms = self._max_sharpe(mu, sigma)
                sharpe_ms = self._calculate_sharpe(weights_ms, mu, sigma)
                methods_results['max_sharpe'] = {
                    'weights': weights_ms,
                    'sharpe': sharpe_ms
                }
            
            # 4. Risk Parity
            if 'risk_parity' in self.methods:
                logger.info("Running risk parity optimization")
                weights_rp = self._risk_parity(sigma)
                sharpe_rp = self._calculate_sharpe(weights_rp, mu, sigma)
                methods_results['risk_parity'] = {
                    'weights': weights_rp,
                    'sharpe': sharpe_rp
                }
            
            # 5. Mean-Variance
            if 'mean_variance' in self.methods:
                logger.info("Running mean-variance optimization")
                weights_mv = self._mean_variance(mu, sigma)
                sharpe_mv = self._calculate_sharpe(weights_mv, mu, sigma)
                methods_results['mean_variance'] = {
                    'weights': weights_mv,
                    'sharpe': sharpe_mv
                }
            
            # Find best method
            best_method = max(methods_results.items(), key=lambda x: x[1]['sharpe'])
            self.best_method = best_method[0]
            best_weights = best_method[1]['weights']
            
            logger.info(f"Best method: {self.best_method} with Sharpe ratio {best_method[1]['sharpe']:.4f}")
            
            # Store cluster weights
            self.cluster_weights = {f'Cluster_{i+1}': best_weights[i] for i in range(n)}
            
            # Distribute weights within clusters (equal weight within each selected cluster)
            portfolio_weights = {}
            for i, cluster in enumerate(clusters):
                if best_weights[i] > 0:
                    cluster_weight = best_weights[i]
                    for asset in cluster:
                        portfolio_weights[asset] = cluster_weight / len(cluster)
                else:
                    for asset in cluster:
                        portfolio_weights[asset] = 0
            
            self.portfolio_weights = portfolio_weights
            
            # Calculate expected return and risk
            expected_return = best_weights.T @ mu * 252  # Annualized
            expected_risk = np.sqrt(best_weights.T @ sigma @ best_weights) * np.sqrt(252)  # Annualized
            sharpe_ratio = expected_return / expected_risk if expected_risk != 0 else 0
            
            # Store results
            result = {
                'portfolio_weights': portfolio_weights,
                'cluster_weights': self.cluster_weights,
                'expected_return': expected_return,
                'expected_risk': expected_risk,
                'sharpe_ratio': sharpe_ratio,
                'method': self.best_method
            }
            
            logger.info(f"Classical optimization completed: expected return = {expected_return:.4f}, "
                       f"expected risk = {expected_risk:.4f}, Sharpe ratio = {sharpe_ratio:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in classical optimization: {e}")
            raise
    
    def _calculate_sharpe(self, weights: np.ndarray, returns: np.ndarray, cov: np.ndarray) -> float:
        """
        Calculate Sharpe ratio for a portfolio.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights.
        returns : np.ndarray
            Expected returns.
        cov : np.ndarray
            Covariance matrix.
            
        Returns:
        --------
        float
            Sharpe ratio.
        """
        portfolio_return = weights.T @ returns * 252  # Annualized
        portfolio_risk = np.sqrt(weights.T @ cov @ weights) * np.sqrt(252)  # Annualized
        return portfolio_return / portfolio_risk if portfolio_risk != 0 else 0
    
    def evaluate_portfolio(self, 
                         returns: pd.DataFrame,
                         benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate portfolio performance.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
        benchmark_returns : pd.Series, optional
            Series of benchmark returns.
            
        Returns:
        --------
        dict
            Dictionary with performance metrics.
        """
        if self.portfolio_weights is None:
            logger.error("No portfolio weights available. Optimize first.")
            raise ValueError("No portfolio weights available. Optimize first.")
        
        try:
            # Create weight vector
            weights = np.zeros(len(returns.columns))
            for i, asset in enumerate(returns.columns):
                if asset in self.portfolio_weights:
                    weights[i] = self.portfolio_weights[asset]
            
            # Calculate portfolio returns
            portfolio_returns = returns.dot(weights)
            
            # Calculate basic performance metrics
            mean_return = portfolio_returns.mean() * 252  # Annualized
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = mean_return / volatility if volatility != 0 else 0
            
            # Calculate drawdowns
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdowns = (cum_returns / running_max) - 1
            max_drawdown = drawdowns.min()
            
            # Calculate skewness and kurtosis
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurt()
            
            # Calculate Sortino ratio (downside risk)
            neg_returns = portfolio_returns.copy()
            neg_returns[neg_returns > 0] = 0
            downside_deviation = neg_returns.std() * np.sqrt(252)
            sortino_ratio = mean_return / downside_deviation if downside_deviation != 0 else 0
            
            # Calculate VaR and CVaR
            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Calculate benchmark-relative metrics
            alpha, beta = 0, 0
            tracking_error = 0
            information_ratio = 0
            
            if benchmark_returns is not None:
                # Align benchmark returns with portfolio returns
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 0:
                    port_returns_aligned = portfolio_returns.loc[common_index]
                    bench_returns_aligned = benchmark_returns.loc[common_index]
                    
                    # Calculate beta and alpha
                    covariance = np.cov(port_returns_aligned, bench_returns_aligned)[0, 1]
                    benchmark_variance = np.var(bench_returns_aligned)
                    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                    alpha = (port_returns_aligned.mean() - beta * bench_returns_aligned.mean()) * 252
                    
                    # Calculate tracking error and information ratio
                    tracking_error = (port_returns_aligned - bench_returns_aligned).std() * np.sqrt(252)
                    excess_return = (port_returns_aligned.mean() - bench_returns_aligned.mean()) * 252
                    information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
            
            # Compile results
            metrics = {
                'expected_return': mean_return,
                'expected_risk': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'alpha': alpha,
                'beta': beta,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio
            }
            
            logger.info(f"Portfolio evaluation completed: Sharpe = {sharpe_ratio:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating portfolio: {e}")
            raise


# Example usage (when run as a script)
if __name__ == "__main__":
    import yaml
    import matplotlib.pyplot as plt
    from data.dataset import FinancialDataset
    from models.clustering import AssetClustering
    
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
    
    # Initialize clustering
    clustering = AssetClustering(config)
    
    # Perform clustering
    n_clusters = 5  # Example
    clusters = clustering.cluster_assets(returns, n_clusters=n_clusters)
    
    # Calculate cluster returns
    cluster_returns = clustering.calculate_cluster_returns(returns)
    
    # Calculate cluster statistics
    cluster_means = cluster_returns.mean()
    cluster_cov = cluster_returns.cov()
    
    # Initialize classical model
    model = ClassicalPortfolioModel(config)
    
    # Run optimization
    result = model.optimize(cluster_means, cluster_cov, clusters)
    
    # Print results
    print("\nPortfolio Weights:")
    for asset, weight in sorted(model.portfolio_weights.items()):
        print(f"  {asset}: {weight:.4f}")
    
    print(f"\nMethod: {model.best_method}")
    print(f"Expected Return: {result['expected_return']:.4f}")
    print(f"Expected Risk: {result['expected_risk']:.4f}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    
    # Plot asset weights by cluster
    plt.figure(figsize=(10, 6))
    
    # Group weights by cluster
    cluster_weights = {}
    for i, cluster in enumerate(clusters):
        cluster_name = f"Cluster {i+1}"
        cluster_weights[cluster_name] = [model.portfolio_weights.get(asset, 0) for asset in cluster]
    
    # Create bar plot
    plt.bar(range(len(cluster_weights)), 
            [sum(weights) for weights in cluster_weights.values()],
            tick_label=list(cluster_weights.keys()))
    
    plt.title('Portfolio Weights by Cluster')
    plt.ylabel('Weight')
    plt.xlabel('Cluster')
    plt.tight_layout()
    plt.show()
