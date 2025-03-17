"""
Quantum portfolio optimization model using Qiskit.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from qiskit import Aer, execute
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_optimization.applications import QuadraticProgram
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver, VQE
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import algorithm_globals

# Configure logging
logger = logging.getLogger(__name__)

class QuantumPortfolioModel:
    """
    Portfolio optimization model using quantum computing techniques.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the quantum portfolio model.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary.
        """
        self.config = config
        self.quantum_config = config['quantum']
        self.optimization_config = config['optimization']
        
        # Quantum settings
        self.backend_type = self.quantum_config['backend_type']
        self.simulator_name = self.quantum_config['simulator']
        self.ansatz_type = self.quantum_config['ansatz_type']
        self.repetitions = self.quantum_config['repetitions_range'][0]  # Default to min
        self.optimization_level = self.quantum_config['optimization_level']
        self.shots = self.quantum_config['shots']
        
        # Optimization settings
        self.risk_aversion = self.optimization_config['risk_aversion_range'][0]  # Default to min
        self.risk_measure = self.optimization_config['risk_measures'][0]  # Default to first
        self.max_weight = self.optimization_config['constraints']['max_weight']
        self.min_weight = self.optimization_config['constraints']['min_weight']
        
        # Results storage
        self.results = {}
        self.portfolio_weights = None
        self.cluster_weights = None
        
        # Set random seed for reproducibility
        algorithm_globals.random_seed = config['training']['random_state']
    
    def configure(self, **kwargs):
        """
        Configure the model parameters.
        
        Parameters:
        -----------
        **kwargs
            Keyword arguments for model parameters.
        """
        valid_params = {
            'risk_aversion', 'risk_measure', 'backend_type', 'ansatz_type',
            'repetitions', 'optimization_level', 'shots', 'max_weight', 'min_weight'
        }
        
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)
                logger.info(f"Set {key} = {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")
    
    def _create_quadratic_program(self, cluster_means: np.ndarray, cluster_cov: np.ndarray) -> QuadraticProgram:
        """
        Create a quadratic program for portfolio optimization.
        
        Parameters:
        -----------
        cluster_means : np.ndarray
            Mean returns for each cluster.
        cluster_cov : np.ndarray
            Covariance matrix for clusters.
            
        Returns:
        --------
        QuadraticProgram
            Qiskit quadratic program.
        """
        try:
            # Number of assets (clusters in this case)
            n = len(cluster_means)
            
            # Create quadratic program
            qp = QuadraticProgram()
            
            # Add variables (continuous, bounded between min_weight and max_weight)
            for i in range(n):
                qp.continuous_var(name=f'x{i}', lowerbound=self.min_weight, upperbound=self.max_weight)
            
            # Set objective function based on selected risk measure
            if self.risk_measure == 'variance':
                # Objective: maximize return - lambda * risk
                # Equivalent to: minimize -return + lambda * risk
                
                # Linear terms (negative returns)
                linear = {f'x{i}': -cluster_means[i] for i in range(n)}
                
                # Quadratic terms (risk)
                quadratic = {}
                for i in range(n):
                    for j in range(n):
                        if cluster_cov[i, j] != 0:
                            quadratic[(f'x{i}', f'x{j}')] = self.risk_aversion * cluster_cov[i, j]
                
                qp.minimize(linear=linear, quadratic=quadratic)
                
            elif self.risk_measure == 'cvar':
                # This is a simplified CVaR implementation
                # For a complete CVaR, we would need to add auxiliary variables and constraints
                # Here we use a sample approximation of CVaR
                
                # Generate scenarios based on historical returns
                # TODO: This is a placeholder for proper CVaR implementation
                logger.warning("Using simplified CVaR approximation")
                
                # Linear terms (negative returns)
                linear = {f'x{i}': -cluster_means[i] for i in range(n)}
                
                # Use covariance as a proxy for tail risk
                quadratic = {}
                for i in range(n):
                    for j in range(n):
                        if cluster_cov[i, j] != 0:
                            # Increase weight for tail risk
                            quadratic[(f'x{i}', f'x{j}')] = 1.5 * self.risk_aversion * cluster_cov[i, j]
                
                qp.minimize(linear=linear, quadratic=quadratic)
                
            elif self.risk_measure == 'drawdown':
                # This is a simplified drawdown implementation
                # For a complete drawdown constraint, we would need a more complex model
                logger.warning("Using simplified drawdown approximation")
                
                # Linear terms (negative returns)
                linear = {f'x{i}': -cluster_means[i] for i in range(n)}
                
                # Use covariance as a proxy, but penalize negative skewness assets more
                # Assumption: assets with higher volatility have higher drawdown potential
                quadratic = {}
                for i in range(n):
                    for j in range(n):
                        if cluster_cov[i, j] != 0:
                            quadratic[(f'x{i}', f'x{j}')] = self.risk_aversion * cluster_cov[i, j]
                
                qp.minimize(linear=linear, quadratic=quadratic)
                
            else:
                raise ValueError(f"Unknown risk measure: {self.risk_measure}")
            
            # Add constraint: sum of weights equals 1
            qp.linear_constraint(linear={f'x{i}': 1 for i in range(n)}, sense='==', rhs=1)
            
            return qp
            
        except Exception as e:
            logger.error(f"Error creating quadratic program: {e}")
            raise
    
    def _setup_quantum_solver(self, n_qubits: int):
        """
        Set up the quantum solver.
        
        Parameters:
        -----------
        n_qubits : int
            Number of qubits for the quantum circuit.
            
        Returns:
        --------
        MinimumEigenOptimizer
            Quantum optimizer.
        """
        try:
            # Set up the quantum backend
            if self.backend_type == 'simulator':
                backend = Aer.get_backend(self.simulator_name)
            else:
                # This would connect to a real quantum device
                # For now, just use simulator
                logger.warning("Real quantum hardware not available, using simulator")
                backend = Aer.get_backend(self.simulator_name)
            
            # Choose the right ansatz
            if self.ansatz_type == 'efficient_su2':
                ansatz = EfficientSU2(n_qubits, reps=self.repetitions, entanglement='linear')
            elif self.ansatz_type == 'two_local':
                ansatz = TwoLocal(n_qubits, 'ry', 'cz', reps=self.repetitions, entanglement='linear')
            elif self.ansatz_type == 'custom':
                # A more sophisticated custom ansatz could be implemented here
                ansatz = TwoLocal(n_qubits, ['ry', 'rz'], ['cz', 'cx'], reps=self.repetitions, entanglement='full')
            else:
                raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")
            
            # Choose algorithm
            if self.backend_type == 'simulator' and n_qubits <= 16:
                # For small problems, exact classical solver is more efficient
                solver = NumPyMinimumEigensolver()
                logger.info(f"Using NumPyMinimumEigensolver for {n_qubits} qubits")
            else:
                # Use QAOA for quantum optimization
                optimizer = COBYLA(maxiter=500)
                qaoa = QAOA(optimizer=optimizer, quantum_instance=backend, reps=self.repetitions)
                solver = qaoa
                logger.info(f"Using QAOA with {self.repetitions} repetitions")
            
            # Create optimizer
            quantum_optimizer = MinimumEigenOptimizer(solver)
            
            return quantum_optimizer
            
        except Exception as e:
            logger.error(f"Error setting up quantum solver: {e}")
            raise
    
    def optimize(self, 
                cluster_means: pd.Series, 
                cluster_cov: pd.DataFrame,
                clusters: List[List[str]]) -> Dict[str, Any]:
        """
        Perform portfolio optimization using quantum computing techniques.
        
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
            
            # Create quadratic program
            qp = self._create_quadratic_program(mu, sigma)
            
            # Convert to QUBO for quantum solver
            qp2qubo = QuadraticProgramToQubo()
            qubo = qp2qubo.convert(qp)
            
            # Set up the quantum solver
            optimizer = self._setup_quantum_solver(n_qubits=n)
            
            # Solve the problem
            result = optimizer.solve(qubo)
            
            # Extract the solution
            x = np.array([result.x[f'x{i}'] for i in range(n)])
            
            # Store cluster weights
            self.cluster_weights = {f'Cluster_{i+1}': x[i] for i in range(n)}
            
            # Distribute weights within clusters (equal weight within each selected cluster)
            portfolio_weights = {}
            for i, cluster in enumerate(clusters):
                if x[i] > 0:
                    cluster_weight = x[i]
                    for asset in cluster:
                        portfolio_weights[asset] = cluster_weight / len(cluster)
                else:
                    for asset in cluster:
                        portfolio_weights[asset] = 0
            
            self.portfolio_weights = portfolio_weights
            
            # Store results
            self.results = {
                'portfolio_weights': portfolio_weights,
                'cluster_weights': self.cluster_weights,
                'objective_value': result.fval,
                'optimization_status': result.status,
                'method': f"Quantum-{self.ansatz_type}-{self.risk_measure}"
            }
            
            logger.info(f"Optimization completed: objective value = {result.fval}")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in quantum optimization: {e}")
            raise
    
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
    from models.clustering import AssetClustering
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
    
    # Calculate returns
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
    
    # Initialize quantum model
    model = QuantumPortfolioModel(config)
    
    # Configure model
    model.configure(risk_aversion=2.0, repetitions=2)
    
    # Run optimization
    result = model.optimize(cluster_means, cluster_cov, clusters)
    
    # Evaluate portfolio
    benchmark_returns = dataset.get_benchmark_returns()
    metrics = model.evaluate_portfolio(returns, benchmark_returns)
    
    # Print results
    print("\nPortfolio Weights:")
    for asset, weight in sorted(model.portfolio_weights.items()):
        print(f"  {asset}: {weight:.4f}")
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
