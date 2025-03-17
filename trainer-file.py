"""
Training module for quantum portfolio optimization models.
Implements hyperparameter optimization and cross-validation.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import joblib
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import yaml
import os
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioTrainer:
    """
    Trainer class for portfolio optimization models.
    Handles hyperparameter tuning and cross-validation.
    """
    
    def __init__(self, config: Dict, dataset, clustering_model, portfolio_model):
        """
        Initialize the portfolio trainer.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary.
        dataset : FinancialDataset
            Dataset object.
        clustering_model : AssetClustering
            Clustering model object.
        portfolio_model : QuantumPortfolioModel
            Portfolio optimization model object.
        """
        self.config = config
        self.dataset = dataset
        self.clustering_model = clustering_model
        self.portfolio_model = portfolio_model
        
        self.training_config = config['training']
        self.hyperopt_method = self.training_config['hyperopt_method']
        self.hyperopt_iterations = self.training_config['hyperopt_iterations']
        self.cv_method = self.training_config['cv_method']
        self.n_splits = self.training_config['n_splits']
        self.test_size = self.training_config['test_size']
        self.scoring_metric = self.training_config['scoring_metric']
        
        # Results storage
        self.best_params = None
        self.cv_results = None
        self.hyperopt_results = None
        self.train_history = []
    
    def _setup_parameter_space(self) -> Dict:
        """
        Set up the hyperparameter search space.
        
        Returns:
        --------
        dict
            Hyperparameter search space.
        """
        try:
            # Define parameter space based on configuration
            if self.hyperopt_method == 'grid':
                # Grid search parameter space (discrete values)
                param_space = {
                    'risk_aversion': np.linspace(
                        self.config['optimization']['risk_aversion_range'][0],
                        self.config['optimization']['risk_aversion_range'][1],
                        5
                    ).tolist(),
                    'n_clusters': list(range(
                        self.config['clustering']['n_clusters_range'][0],
                        self.config['clustering']['n_clusters_range'][1] + 1
                    )),
                    'risk_measure': self.config['optimization']['risk_measures'],
                    'repetitions': list(range(
                        self.config['quantum']['repetitions_range'][0],
                        self.config['quantum']['repetitions_range'][1] + 1
                    )),
                    'ansatz_type': ['efficient_su2', 'two_local'] if 'custom' not in self.config['quantum']['ansatz_type'] else ['efficient_su2', 'two_local', 'custom']
                }
                
            elif self.hyperopt_method in ['random', 'bayesian']:
                # Hyperopt parameter space (for random and Bayesian optimization)
                param_space = {
                    'risk_aversion': hp.uniform(
                        'risk_aversion',
                        self.config['optimization']['risk_aversion_range'][0],
                        self.config['optimization']['risk_aversion_range'][1]
                    ),
                    'n_clusters': hp.quniform(
                        'n_clusters',
                        self.config['clustering']['n_clusters_range'][0],
                        self.config['clustering']['n_clusters_range'][1],
                        1
                    ),
                    'risk_measure': hp.choice(
                        'risk_measure',
                        self.config['optimization']['risk_measures']
                    ),
                    'repetitions': hp.quniform(
                        'repetitions',
                        self.config['quantum']['repetitions_range'][0],
                        self.config['quantum']['repetitions_range'][1],
                        1
                    ),
                    'ansatz_type': hp.choice(
                        'ansatz_type',
                        ['efficient_su2', 'two_local'] if 'custom' not in self.config['quantum']['ansatz_type'] else ['efficient_su2', 'two_local', 'custom']
                    )
                }
            else:
                raise ValueError(f"Unknown hyperparameter optimization method: {self.hyperopt_method}")
            
            return param_space
            
        except Exception as e:
            logger.error(f"Error setting up parameter space: {e}")
            raise
    
    def _evaluate_params(self, params: Dict, train_returns: pd.DataFrame, test_returns: pd.DataFrame) -> Dict:
        """
        Evaluate a set of hyperparameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary of hyperparameters.
        train_returns : pd.DataFrame
            Training returns data.
        test_returns : pd.DataFrame
            Test returns data.
            
        Returns:
        --------
        dict
            Evaluation results.
        """
        try:
            # Extract parameters
            n_clusters = int(params['n_clusters'])
            risk_aversion = float(params['risk_aversion'])
            risk_measure = params['risk_measure']
            repetitions = int(params['repetitions'])
            ansatz_type = params['ansatz_type']
            
            logger.info(f"Evaluating: n_clusters={n_clusters}, risk_aversion={risk_aversion:.2f}, "
                       f"risk_measure={risk_measure}, repetitions={repetitions}, ansatz_type={ansatz_type}")
            
            # Perform clustering
            clusters = self.clustering_model.cluster_assets(train_returns, n_clusters=n_clusters)
            
            # Calculate cluster returns
            cluster_returns = self.clustering_model.calculate_cluster_returns(train_returns)
            
            # Calculate cluster statistics
            cluster_means = cluster_returns.mean()
            cluster_cov = cluster_returns.cov()
            
            # Configure portfolio model
            self.portfolio_model.configure(
                risk_aversion=risk_aversion,
                risk_measure=risk_measure,
                repetitions=repetitions,
                ansatz_type=ansatz_type
            )
            
            # Run optimization
            opt_result = self.portfolio_model.optimize(cluster_means, cluster_cov, clusters)
            
            # Evaluate on test data
            benchmark_returns = self.dataset.get_benchmark_returns()
            # Align benchmark returns with test data
            if benchmark_returns is not None:
                test_benchmark_returns = benchmark_returns.loc[test_returns.index.intersection(benchmark_returns.index)]
            else:
                test_benchmark_returns = None
                
            metrics = self.portfolio_model.evaluate_portfolio(test_returns, test_benchmark_returns)
            
            # Track the iteration results
            result = {
                'params': params.copy(),
                'train_metrics': {
                    'expected_return': opt_result.get('expected_return', 0),
                    'expected_risk': opt_result.get('expected_risk', 0),
                    'sharpe_ratio': opt_result.get('sharpe_ratio', 0),
                },
                'test_metrics': metrics,
                'portfolio_weights': self.portfolio_model.portfolio_weights,
                'cluster_weights': self.portfolio_model.cluster_weights
            }
            
            # Log results
            score = metrics.get(self.scoring_metric, 0)
            logger.info(f"Evaluation completed: {self.scoring_metric}={score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return {
                'params': params,
                'error': str(e),
                'test_metrics': {self.scoring_metric: -np.inf}
            }
    
    def _objective_function(self, params: Dict) -> Dict:
        """
        Objective function for hyperparameter optimization.
        
        Parameters:
        -----------
        params : dict
            Dictionary of hyperparameters.
            
        Returns:
        --------
        dict
            Hyperopt result dictionary.
        """
        try:
            # Ensure integer parameters
            if 'n_clusters' in params:
                params['n_clusters'] = int(params['n_clusters'])
            if 'repetitions' in params:
                params['repetitions'] = int(params['repetitions'])
            
            # Get training and test data
            returns = self.dataset.returns
            
            # Create cross-validation splits
            splits = self.dataset.create_time_series_splits(
                data=returns,
                method=self.cv_method,
                n_splits=self.n_splits,
                test_size=self.test_size
            )
            
            if not splits:
                logger.error("No valid splits created")
                return {'loss': np.inf, 'status': STATUS_OK}
            
            # Cross-validation
            cv_scores = []
            
            for i, (train, test) in enumerate(splits):
                # Evaluate parameters on this split
                eval_result = self._evaluate_params(params, train, test)
                
                # Get score from test metrics
                score = eval_result['test_metrics'].get(self.scoring_metric, -np.inf)
                cv_scores.append(score)
                
                # Track iteration
                self.train_history.append(eval_result)
            
            # Calculate mean score across CV splits
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # For metrics where higher is better (like Sharpe ratio)
            # For metrics where lower is better (like drawdown), negate the score
            if self.scoring_metric in ['max_drawdown', 'expected_risk', 'var_95', 'cvar_95']:
                loss = mean_score  # Lower is better
            else:
                loss = -mean_score  # Higher is better
            
            logger.info(f"Cross-validation mean {self.scoring_metric}: {mean_score:.4f} (±{std_score:.4f})")
            
            return {
                'loss': loss,
                'mean_score': mean_score,
                'std_score': std_score,
                'params': params,
                'status': STATUS_OK
            }
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return {
                'loss': np.inf,
                'status': STATUS_OK,
                'error': str(e)
            }
    
    def train(self) -> Dict:
        """
        Train the portfolio model by finding optimal hyperparameters.
        
        Returns:
        --------
        dict
            Training results.
        """
        try:
            start_time = time.time()
            
            # Set up parameter space
            param_space = self._setup_parameter_space()
            
            if self.hyperopt_method == 'grid':
                # Grid search
                logger.info(f"Starting grid search with {len(ParameterGrid(param_space))} combinations")
                best_score = -np.inf
                best_params = None
                
                # Iterate over parameter grid
                for params in ParameterGrid(param_space):
                    # Evaluate parameters
                    result = self._objective_function(params)
                    
                    # Track best parameters
                    score = -result['loss']  # Convert back to original scale
                    if score > best_score:
                        best_score = score
                        best_params = params
                        logger.info(f"New best: {self.scoring_metric}={best_score:.4f}, params={best_params}")
                
                # Format results
                self.best_params = best_params
                self.hyperopt_results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'method': 'grid_search',
                    'n_iterations': len(ParameterGrid(param_space))
                }
                
            elif self.hyperopt_method in ['random', 'bayesian']:
                # Hyperopt for random or Bayesian optimization
                trials = Trials()
                
                if self.hyperopt_method == 'random':
                    logger.info(f"Starting random search with {self.hyperopt_iterations} iterations")
                    algo = 'random.suggest'
                else:
                    logger.info(f"Starting Bayesian optimization with {self.hyperopt_iterations} iterations")
                    algo = tpe.suggest
                
                # Run optimization
                best = fmin(
                    fn=self._objective_function,
                    space=param_space,
                    algo=algo,
                    max_evals=self.hyperopt_iterations,
                    trials=trials
                )
                
                # Process the best parameters
                best_params = {
                    'risk_aversion': best['risk_aversion'],
                    'n_clusters': int(best['n_clusters']),
                    'repetitions': int(best['repetitions']),
                    'risk_measure': self.config['optimization']['risk_measures'][best['risk_measure']] if isinstance(best['risk_measure'], int) else best['risk_measure'],
                    'ansatz_type': ['efficient_su2', 'two_local', 'custom'][best['ansatz_type']] if isinstance(best['ansatz_type'], int) else best['ansatz_type']
                }
                
                # Get the best score
                best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
                best_score = -trials.trials[best_trial_idx]['result']['loss']  # Convert back to original scale
                
                # Format results
                self.best_params = best_params
                self.hyperopt_results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'method': self.hyperopt_method,
                    'n_iterations': self.hyperopt_iterations,
                    'trials': trials
                }
                
                logger.info(f"Optimization completed: best {self.scoring_metric}={best_score:.4f}, params={best_params}")
                
            else:
                raise ValueError(f"Unknown hyperparameter optimization method: {self.hyperopt_method}")
            
            # Calculate total training time
            training_time = time.time() - start_time
            
            # Compile final results
            self.cv_results = {
                'best_params': self.best_params,
                'best_score': best_score,
                'training_time': training_time,
                'train_history': self.train_history,
                'scoring_metric': self.scoring_metric,
                'hyperopt_method': self.hyperopt_method
            }
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            return self.cv_results
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise
    
    def save_results(self, output_dir: str = 'results'):
        """
        Save training results.
        
        Parameters:
        -----------
        output_dir : str
            Output directory.
        """
        if self.cv_results is None:
            logger.error("No results to save. Run training first.")
            return
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save results
            results_file = os.path.join(output_dir, f'training_results_{timestamp}.joblib')
            joblib.dump(self.cv_results, results_file)
            
            # Save best parameters
            params_file = os.path.join(output_dir, f'best_params_{timestamp}.yaml')
            with open(params_file, 'w') as file:
                yaml.dump(self.best_params, file)
            
            logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def plot_training_results(self, figsize=(12, 8)) -> plt.Figure:
        """
        Plot training results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with training plots.
        """
        if not self.train_history:
            logger.error("No training history to plot. Run training first.")
            return None
        
        try:
            # Extract data from training history
            iterations = list(range(1, len(self.train_history) + 1))
            scores = [result['test_metrics'].get(self.scoring_metric, np.nan) for result in self.train_history]
            
            # Extract parameter values
            param_values = {}
            for param in ['risk_aversion', 'n_clusters', 'repetitions']:
                param_values[param] = [result['params'].get(param, np.nan) for result in self.train_history]
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            
            # Plot score over iterations
            axs[0, 0].plot(iterations, scores, 'o-', color='royalblue')
            axs[0, 0].set_title(f'{self.scoring_metric} Over Iterations')
            axs[0, 0].set_xlabel('Iteration')
            axs[0, 0].set_ylabel(self.scoring_metric)
            axs[0, 0].grid(True, alpha=0.3)
            
            # Plot parameters over iterations
            ax_mapping = {
                'risk_aversion': (0, 1),
                'n_clusters': (1, 0),
                'repetitions': (1, 1)
            }
            
            for param, (i, j) in ax_mapping.items():
                axs[i, j].plot(iterations, param_values[param], 'o-', color='firebrick')
                axs[i, j].set_title(f'{param} Over Iterations')
                axs[i, j].set_xlabel('Iteration')
                axs[i, j].set_ylabel(param)
                axs[i, j].grid(True, alpha=0.3)
            
            # Add best parameters as text
            if self.best_params:
                best_params_text = "\n".join([f"{k}: {v}" for k, v in self.best_params.items()])
                fig.text(0.5, 0.01, f"Best Parameters:\n{best_params_text}", 
                        ha='center', va='bottom', bbox=dict(boxstyle='round', alpha=0.1))
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting training results: {e}")
            raise
    
    def final_model(self) -> Dict:
        """
        Train a final model using the best parameters.
        
        Returns:
        --------
        dict
            Final model results.
        """
        if self.best_params is None:
            logger.error("No best parameters available. Run training first.")
            return None
        
        try:
            # Get full dataset
            returns = self.dataset.returns
            
            # Configure clustering with best parameters
            n_clusters = self.best_params['n_clusters']
            
            # Perform clustering
            clusters = self.clustering_model.cluster_assets(returns, n_clusters=n_clusters)
            
            # Calculate cluster returns
            cluster_returns = self.clustering_model.calculate_cluster_returns(returns)
            
            # Calculate cluster statistics
            cluster_means = cluster_returns.mean()
            cluster_cov = cluster_returns.cov()
            
            # Configure portfolio model with best parameters
            self.portfolio_model.configure(**{k: v for k, v in self.best_params.items() if k != 'n_clusters'})
            
            # Run optimization
            opt_result = self.portfolio_model.optimize(cluster_means, cluster_cov, clusters)
            
            # Evaluate on full dataset
            benchmark_returns = self.dataset.get_benchmark_returns()
            metrics = self.portfolio_model.evaluate_portfolio(returns, benchmark_returns)
            
            # Compile results
            final_results = {
                'best_params': self.best_params,
                'portfolio_weights': self.portfolio_model.portfolio_weights,
                'cluster_weights': self.portfolio_model.cluster_weights,
                'clusters': clusters,
                'metrics': metrics
            }
            
            logger.info(f"Final model trained: {self.scoring_metric}={metrics.get(self.scoring_metric, 0):.4f}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error training final model: {e}")
            raise


# Example usage (when run as a script)
if __name__ == "__main__":
    import yaml
    from data.dataset import FinancialDataset
    from models.clustering import AssetClustering
    from models.quantum_model import QuantumPortfolioModel
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open('../config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize components
    dataset = FinancialDataset(config_path='../config/default.yaml')
    clustering = AssetClustering(config)
    portfolio_model = QuantumPortfolioModel(config)
    
    # Fetch data
    prices = dataset.fetch_data()
    returns = dataset.calculate_returns()
    
    # Initialize trainer
    trainer = PortfolioTrainer(config, dataset, clustering, portfolio_model)
    
    # Run training
    results = trainer.train()
    
    # Save results
    trainer.save_results()
    
    # Plot training results
    fig = trainer.plot_training_results()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    
    # Train final model
    final_results = trainer.final_model()
    
    # Print final weights
    print("\nFinal Portfolio Weights:")
    for asset, weight in sorted(final_results['portfolio_weights'].items()):
        print(f"  {asset}: {weight:.4f}")
    
    print("\nFinal Performance Metrics:")
    for metric, value in final_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
