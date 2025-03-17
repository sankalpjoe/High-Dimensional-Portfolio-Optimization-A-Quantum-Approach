"""
Main execution script for quantum portfolio optimization.
"""

import argparse
import logging
import os
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd

from data.dataset import FinancialDataset
from models.clustering import AssetClustering
from models.quantum_model import QuantumPortfolioModel
from models.classical_model import ClassicalPortfolioModel
from training.trainer import PortfolioTrainer
from evaluation.visualization import PortfolioVisualizer

def setup_logging(config):
    """Set up logging configuration."""
    log_level = getattr(logging, config['logging']['level'])
    log_file = config['logging']['log_file']
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    
    return logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantum Portfolio Optimization')
    
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'optimize', 'evaluate'], default='optimize',
                       help='Operation mode')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--tickers', type=str, nargs='+',
                       help='List of ticker symbols (overrides config)')
    parser.add_argument('--start_date', type=str,
                       help='Start date in YYYY-MM-DD format (overrides config)')
    parser.add_argument('--end_date', type=str,
                       help='End date in YYYY-MM-DD format (overrides config)')
    parser.add_argument('--n_clusters', type=int,
                       help='Number of clusters (overrides config)')
    parser.add_argument('--risk_aversion', type=float,
                       help='Risk aversion parameter (overrides config)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save plots')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with classical optimization')
    parser.add_argument('--load_params', type=str,
                       help='Path to saved parameters file')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")

def save_results(results, output_dir, prefix='portfolio'):
    """Save results to files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save weights to CSV
    weights_file = os.path.join(output_dir, f'{prefix}_weights_{timestamp}.csv')
    weights_df = pd.DataFrame(results['portfolio_weights'].items(), columns=['Asset', 'Weight'])
    weights_df.to_csv(weights_file, index=False)
    
    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, f'{prefix}_metrics_{timestamp}.csv')
    metrics_df = pd.DataFrame(results['metrics'].items(), columns=['Metric', 'Value'])
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save full results to YAML
    results_file = os.path.join(output_dir, f'{prefix}_results_{timestamp}.yaml')
    with open(results_file, 'w') as file:
        # Convert numpy values to Python types for YAML serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, np.number) else v 
                                          for k, v in value.items()}
            else:
                serializable_results[key] = value
        
        yaml.dump(serializable_results, file)
    
    print(f"Results saved to {output_dir}")
    return {
        'weights_file': weights_file,
        'metrics_file': metrics_file,
        'results_file': results_file
    }

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting quantum portfolio optimization in {args.mode} mode")
    
    # Override config with command line arguments
    if args.tickers:
        config['data']['default_tickers'] = args.tickers
    if args.start_date:
        config['data']['default_start_date'] = args.start_date
    if args.end_date:
        config['data']['default_end_date'] = args.end_date
    if args.n_clusters:
        config['clustering']['n_clusters_range'] = [args.n_clusters, args.n_clusters]
    if args.risk_aversion:
        config['optimization']['risk_aversion_range'] = [args.risk_aversion, args.risk_aversion]
    
    # Initialize components
    logger.info("Initializing components")
    dataset = FinancialDataset(config_path=args.config)
    clustering = AssetClustering(config)
    quantum_model = QuantumPortfolioModel(config)
    
    # Load parameters if specified
    if args.load_params:
        try:
            with open(args.load_params, 'r') as file:
                params = yaml.safe_load(file)
            if 'n_clusters' in params:
                config['clustering']['n_clusters_range'] = [params['n_clusters'], params['n_clusters']]
            quantum_model.configure(**{k: v for k, v in params.items() if k != 'n_clusters'})
            logger.info(f"Loaded parameters from {args.load_params}")
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
    
    # Fetch data
    logger.info("Fetching data")
    prices = dataset.fetch_data()
    returns = dataset.calculate_returns()
    
    # Operating modes
    if args.mode == 'train':
        # Training mode: find optimal hyperparameters
        logger.info("Running in training mode")
        
        # Initialize trainer
        trainer = PortfolioTrainer(config, dataset, clustering, quantum_model)
        
        # Run training
        results = trainer.train()
        
        # Save results
        trainer.save_results(output_dir=args.output_dir)
        
        # Plot training results
        if args.plot:
            logger.info("Generating training plots")
            fig = trainer.plot_training_results()
            
            # Save plot
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = os.path.join(args.output_dir, f'training_results_{timestamp}.png')
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            logger.info(f"Plot saved to {plot_file}")
        
        # Train final model
        logger.info("Training final model with best parameters")
        final_results = trainer.final_model()
        
        # Save final model results
        save_results(final_results, args.output_dir, prefix='final_model')
        
        # Generate portfolio visualization
        if args.plot:
            logger.info("Generating portfolio visualization")
            visualizer = PortfolioVisualizer(config)
            
            # Plot portfolio weights
            fig_weights = visualizer.plot_portfolio_weights(final_results['portfolio_weights'])
            
            # Plot performance metrics
            fig_metrics = visualizer.plot_performance_metrics(final_results['metrics'])
            
            # Save plots
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            weights_file = os.path.join(args.output_dir, f'portfolio_weights_{timestamp}.png')
            metrics_file = os.path.join(args.output_dir, f'performance_metrics_{timestamp}.png')
            
            fig_weights.savefig(weights_file, dpi=300, bbox_inches='tight')
            fig_metrics.savefig(metrics_file, dpi=300, bbox_inches='tight')
            
            logger.info(f"Plots saved to {args.output_dir}")
        
    elif args.mode == 'optimize':
        # Optimization mode: run a single optimization
        logger.info("Running in optimization mode")
        
        # Set the number of clusters
        n_clusters = config['clustering']['n_clusters_range'][0]
        
        # Perform clustering
        logger.info(f"Performing clustering with {n_clusters} clusters")
        clusters = clustering.cluster_assets(returns, n_clusters=n_clusters)
        
        # Calculate cluster returns
        cluster_returns = clustering.calculate_cluster_returns(returns)
        
        # Calculate cluster statistics
        cluster_means = cluster_returns.mean()
        cluster_cov = cluster_returns.cov()
        
        # Run quantum optimization
        logger.info("Running quantum optimization")
        quantum_result = quantum_model.optimize(cluster_means, cluster_cov, clusters)
        
        # Get benchmark returns
        benchmark_returns = dataset.get_benchmark_returns()
        
        # Evaluate quantum portfolio
        quantum_metrics = quantum_model.evaluate_portfolio(returns, benchmark_returns)
        quantum_result['metrics'] = quantum_metrics
        
        # Save quantum results
        save_results(quantum_result, args.output_dir, prefix='quantum')
        
        # Run classical optimization for comparison
        if args.compare:
            logger.info("Running classical optimization for comparison")
            classical_model = ClassicalPortfolioModel(config)
            classical_result = classical_model.optimize(cluster_means, cluster_cov, clusters)
            
            # Evaluate classical portfolio
            classical_metrics = classical_model.evaluate_portfolio(returns, benchmark_returns)
            classical_result['metrics'] = classical_metrics
            
            # Save classical results
            save_results(classical_result, args.output_dir, prefix='classical')
            
            # Compare results
            logger.info("\nPerformance Comparison:")
            for metric in ['sharpe_ratio', 'expected_return', 'expected_risk', 'max_drawdown']:
                quantum_value = quantum_metrics.get(metric, 0)
                classical_value = classical_metrics.get(metric, 0)
                diff_pct = ((quantum_value / classical_value) - 1) * 100 if classical_value != 0 else 0
                logger.info(f"  {metric}: Quantum={quantum_value:.4f}, Classical={classical_value:.4f}, Diff={diff_pct:.2f}%")
        
        # Generate visualizations
        if args.plot:
            logger.info("Generating portfolio visualizations")
            visualizer = PortfolioVisualizer(config)
            
            # Plot cluster map
            fig_clusters = clustering.plot_hierarchical_clustering(returns)
            
            # Plot portfolio weights
            fig_weights = visualizer.plot_portfolio_weights(quantum_model.portfolio_weights)
            
            # Plot performance metrics
            fig_metrics = visualizer.plot_performance_metrics(quantum_metrics)
            
            # Plot efficient frontier if comparing with classical
            if args.compare:
                fig_frontier = visualizer.plot_efficient_frontier(
                    returns,
                    highlight_portfolios={
                        'Quantum': {
                            'portfolio_weights': quantum_model.portfolio_weights,
                            'expected_return': quantum_metrics['expected_return'],
                            'expected_risk': quantum_metrics['expected_risk']
                        },
                        'Classical': {
                            'portfolio_weights': classical_model.portfolio_weights,
                            'expected_return': classical_metrics['expected_return'],
                            'expected_risk': classical_metrics['expected_risk']
                        }
                    }
                )
            
            # Save plots
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            clusters_file = os.path.join(args.output_dir, f'asset_clusters_{timestamp}.png')
            weights_file = os.path.join(args.output_dir, f'portfolio_weights_{timestamp}.png')
            metrics_file = os.path.join(args.output_dir, f'performance_metrics_{timestamp}.png')
            
            fig_clusters.savefig(clusters_file, dpi=300, bbox_inches='tight')
            fig_weights.savefig(weights_file, dpi=300, bbox_inches='tight')
            fig_metrics.savefig(metrics_file, dpi=300, bbox_inches='tight')
            
            if args.compare:
                frontier_file = os.path.join(args.output_dir, f'efficient_frontier_{timestamp}.png')
                fig_frontier.savefig(frontier_file, dpi=300, bbox_inches='tight')
            
            logger.info(f"Plots saved to {args.output_dir}")
        
    elif args.mode == 'evaluate':
        # Evaluation mode: evaluate an existing portfolio
        logger.info("Running in evaluation mode")
        
        if not args.load_params:
            logger.error("No parameters file specified. Use --load_params to specify a file.")
            return
        
        # Load parameters
        with open(args.load_params, 'r') as file:
            params = yaml.safe_load(file)
        
        # Set the number of clusters
        n_clusters = params.get('n_clusters', config['clustering']['n_clusters_range'][0])
        
        # Perform clustering
        logger.info(f"Performing clustering with {n_clusters} clusters")
        clusters = clustering.cluster_assets(returns, n_clusters=n_clusters)
        
        # Calculate cluster returns
        cluster_returns = clustering.calculate_cluster_returns(returns)
        
        # Calculate cluster statistics
        cluster_means = cluster_returns.mean()
        cluster_cov = cluster_returns.cov()
        
        # Configure quantum model
        quantum_model.configure(**{k: v for k, v in params.items() if k != 'n_clusters'})
        
        # Run optimization
        logger.info("Running quantum optimization with loaded parameters")
        quantum_result = quantum_model.optimize(cluster_means, cluster_cov, clusters)
        
        # Get benchmark returns
        benchmark_returns = dataset.get_benchmark_returns()
        
        # Evaluate quantum portfolio
        quantum_metrics = quantum_model.evaluate_portfolio(returns, benchmark_returns)
        quantum_result['metrics'] = quantum_metrics
        
        # Save results
        save_results(quantum_result, args.output_dir, prefix='evaluation')
        
        # Generate visualizations
        if args.plot:
            logger.info("Generating evaluation visualizations")
            visualizer = PortfolioVisualizer(config)
            
            # Plot portfolio weights
            fig_weights = visualizer.plot_portfolio_weights(quantum_model.portfolio_weights)
            
            # Plot performance metrics
            fig_metrics = visualizer.plot_performance_metrics(quantum_metrics)
            
            # Plot performance over time
            portfolio_returns = returns.dot(np.array(
                [quantum_model.portfolio_weights.get(asset, 0) for asset in returns.columns]
            ))
            
            fig_performance = visualizer.plot_performance_over_time(
                portfolio_returns,
                benchmark_returns
            )
            
            # Save plots
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            weights_file = os.path.join(args.output_dir, f'eval_weights_{timestamp}.png')
            metrics_file = os.path.join(args.output_dir, f'eval_metrics_{timestamp}.png')
            perf_file = os.path.join(args.output_dir, f'eval_performance_{timestamp}.png')
            
            fig_weights.savefig(weights_file, dpi=300, bbox_inches='tight')
            fig_metrics.savefig(metrics_file, dpi=300, bbox_inches='tight')
            fig_performance.savefig(perf_file, dpi=300, bbox_inches='tight')
            
            logger.info(f"Plots saved to {args.output_dir}")
    
    logger.info("Quantum portfolio optimization completed successfully")

if __name__ == "__main__":
    main()
