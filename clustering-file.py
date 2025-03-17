"""
Asset clustering methods for dimensionality reduction in portfolio optimization.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

class AssetClustering:
    """
    Class for clustering financial assets based on return patterns.
    Implements various clustering algorithms and metrics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the asset clustering.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary.
        """
        self.config = config['clustering']
        self.method = self.config['method']
        self.n_clusters_range = self.config['n_clusters_range']
        self.distance_metric = self.config['distance_metric']
        self.linkage_method = self.config['linkage_method']
        
        # Results storage
        self.clusters = None
        self.labels = None
        self.cluster_returns = None
        self.cluster_performances = None
        self.silhouette_scores = {}
        self.ch_scores = {}
    
    def calculate_distance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate distance matrix between assets.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
            
        Returns:
        --------
        np.ndarray
            Distance matrix.
        """
        try:
            if self.distance_metric == 'correlation':
                # Calculate correlation matrix
                corr_matrix = returns.corr()
                
                # Convert correlation to distance (higher correlation = lower distance)
                dist_matrix = 1 - np.abs(corr_matrix)
                
            elif self.distance_metric == 'euclidean':
                # Transpose so that each row is an asset (feature) instead of a time point
                returns_T = returns.T
                
                # Calculate Euclidean distance
                dist_matrix = pdist(returns_T, metric='euclidean')
                dist_matrix = squareform(dist_matrix)
                
                # Normalize distances
                dist_matrix = dist_matrix / np.max(dist_matrix)
                
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
            return dist_matrix
            
        except Exception as e:
            logger.error(f"Error calculating distance matrix: {e}")
            raise
    
    def find_optimal_clusters(self, returns: pd.DataFrame) -> int:
        """
        Find the optimal number of clusters based on silhouette and
        Calinski-Harabasz scores.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
            
        Returns:
        --------
        int
            Optimal number of clusters.
        """
        try:
            # Prepare for clustering
            if self.method == 'hierarchical':
                # Calculate distance matrix
                dist_matrix = self.calculate_distance_matrix(returns)
                condensed_dist = squareform(dist_matrix)
                
                # Compute linkage matrix
                Z = linkage(condensed_dist, method=self.linkage_method)
                
                for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
                    # Get cluster labels
                    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-based indexing
                    
                    # Calculate silhouette score
                    if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters
                        silhouette = silhouette_score(dist_matrix, labels, metric='precomputed')
                        self.silhouette_scores[n_clusters] = silhouette
                        
                        # Calculate Calinski-Harabasz score (requires original data, not distance)
                        ch_score = calinski_harabasz_score(returns.T, labels)
                        self.ch_scores[n_clusters] = ch_score
                        
                        logger.info(f"n_clusters={n_clusters}, silhouette={silhouette:.4f}, CH={ch_score:.4f}")
            
            elif self.method == 'kmeans':
                for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
                    # Fit KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(returns.T)  # Transpose for assets as samples
                    
                    # Calculate silhouette score
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(returns.T, labels)
                        self.silhouette_scores[n_clusters] = silhouette
                        
                        # Calculate Calinski-Harabasz score
                        ch_score = calinski_harabasz_score(returns.T, labels)
                        self.ch_scores[n_clusters] = ch_score
                        
                        logger.info(f"n_clusters={n_clusters}, silhouette={silhouette:.4f}, CH={ch_score:.4f}")
            
            else:
                raise ValueError(f"Unknown clustering method: {self.method}")
            
            # Determine optimal number of clusters
            # Normalize scores
            sil_scores = np.array(list(self.silhouette_scores.values()))
            ch_scores = np.array(list(self.ch_scores.values()))
            
            norm_sil = (sil_scores - sil_scores.min()) / (sil_scores.max() - sil_scores.min() + 1e-10)
            norm_ch = (ch_scores - ch_scores.min()) / (ch_scores.max() - ch_scores.min() + 1e-10)
            
            # Combine scores (equal weight)
            combined_scores = 0.5 * norm_sil + 0.5 * norm_ch
            
            # Find optimal
            optimal_idx = np.argmax(combined_scores)
            optimal_n_clusters = list(self.silhouette_scores.keys())[optimal_idx]
            
            logger.info(f"Optimal number of clusters: {optimal_n_clusters}")
            return optimal_n_clusters
            
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {e}")
            raise
    
    def cluster_assets(self, 
                        returns: pd.DataFrame, 
                        n_clusters: Optional[int] = None, 
                        method: Optional[str] = None) -> List[List[str]]:
        """
        Perform clustering on assets based on return patterns.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
        n_clusters : int, optional
            Number of clusters. If None, find optimal number.
        method : str, optional
            Clustering method. If None, use the method from config.
            
        Returns:
        --------
        list of lists
            List of asset clusters, where each cluster is a list of ticker symbols.
        """
        method = method or self.method
        
        try:
            # Find optimal number of clusters if not provided
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters(returns)
            
            # Perform clustering
            if method == 'hierarchical':
                # Calculate distance matrix
                dist_matrix = self.calculate_distance_matrix(returns)
                condensed_dist = squareform(dist_matrix)
                
                # Perform hierarchical clustering
                Z = linkage(condensed_dist, method=self.linkage_method)
                labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-based indexing
                
            elif method == 'kmeans':
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(returns.T)  # Transpose for assets as samples
                
            elif method == 'dbscan':
                # Perform DBSCAN clustering
                # Note: DBSCAN doesn't take n_clusters as a parameter, but will find clusters based on density
                dbscan = DBSCAN(eps=0.3, min_samples=3)
                labels = dbscan.fit_predict(returns.T)
                
                # Handle outliers (label -1)
                if -1 in labels:
                    # Create a new label for each outlier
                    max_label = labels.max()
                    outliers = np.where(labels == -1)[0]
                    for i, idx in enumerate(outliers):
                        labels[idx] = max_label + 1 + i
                
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Store labels
            self.labels = labels
            
            # Group assets into clusters
            clusters = []
            for i in range(len(np.unique(labels))):
                cluster_members = [returns.columns[j] for j in range(len(labels)) if labels[j] == i]
                if cluster_members:
                    clusters.append(cluster_members)
            
            self.clusters = clusters
            
            logger.info(f"Clustered assets into {len(clusters)} groups")
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering assets: {e}")
            raise
    
    def calculate_cluster_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns for each cluster (equal-weighted within clusters).
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of cluster returns.
        """
        if self.clusters is None:
            logger.error("No clusters available. Perform clustering first.")
            raise ValueError("No clusters available. Perform clustering first.")
        
        try:
            # Calculate cluster-level returns (equal weighted within clusters)
            cluster_returns = pd.DataFrame(index=returns.index)
            
            for i, cluster in enumerate(self.clusters):
                weights = np.ones(len(cluster)) / len(cluster)
                cluster_ret = returns[cluster].dot(weights)
                cluster_returns[f'Cluster_{i+1}'] = cluster_ret
            
            self.cluster_returns = cluster_returns
            
            logger.info(f"Calculated returns for {cluster_returns.shape[1]} clusters")
            return cluster_returns
            
        except Exception as e:
            logger.error(f"Error calculating cluster returns: {e}")
            raise
    
    def analyze_clusters(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the characteristics of each cluster.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
            
        Returns:
        --------
        dict
            Dictionary with cluster analysis.
        """
        if self.clusters is None:
            logger.error("No clusters available. Perform clustering first.")
            raise ValueError("No clusters available. Perform clustering first.")
        
        try:
            analysis = {}
            
            # Calculate cluster-level returns if not already done
            if self.cluster_returns is None:
                self.calculate_cluster_returns(returns)
            
            # Calculate performance metrics for each cluster
            performance = {}
            for column in self.cluster_returns.columns:
                cluster_idx = int(column.split('_')[1]) - 1
                cluster_assets = self.clusters[cluster_idx]
                
                # Performance metrics
                mean_return = self.cluster_returns[column].mean() * 252  # Annualized
                volatility = self.cluster_returns[column].std() * np.sqrt(252)  # Annualized
                sharpe = mean_return / volatility if volatility != 0 else 0
                
                # Calculate drawdowns
                cum_returns = (1 + self.cluster_returns[column]).cumprod()
                running_max = cum_returns.cummax()
                drawdowns = (cum_returns / running_max) - 1
                max_drawdown = drawdowns.min()
                
                # Calculate correlation matrix within cluster
                if len(cluster_assets) > 1:
                    intra_corr = returns[cluster_assets].corr().values
                    avg_intra_corr = np.sum(np.triu(intra_corr, k=1)) / (len(cluster_assets) * (len(cluster_assets) - 1) / 2)
                else:
                    avg_intra_corr = 1.0
                
                performance[column] = {
                    'assets': cluster_assets,
                    'size': len(cluster_assets),
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'avg_intra_correlation': avg_intra_corr
                }
            
            self.cluster_performances = performance
            analysis['performance'] = performance
            
            # Calculate inter-cluster correlation
            inter_corr = self.cluster_returns.corr()
            analysis['inter_cluster_correlation'] = inter_corr.to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing clusters: {e}")
            raise
    
    def plot_hierarchical_clustering(self, returns: pd.DataFrame, figsize=(12, 8)) -> plt.Figure:
        """
        Plot dendrogram for hierarchical clustering.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame of asset returns.
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with dendrogram.
        """
        try:
            # Calculate distance matrix
            dist_matrix = self.calculate_distance_matrix(returns)
            condensed_dist = squareform(dist_matrix)
            
            # Compute linkage matrix
            Z = linkage(condensed_dist, method=self.linkage_method)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot dendrogram
            dendrogram(
                Z,
                labels=returns.columns,
                ax=ax,
                leaf_rotation=90,
                color_threshold=0.7 * max(Z[:, 2])
            )
            
            # Add labels and title
            ax.set_title('Hierarchical Clustering Dendrogram')
            ax.set_xlabel('Assets')
            ax.set_ylabel('Distance')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting hierarchical clustering: {e}")
            raise
    
    def plot_cluster_correlation_heatmap(self, figsize=(10, 8)) -> plt.Figure:
        """
        Plot correlation heatmap for clusters.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with heatmap.
        """
        if self.cluster_returns is None:
            logger.error("No cluster returns available. Calculate cluster returns first.")
            raise ValueError("No cluster returns available. Calculate cluster returns first.")
        
        try:
            # Calculate correlation matrix
            corr_matrix = self.cluster_returns.corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
            
            # Add labels
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns)
            ax.set_yticklabels(corr_matrix.columns)
            
            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white")
            
            # Add title
            ax.set_title('Cluster Correlation Heatmap')
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting cluster correlation heatmap: {e}")
            raise
    
    def plot_cluster_performance(self, figsize=(12, 8)) -> plt.Figure:
        """
        Plot performance metrics for each cluster.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib figure with performance plot.
        """
        if self.cluster_performances is None:
            logger.error("No cluster performances available. Analyze clusters first.")
            raise ValueError("No cluster performances available. Analyze clusters first.")
        
        try:
            # Extract performance metrics
            clusters = list(self.cluster_performances.keys())
            returns = [self.cluster_performances[c]['mean_return'] * 100 for c in clusters]
            vols = [self.cluster_performances[c]['volatility'] * 100 for c in clusters]
            sharpes = [self.cluster_performances[c]['sharpe_ratio'] for c in clusters]
            sizes = [self.cluster_performances[c]['size'] for c in clusters]
            
            # Normalize sizes for scatter plot
            norm_sizes = np.array(sizes) * 20
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Plot risk-return scatter
            scatter = ax1.scatter(vols, returns, s=norm_sizes, c=sharpes, cmap='viridis', alpha=0.7)
            
            # Add labels for each cluster
            for i, cluster in enumerate(clusters):
                ax1.annotate(cluster, (vols[i], returns[i]),
                             xytext=(5, 5), textcoords='offset points')
            
            # Add labels and title
            ax1.set_xlabel('Volatility (%)')
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Cluster Risk-Return Profile')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar for Sharpe ratio
            cbar = fig.colorbar(scatter, ax=ax1)
            cbar.set_label('Sharpe Ratio')
            
            # Plot bar chart of average intra-correlation
            intra_corrs = [self.cluster_performances[c]['avg_intra_correlation'] for c in clusters]
            ax2.bar(clusters, intra_corrs, alpha=0.7)
            
            # Add labels and title
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Average Intra-Correlation')
            ax2.set_title('Cluster Cohesion')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1)
            
            # Rotate x labels
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting cluster performance: {e}")
            raise


# Example usage (when run as a script)
if __name__ == "__main__":
    import yaml
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open('../config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize clustering
    clustering = AssetClustering(config)
    
    # Generate some sample return data for testing
    np.random.seed(42)
    n_assets = 20
    n_timepoints = 1000
    
    # Create correlated returns to simulate asset groups
    group1 = np.random.randn(n_timepoints, 5) * 0.01 + np.random.randn(n_timepoints, 1) * 0.02
    group2 = np.random.randn(n_timepoints, 5) * 0.01 + np.random.randn(n_timepoints, 1) * 0.02
    group3 = np.random.randn(n_timepoints, 5) * 0.01 + np.random.randn(n_timepoints, 1) * 0.02
    group4 = np.random.randn(n_timepoints, 5) * 0.01 + np.random.randn(n_timepoints, 1) * 0.02
    
    returns_data = np.hstack([group1, group2, group3, group4])
    
    # Create DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_timepoints)
    assets = [f'Asset_{i+1}' for i in range(n_assets)]
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)
    
    # Find optimal number of clusters
    optimal_n_clusters = clustering.find_optimal_clusters(returns)
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    
    # Perform clustering
    clusters = clustering.cluster_assets(returns, n_clusters=optimal_n_clusters)
    
    # Print clusters
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {', '.join(cluster)}")
    
    # Calculate cluster returns
    cluster_returns = clustering.calculate_cluster_returns(returns)
    
    # Analyze clusters
    analysis = clustering.analyze_clusters(returns)
    
    # Plot results
    fig1 = clustering.plot_hierarchical_clustering(returns)
    fig2 = clustering.plot_cluster_correlation_heatmap()
    fig3 = clustering.plot_cluster_performance()
    
    plt.show()
