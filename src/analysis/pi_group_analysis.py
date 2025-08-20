"""
PI Group Analysis module for analyzing PI group performance and efficiency.

This module provides functionality to analyze PI groups, identify inefficient groups,
and generate comparative statistics for PI group reporting.
"""

import pandas as pd
import numpy as np

from src.analysis.efficiency_analysis import EfficiencyAnalysis
from src.utilities.load_and_preprocess_jobs import load_and_preprocess_jobs


class PIGroupAnalysis:
    """
    Class to analyze PI group performance and efficiency metrics.
    
    This class builds upon EfficiencyAnalysis to provide PI group-specific
    analysis capabilities including group comparisons and rankings.
    """
    
    def __init__(self, db_path: str) -> None:
        """
        Initialize PIGroupAnalysis with database connection.
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path
        self.efficiency_analyzer: EfficiencyAnalysis | None = None
        self._jobs_df: pd.DataFrame | None = None
        self._pi_metrics: pd.DataFrame | None = None
        self._user_metrics: pd.DataFrame | None = None
        
    def _load_data(self) -> None:
        """Load and prepare data for analysis if not already loaded."""
        if self._jobs_df is None:
            try:
                # Load and preprocess jobs data using the utility function
                self._jobs_df = load_and_preprocess_jobs(
                    db_path=self.db_path,
                    include_cpu_only_jobs=False,
                    include_failed_cancelled_jobs=False,
                    min_elapsed_seconds=600,
                )
                
                if len(self._jobs_df) == 0:
                    print("No job data found in database")
                    return
                
                # Initialize efficiency analyzer
                self.efficiency_analyzer = EfficiencyAnalysis(self._jobs_df)
                
                # Filter jobs for analysis
                filtered_jobs = self.efficiency_analyzer.filter_jobs_for_analysis(
                    requested_vram_filter={"min": 0, "max": np.inf, "inclusive": False}
                )
                
                # Calculate all efficiency metrics
                self.efficiency_analyzer.calculate_all_efficiency_metrics(filtered_jobs)
                
                # Cache the calculated metrics
                self._user_metrics = self.efficiency_analyzer.users_with_efficiency_metrics
                self._pi_metrics = self.efficiency_analyzer.pi_accounts_with_efficiency_metrics
                
                print(f"Loaded and processed {len(self._jobs_df)} job records for PI group analysis")
                
            except Exception as e:
                print(f"Error loading data: {e}")
                self._jobs_df = None
                self.efficiency_analyzer = None
    
    def get_pi_group_metrics(self, pi_account: str) -> pd.DataFrame:
        """
        Get efficiency metrics for a specific PI group.
        
        Args:
            pi_account: PI account name to get metrics for
            
        Returns:
            DataFrame with PI group metrics (single row)
        """
        self._load_data()
        
        if self._pi_metrics is None:
            return pd.DataFrame()
            
        return self._pi_metrics[self._pi_metrics['pi_account'] == pi_account].copy()
    
    def get_pi_group_users(self, pi_account: str) -> pd.DataFrame:
        """
        Get all users and their metrics for a specific PI group.
        
        Args:
            pi_account: PI account name to get users for
            
        Returns:
            DataFrame with user metrics for the PI group
        """
        self._load_data()
        
        if self._user_metrics is None:
            return pd.DataFrame()
            
        return self._user_metrics[self._user_metrics['pi_account'] == pi_account].copy()
    
    def get_worst_users_in_pi_group(self, pi_account: str, n_users: int = 5) -> pd.DataFrame:
        """
        Get the worst performing users in a PI group based on VRAM efficiency.
        
        Args:
            pi_account: PI account name
            n_users: Number of worst users to return
            
        Returns:
            DataFrame with worst performing users
        """
        pi_users = self.get_pi_group_users(pi_account)
        
        if len(pi_users) == 0:
            return pd.DataFrame()
        
        # Convert efficiency column to numeric, handling any non-numeric values
        pi_users = pi_users.copy()
        pi_users['expected_value_requested_vram_efficiency'] = pd.to_numeric(
            pi_users['expected_value_requested_vram_efficiency'], 
            errors='coerce'
        )
        
        # Drop rows where efficiency is NaN (couldn't be converted to numeric)
        pi_users_clean = pi_users.dropna(subset=['expected_value_requested_vram_efficiency'])
        
        if len(pi_users_clean) == 0:
            print(f"No users with valid efficiency data for PI account {pi_account}")
            return pd.DataFrame()
        
        # Sort by efficiency (ascending = worst first) and take top n
        worst_users = pi_users_clean.nsmallest(
            n_users, 
            'expected_value_requested_vram_efficiency'
        ).copy()
        
        return worst_users
    
    def get_pi_group_comparison_statistics(self, pi_account: str) -> dict:
        """
        Get comparison statistics for a PI group against all other groups.
        
        Args:
            pi_account: PI account name
            
        Returns:
            Dictionary with comparison statistics
        """
        self._load_data()
        
        if self._pi_metrics is None:
            return {}
        
        pi_metrics = self.get_pi_group_metrics(pi_account)
        if len(pi_metrics) == 0:
            return {}
        
        pi_row = pi_metrics.iloc[0]
        all_pi_metrics = self._pi_metrics
        
        # Ensure we have more than just this PI group
        if len(all_pi_metrics) <= 1:
            return {
                'efficiency_percentile': 50.0,
                'vram_hours_percentile': 50.0, 
                'job_count_percentile': 50.0,
                'total_pi_groups': len(all_pi_metrics),
                'efficiency_rank': 1,
                'avg_efficiency_all_groups': pi_row['expected_value_requested_vram_efficiency'],
                'median_efficiency_all_groups': pi_row['expected_value_requested_vram_efficiency']
            }
        
        # Calculate percentiles (what percentage of groups this group is better than)
        efficiency_percentile = (
            (all_pi_metrics['expected_value_requested_vram_efficiency'] < 
             pi_row['expected_value_requested_vram_efficiency']).sum() / len(all_pi_metrics) * 100
        )
        
        vram_hours_percentile = (
            (all_pi_metrics['pi_acc_vram_hours'] < 
             pi_row['pi_acc_vram_hours']).sum() / len(all_pi_metrics) * 100
        )
        
        job_count_percentile = (
            (all_pi_metrics['job_count'] < 
             pi_row['job_count']).sum() / len(all_pi_metrics) * 100
        )
        
        return {
            'efficiency_percentile': efficiency_percentile,
            'vram_hours_percentile': vram_hours_percentile, 
            'job_count_percentile': job_count_percentile,
            'total_pi_groups': len(all_pi_metrics),
            'efficiency_rank': len(all_pi_metrics) - int(efficiency_percentile * len(all_pi_metrics) / 100),
            'avg_efficiency_all_groups': all_pi_metrics['expected_value_requested_vram_efficiency'].mean(),
            'median_efficiency_all_groups': all_pi_metrics['expected_value_requested_vram_efficiency'].median()
        }
    
    def get_pi_group_size_comparison(self, pi_account: str) -> dict:
        """
        Get size comparison statistics for a PI group.
        
        Args:
            pi_account: PI account name
            
        Returns:
            Dictionary with size comparison statistics
        """
        self._load_data()
        
        if self._pi_metrics is None:
            return {}
        
        pi_metrics = self.get_pi_group_metrics(pi_account)
        if len(pi_metrics) == 0:
            return {}
        
        pi_row = pi_metrics.iloc[0]
        all_pi_metrics = self._pi_metrics
        
        user_count = pi_row['user_count']
        
        # Categorize group size
        if user_count == 1:
            size_category = "Individual"
        elif user_count <= 3:
            size_category = "Small"
        elif user_count <= 8:
            size_category = "Medium"
        else:
            size_category = "Large"
        
        # Calculate size statistics
        avg_users_per_group = all_pi_metrics['user_count'].mean()
        median_users_per_group = all_pi_metrics['user_count'].median()
        
        return {
            'user_count': user_count,
            'size_category': size_category,
            'avg_users_per_group': avg_users_per_group,
            'median_users_per_group': median_users_per_group,
            'larger_groups_count': (all_pi_metrics['user_count'] > user_count).sum(),
            'smaller_groups_count': (all_pi_metrics['user_count'] < user_count).sum()
        }
    
    def calculate_analysis_period(self) -> str:
        """
        Calculate and return a descriptive string for the analysis period.
        
        Returns:
            String describing the analysis period
        """
        self._load_data()
        
        if self._jobs_df is None or 'StartTime' not in self._jobs_df.columns:
            return "Unknown period"
        
        start_date = self._jobs_df['StartTime'].min()
        end_date = self._jobs_df['StartTime'].max()
        duration = end_date - start_date
        
        if duration.days > 365:
            return f"{duration.days // 365} year{'s' if duration.days // 365 > 1 else ''}"
        elif duration.days > 30:
            return f"{duration.days // 30} month{'s' if duration.days // 30 > 1 else ''}"
        else:
            return f"{duration.days} day{'s' if duration.days > 1 else ''}"
    
    def identify_inefficient_pi_groups(
        self, 
        efficiency_threshold: float = 0.3,
        min_jobs: int = 50,
        min_users: int = 1,
        top_n: int | None = None
    ) -> pd.DataFrame:
        """
        Identify inefficient PI groups based on criteria.
        
        Args:
            efficiency_threshold: Maximum efficiency threshold for inefficient groups
            min_jobs: Minimum number of jobs required
            min_users: Minimum number of users required
            top_n: Number of top inefficient groups to return (None for all)
            
        Returns:
            DataFrame with inefficient PI groups sorted by efficiency (worst first)
        """
        self._load_data()
        
        if self._pi_metrics is None:
            return pd.DataFrame()
        
        # Apply filters
        filtered_groups = self._pi_metrics[
            (self._pi_metrics['expected_value_requested_vram_efficiency'] <= efficiency_threshold) &
            (self._pi_metrics['job_count'] >= min_jobs) &
            (self._pi_metrics['user_count'] >= min_users)
        ].copy()
        
        # Sort by efficiency (worst first)
        filtered_groups = filtered_groups.sort_values(
            'expected_value_requested_vram_efficiency', 
            ascending=True
        )
        
        # Limit to top N if specified
        if top_n is not None:
            filtered_groups = filtered_groups.head(top_n)
        
        return filtered_groups
    
    def get_all_pi_groups(self) -> pd.DataFrame:
        """
        Get all PI groups with their metrics.
        
        Returns:
            DataFrame with all PI group metrics
        """
        self._load_data()
        
        if self._pi_metrics is None:
            return pd.DataFrame()
        
        return self._pi_metrics.copy()
    
    def get_pi_group_job_details(self, pi_account: str) -> pd.DataFrame:
        """
        Get detailed job information for a specific PI group.
        
        Args:
            pi_account: PI account name
            
        Returns:
            DataFrame with job details for the PI group
        """
        self._load_data()
        
        if self.efficiency_analyzer is None or self.efficiency_analyzer.jobs_with_efficiency_metrics is None:
            return pd.DataFrame()
        
        jobs_with_metrics = self.efficiency_analyzer.jobs_with_efficiency_metrics
        
        # Filter jobs for this PI account
        pi_jobs = jobs_with_metrics[
            jobs_with_metrics['Account'] == pi_account
        ].copy()
        
        return pi_jobs
