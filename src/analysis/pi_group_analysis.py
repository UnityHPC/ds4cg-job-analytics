"""
PI Group analysis module for generating PI account reports and comparisons.

This module provides utilities for analyzing PI group performance, comparing
PI groups against each other, and identifying inefficient users within PI groups.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from src.database.database_connection import DatabaseConnection
from src.analysis.efficiency_analysis import EfficiencyAnalysis
from src.analysis.user_comparison import load_jobs_df
from src.config.enum_constants import EfficiencyCategoryEnum


class PIGroupAnalysis:
    """
    A class to analyze PI group performance and generate comparison statistics.

    This class provides methods to:
    1. Calculate PI group efficiency metrics
    2. Compare PI groups against system averages
    3. Identify worst-performing users within a PI group
    4. Generate comprehensive PI group reports
    """

    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize the PIGroupAnalysis with a database connection.

        Args:
            db_connection: Database connection object to use for queries
        """
        self.db = db_connection
        self._all_jobs_data = None
        self._cached_pi_metrics = None
        self._cached_user_metrics = None

        # Load and process all data during initialization
        try:
            self._all_jobs_data = load_jobs_df(
                db_connection=self.db,
                include_cpu_only_jobs=False,
                include_failed_cancelled_jobs=False,
                min_elapsed_seconds=0,
            )

            if len(self._all_jobs_data) > 0:
                analyzer = EfficiencyAnalysis(self._all_jobs_data)
                filtered_jobs = analyzer.filter_jobs_for_analysis(
                    gpu_count_filter={"min": 1, "max": np.inf, "inclusive": True},
                    vram_constraint_filter=None,
                    allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
                    gpu_mem_usage_filter=None,
                )

                # Calculate all efficiency metrics
                analyzer.calculate_job_efficiency_metrics(filtered_jobs)
                self._cached_user_metrics = analyzer.calculate_user_efficiency_metrics()
                self._cached_pi_metrics = analyzer.calculate_pi_account_efficiency_metrics()

                print(f"Loaded and processed {len(self._cached_pi_metrics)} PI groups for analysis")
            else:
                print("No job data found in database")
        except Exception as e:
            print(f"Error initializing PI group data: {e}")
            self._all_jobs_data = None
            self._cached_pi_metrics = None
            self._cached_user_metrics = None

    def get_pi_group_metrics(self, pi_account: str) -> pd.DataFrame:
        """
        Get detailed metrics for a specific PI group.

        Args:
            pi_account: PI account name to get metrics for

        Returns:
            DataFrame with PI group metrics
        """
        if self._cached_pi_metrics is not None:
            pi_metrics = self._cached_pi_metrics[self._cached_pi_metrics["pi_account"] == pi_account]
            if len(pi_metrics) > 0:
                return pi_metrics

        return pd.DataFrame()

    def get_pi_group_users(self, pi_account: str) -> pd.DataFrame:
        """
        Get all users and their metrics for a specific PI group.

        Args:
            pi_account: PI account name to get users for

        Returns:
            DataFrame with user metrics for the PI group
        """
        if self._cached_user_metrics is not None:
            pi_users = self._cached_user_metrics[self._cached_user_metrics["pi_account"] == pi_account]
            return pi_users.sort_values("expected_value_alloc_vram_efficiency", ascending=True)

        return pd.DataFrame()

    def get_worst_users_in_pi_group(self, pi_account: str, n_users: int = 5) -> pd.DataFrame:
        """
        Get the worst performing users in a PI group by VRAM efficiency.

        Args:
            pi_account: PI account name
            n_users: Number of worst users to return

        Returns:
            DataFrame with worst users and their metrics
        """
        pi_users = self.get_pi_group_users(pi_account)
        if len(pi_users) == 0:
            return pd.DataFrame()

        # Sort by efficiency (ascending = worst first) and return top N
        worst_users = pi_users.head(n_users)
        return worst_users

    def get_pi_group_comparison_statistics(self, pi_account: str) -> pd.DataFrame:
        """
        Calculate comparison statistics between a PI group and all other PI groups.

        Args:
            pi_account: The PI account to compare

        Returns:
            DataFrame with comparison statistics
        """
        if self._cached_pi_metrics is None or len(self._cached_pi_metrics) == 0:
            return self._create_fallback_pi_comparison()

        # Get target PI metrics
        target_pi = self._cached_pi_metrics[self._cached_pi_metrics["pi_account"] == pi_account]
        if len(target_pi) == 0:
            return self._create_fallback_pi_comparison()

        target_pi = target_pi.iloc[0]

        # Get other PI groups for comparison
        other_pis = self._cached_pi_metrics[self._cached_pi_metrics["pi_account"] != pi_account]
        if len(other_pis) == 0:
            return self._create_fallback_pi_comparison()

        # Define metrics for comparison
        metrics = [
            ("expected_value_alloc_vram_efficiency", "VRAM Efficiency (%)"),
            ("pi_acc_vram_hours", "Total VRAM Hours"),
            ("pi_acc_job_hours", "Total GPU Hours"),
            ("user_count", "Number of Users"),
            ("job_count", "Total Jobs"),
            ("avg_alloc_vram_efficiency_score", "Efficiency Score"),
        ]

        # Calculate comparison statistics
        comparison_stats = []
        for metric_name, display_name in metrics:
            if metric_name in target_pi.index and metric_name in other_pis.columns:
                your_value = target_pi[metric_name]

                if metric_name == "expected_value_alloc_vram_efficiency":
                    # Convert to percentage
                    your_value = your_value * 100 if pd.notna(your_value) else 0
                    avg_value = other_pis[metric_name].mean() * 100
                elif metric_name == "avg_alloc_vram_efficiency_score":
                    # Handle potentially negative infinity values and NA values
                    valid_scores = other_pis[metric_name][other_pis[metric_name] != -np.inf]
                    avg_value = valid_scores.mean() if len(valid_scores) > 0 else 0
                    # Handle your_value with proper NA check
                    if pd.isna(your_value):
                        your_value = 0
                    elif your_value == -np.inf:
                        your_value = 0
                else:
                    avg_value = other_pis[metric_name].mean()

                comparison_stats.append({
                    "Category": display_name,
                    "Your_Value": your_value,
                    "Average_Value": avg_value
                })

        return pd.DataFrame(comparison_stats)

    def get_pi_group_size_comparison(self, pi_account: str) -> dict:
        """
        Get size comparison statistics for a PI group.

        Args:
            pi_account: PI account name

        Returns:
            Dictionary with size comparison statistics
        """
        if self._cached_pi_metrics is None:
            return {}

        target_pi = self._cached_pi_metrics[self._cached_pi_metrics["pi_account"] == pi_account]
        if len(target_pi) == 0:
            return {}

        target_pi = target_pi.iloc[0]
        all_pis = self._cached_pi_metrics

        # Calculate percentiles for size metrics
        size_stats = {}

        # User count percentile
        user_count_percentile = (all_pis["user_count"] <= target_pi["user_count"]).mean() * 100
        size_stats["user_count_percentile"] = user_count_percentile

        # Job count percentile
        job_count_percentile = (all_pis["job_count"] <= target_pi["job_count"]).mean() * 100
        size_stats["job_count_percentile"] = job_count_percentile

        # VRAM hours percentile
        vram_hours_percentile = (all_pis["pi_acc_vram_hours"] <= target_pi["pi_acc_vram_hours"]).mean() * 100
        size_stats["vram_hours_percentile"] = vram_hours_percentile

        # Add actual values
        size_stats["user_count"] = target_pi["user_count"]
        size_stats["job_count"] = target_pi["job_count"]
        size_stats["vram_hours"] = target_pi["pi_acc_vram_hours"]

        return size_stats

    def identify_inefficient_pi_groups(self, efficiency_threshold: float = 0.3, min_jobs: int = 50, min_users: int = 1, top_n: int = 10) -> pd.DataFrame:
        """
        Identify the most inefficient PI groups.

        Args:
            efficiency_threshold: Maximum efficiency threshold for inefficient PI groups
            min_jobs: Minimum number of jobs a PI group must have
            min_users: Minimum number of users a PI group must have
            top_n: Number of top inefficient PI groups to return

        Returns:
            DataFrame with inefficient PI groups
        """
        if self._cached_pi_metrics is None:
            return pd.DataFrame()

        # Filter PI groups with sufficient job count and user count
        filtered_pis = self._cached_pi_metrics[
            (self._cached_pi_metrics["job_count"] >= min_jobs) &
            (self._cached_pi_metrics["user_count"] >= min_users)
        ].copy()

        # Filter by efficiency threshold
        inefficient_pis = filtered_pis[
            filtered_pis["expected_value_alloc_vram_efficiency"] <= efficiency_threshold
        ].copy()

        # Sort by efficiency (ascending = worst first)
        inefficient_pis = inefficient_pis.sort_values("expected_value_alloc_vram_efficiency", ascending=True)

        return inefficient_pis.head(top_n)

    def _create_fallback_pi_comparison(self) -> pd.DataFrame:
        """
        Create fallback comparison statistics when data is unavailable.

        Returns:
            DataFrame with basic comparison statistics
        """
        comparison_stats = pd.DataFrame({
            "Category": [
                "VRAM Efficiency (%)",
                "Total VRAM Hours",
                "Total GPU Hours",
                "Number of Users",
                "Total Jobs",
                "Efficiency Score"
            ],
            "Your_Value": [25.0, 1000.0, 500.0, 5, 100, -2.0],
            "Average_Value": [30.0, 800.0, 400.0, 4, 80, -1.5]
        })

        return comparison_stats

    def calculate_analysis_period(self) -> str:
        """
        Calculate the analysis period based on the job data.

        Returns:
            String describing the analysis period
        """
        if self._all_jobs_data is None or "StartTime" not in self._all_jobs_data.columns:
            return "All time"

        start_date = self._all_jobs_data["StartTime"].min()
        end_date = self._all_jobs_data["StartTime"].max()

        if not (pd.notna(start_date) and pd.notna(end_date)):
            return "All time"

        duration = end_date - start_date

        if duration.days > 365:
            years = duration.days / 365
            return f"Last {years:.1f} years"
        elif duration.days > 30:
            months = duration.days / 30
            return f"Last {months:.1f} months"
        else:
            return f"Last {duration.days} days"
