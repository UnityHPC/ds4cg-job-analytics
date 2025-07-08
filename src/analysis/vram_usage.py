"""Functions to analyze efficiency of Jobs based on their VRAM usage.

The aim is to identify potential inefficiencies in GPU usage and notify users or PIs about these issues.
"""

from pathlib import Path
from src.preprocess.preprocess import preprocess_data
from src.database.database_connection import DatabaseConnection  # Make sure this matches the actual file/class name
from src.config.constants import MIN_ELAPSED_SECONDS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess.preprocess import ram_map
import os

def load_jobs_dataframe_from_duckdb(
    db_path: str = None,
    table_name: str = "Jobs",
    sample_size: int = None,
    random_state: int = None,
    query: str = None
) -> pd.DataFrame:
    """
    Connect to the DuckDB slurm_data_small.db and return the jobs table as a pandas DataFrame.

    Args:
        db_path (str or Path, optional): Path to the DuckDB database. Defaults to 'data/slurm_data_small.db'.
        table_name (str, optional): Table name to query. Defaults to 'Jobs'.
        sample_size (int, optional): Number of samples to return.
        random_state (int, optional): Random state for sampling.
        query (str, optional): SQL query to execute.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    if db_path is None:
        db_path = Path(__file__).resolve().parents[2] / "data" / "slurm_data.db"
    db = DatabaseConnection(str(db_path))
    jobs_df = db.fetch_all_jobs(table_name=table_name) if query is None else db.fetch_query(query)
    processed_data = preprocess_data(
        jobs_df,
        include_failed_cancelled_jobs=False,
    )
    db.disconnect()
    if sample_size is not None:
        processed_data = processed_data.sample(n=sample_size, random_state=random_state)
    return processed_data


class EfficiencyAnalysis:
    """Class to encapsulate the efficiency analysis of jobs based on VRAM usage.

It provides methods to load data, analyze workload efficiency, and evaluate CPU-GPU usage patterns.
"""

    def __init__(
        self,
        db_path: str = None,
        table_name: str = "Jobs",
        sample_size: int = None,
        random_state: int = None,
        query: str = None
    ):
        self.jobs_df = load_jobs_dataframe_from_duckdb(
            db_path=db_path,
            table_name=table_name,
            sample_size=sample_size,
            random_state=random_state,
            query=query
        )
        self.efficiency_df = None
        self.analysis_results = None

    def calculate_efficiency_metrics(
        self,
        vram_constraint_filter=None,
        vram_constraint_filter=None,
        allocated_vram_greater_than=0,
        gpu_mem_usage_min=None,
        gpu_mem_usage_max=None,
        gpu_mem_usage_exact=None,
        gpus_min=1,
        elapsed_seconds_min=MIN_ELAPSED_SECONDS,
    ) -> pd.DataFrame:
        elapsed_seconds_min=DEFAULT_MIN_ELAPSED_SECONDS,
    ) -> pd.DataFrame:
        """
        Analyze jobs based on constraints, GPU allocation, and usage criteria.

        Args:
            vram_constraint_filter (int, float, or callable): Value or function to filter vram_constraint
            allocated_vram_greater_than (int, float): allocated_vram greater than this value
            gpu_mem_usage_min (int, float, optional): Minimum GPUMemUsage (inclusive)
            gpu_mem_usage_max (int, float, optional): Maximum GPUMemUsage (inclusive)
            gpu_mem_usage_exact (int, float, optional): If set, select only rows where GPUMemUsage == this value
            gpus_min (int): Minimum GPUs allocated
            elapsed_seconds_min (int): Minimum elapsed time in seconds

        Returns:
            DataFrame: Filtered jobs with efficiency metrics added
        """
        # Flexible filter for vram_constraint
        if vram_constraint_filter is not None:
            if callable(vram_constraint_filter):
                mask = self.jobs_df["vram_constraint"].apply(vram_constraint_filter)
            else:
                mask = self.jobs_df["vram_constraint"] == vram_constraint_filter
        else:
            mask = pd.Series([True] * len(self.jobs_df), index=self.jobs_df.index)
        # GPU memory usage filter
        gpu_mem_mask = pd.Series([True] * len(self.jobs_df), index=self.jobs_df.index)
        if gpu_mem_usage_exact is not None:
            gpu_mem_mask &= self.jobs_df["GPUMemUsage"] == gpu_mem_usage_exact
        else:
            if gpu_mem_usage_min is not None:
                gpu_mem_mask &= self.jobs_df["GPUMemUsage"] >= gpu_mem_usage_min
            if gpu_mem_usage_max is not None:
                gpu_mem_mask &= self.jobs_df["GPUMemUsage"] <= gpu_mem_usage_max

        filtered_jobs = self.jobs_df[
            mask
            & (self.jobs_df["allocated_vram"] > allocated_vram_greater_than)
            & gpu_mem_mask
            & (self.jobs_df["GPUs"] >= gpus_min)
            & (self.jobs_df["Elapsed"].dt.total_seconds() >= elapsed_seconds_min)
        ].copy()

        # Calculate efficiency metrics
        filtered_jobs["gpu_memory_used_gb"] = filtered_jobs["GPUMemUsage"] / (2 ** 30)
        filtered_jobs["vram_efficiency"] = filtered_jobs["gpu_memory_used_gb"] / filtered_jobs["allocated_vram"]
        filtered_jobs["gpu_hours"] = (
            filtered_jobs["Elapsed"].dt.total_seconds() * filtered_jobs["GPUs"]
        ) / 3600

        # Calculate weighted_vram_efficiency per job, normalized by total gpu_hours for that specific User
        user_gpu_hours = filtered_jobs.groupby("User")["gpu_hours"].transform("sum")
        filtered_jobs["user_weighted_vram_efficiency"] = (
            filtered_jobs["vram_efficiency"] * 100 * filtered_jobs["gpu_hours"]
        ) / user_gpu_hours

        # Calculate weighted vram efficiency per job, normalized by total gpu_hours for that specific PI
        pi_gpu_hours = filtered_jobs.groupby("Account")["gpu_hours"].transform("sum")
        filtered_jobs["pi_weighted_vram_efficiency"] = (
            filtered_jobs["vram_efficiency"] * 100 * filtered_jobs["gpu_hours"]
        ) / pi_gpu_hours

        # Categorize by efficiency
        filtered_jobs["efficiency_category"] = pd.cut(
            filtered_jobs["vram_efficiency"],
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=["Very Low (<10%)", "Low (10-30%)", "Medium (30-60%)", "High (60-100%)"],
        )

        # Add CPU memory analysis if available
        if "CPUMemUsage" in self.jobs_df.columns:
            filtered_jobs["cpu_memory_gb"] = filtered_jobs["CPUMemUsage"] / (2 ** 30)
            filtered_jobs["cpu_gpu_ratio"] = (
                filtered_jobs["cpu_memory_gb"] /
                filtered_jobs["gpu_memory_used_gb"].clip(lower=0.1)
            )

        # Duration analysis
        filtered_jobs["duration_category"] = pd.cut(
            filtered_jobs["gpu_hours"],
            bins=[0, 1, 6, 24, 48, float("inf")],
            labels=["Short (<1h)", "Medium (1-6h)", "Long (6-24h)", "Under two days (24-48h)", "Over two days (>48h)"]
        )

        self.efficiency_df = filtered_jobs
        return self.efficiency_df


    def evaluate_cpu_gpu_usage(
            self,
            hours_percentage_threshold=25,
            vram_efficiency_threshold=0.3
        ):
        """
        This method evaluates the efficiency of GPU jobs based on VRAM usage and CPU-GPU balance.

        Args:
            hours_percentage_threshold (float): Threshold for high waste GPU hours as a percentage of total GPU hours
            vram_efficiency_threshold (float): Threshold for VRAM efficiency to consider a job as high waste

        Returns:
            dict: Analysis results with balance patterns and recommendations
        """
        #TODO: Needs refactoring to parametrize the analysis thresholds and make it more flexible
        #TODO: Need to separate the analysis into different methods for clarity
        #TODO: Separate the CPU-GPU balance analysis from the VRAM efficiency analysis

        analysis = {}

        # Ensure efficiency_df is available
        if self.efficiency_df is None:
            raise ValueError("Efficiency DataFrame is not available. Please run analyze_workload_efficiency first.")
        
        # Overall statistics
        analysis["total_jobs"] = len(self.efficiency_df)
        analysis["total_gpu_hours"] = self.efficiency_df["gpu_hours"].sum()
        analysis["avg_efficiency"] = self.efficiency_df["vram_efficiency"].mean()
        analysis["median_efficiency"] = self.efficiency_df["vram_efficiency"].median()

        # Efficiency distribution analysis
        efficiency_analysis = (
            self.efficiency_df.groupby("efficiency_category", observed=False)
            .agg(
                {
                    "JobID": "count",
                    "gpu_hours": "sum",
                    "vram_efficiency": "mean",
                    "allocated_vram": "mean",
                    "gpu_memory_used_gb": "mean",
                }
            )
            .round(3)
        )

        efficiency_analysis.columns = [
            "Job_Count",
            "GPU_Hours",
            "Avg_Efficiency",
            "Avg_Allocated_GB",
            "Avg_Used_GB",
        ]
        efficiency_analysis["Share of tota GPU Hours"] = (
            efficiency_analysis["GPU_Hours"] / analysis["total_gpu_hours"] * 100
        ).round(1)
        analysis["efficiency_patterns"] = efficiency_analysis

        # CPU-GPU balance analysis (if CPU data available)
        if "cpu_gpu_ratio" in self.efficiency_df.columns:
            # Categorize workloads by CPU-GPU balance
            self.efficiency_df["workload_type"] = pd.cut(
                self.efficiency_df["cpu_gpu_ratio"],
                bins=[0, 1, 5, 20, float("inf")],
                labels=[
                    "GPU-intensive (CPU<GPU)",
                    "Balanced (CPU≈GPU)",
                    "CPU-heavy (CPU>GPU)",
                    "Very CPU-heavy (CPU>>GPU)",
                ],
            )

            balance_analysis = self.efficiency_df.groupby("workload_type", observed=False).agg(
                {"JobID": "count", "gpu_hours": "sum", "vram_efficiency": "mean", "cpu_gpu_ratio": "mean"}
            )
            analysis["cpu_gpu_balance"] = balance_analysis

        # Over-allocation analysis
        high_waste_jobs = self.efficiency_df[self.efficiency_df["vram_efficiency"] <= vram_efficiency_threshold]
        analysis["high_waste_jobs"] = len(high_waste_jobs)
        analysis["high_waste_gpu_hours"] = high_waste_jobs["gpu_hours"].sum()
        analysis["high_waste_hours_share"] = analysis["high_waste_gpu_hours"] / analysis["total_gpu_hours"] * 100

        # Duration vs efficiency correlation
        duration_efficiency = self.efficiency_df.groupby("duration_category", observed=False).agg(
            {"JobID": "count", "vram_efficiency": "mean", "gpu_hours": "sum"}
        )
        analysis["duration_efficiency_patterns"] = duration_efficiency

        # Generate recommendations
        analysis_report = []

        low_efficiency_hours = efficiency_analysis.loc[
            efficiency_analysis.index.isin(["Very Low (<10%)", "Low (10-30%)"]), "GPU_Hours"
        ].sum()
        low_efficiency_percentage = low_efficiency_hours / analysis["total_gpu_hours"] * 100

        if low_efficiency_percentage > 50:
            analysis_report.append("CRITICAL: >50% of GPU hours have <30% efficiency - immediate optimization needed")
        elif low_efficiency_percentage > 30:
            analysis_report.append("HIGH PRIORITY: Significant inefficiency detected - user education campaign needed")

        if analysis["high_waste_hours_share"] > hours_percentage_threshold:
            analysis_report.append(
                    f"MAJOR OVER-ALLOCATION: >{hours_percentage_threshold}% of total GPU hours has been wasted "
                    f"with jobs with less than {vram_efficiency_threshold * 100}% efficiency."
            )

        analysis["report"] = analysis_report

        self.analysis_results = analysis

        return self.analysis_results
    
    def find_inefficient_users_weighted_by_hours(self, efficiency_threshold=0.3, min_jobs=5):
        """
        Identify users with low average VRAM efficiency across their jobs, weighted by the hours they were inefficient.

        Args:
            efficiency_threshold (float): Threshold for VRAM efficiency to consider a user as inefficient
            min_jobs (int): Minimum number of jobs a user must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with users and their average VRAM efficiency
        """
        if self.efficiency_df is None:
            raise ValueError("Efficiency DataFrame is not available. Please run calculate_efficiency_metrics first.")

        inefficient_users = (
            self.efficiency_df[self.efficiency_df["vram_efficiency"] < efficiency_threshold]
            .groupby("User", observed=False)
            .agg(
            Job_Count=("JobID", "count"),
            Avg_Allocated_VRAM=("allocated_vram", "mean"),
            Total_GPU_Hours=("gpu_hours", "sum"),
            Avg_GPUs=("GPUs", "mean"),
            Avg_Weighted_VRAM_Efficiency=("user_weighted_vram_efficiency", "mean"),
            )
            .reset_index()
        )

        # Multiply share of total gpu hours by weighted vram efficiency to get the new metric
        inefficient_users["Weighted_Efficiency_Contribution"] = (
            inefficient_users["Total_GPU_Hours"]
            * inefficient_users["Avg_Weighted_VRAM_Efficiency"]
            / inefficient_users["Total_GPU_Hours"].sum()
        )


        # Only include users with at least 5 jobs
        inefficient_users = inefficient_users[inefficient_users["Job_Count"] >= 5]

        # Sort by the new metric ascending (lower is worse)
        inefficient_users = inefficient_users.sort_values(
            "Weighted_Efficiency_Contribution",
            ascending=True
        )
        return inefficient_users

    def find_inefficient_pis_weighted_by_hours(self, efficiency_threshold=0.3, min_jobs=5):
        """
        Identify PIs with low average VRAM efficiency across their jobs, weighted by the hours they were inefficient.

        Args:
            efficiency_threshold (float): Threshold for VRAM efficiency to consider a PI as inefficient
            min_jobs (int): Minimum number of jobs a PI must have to be included in the analysis

        Returns:
            pd.DataFrame: DataFrame with PIs and their average VRAM efficiency
        """
        if self.efficiency_df is None:
            raise ValueError("Efficiency DataFrame is not available. Please run calculate_efficiency_metrics first.")

        inefficient_pis = (
            self.efficiency_df[self.efficiency_df["vram_efficiency"] < efficiency_threshold]
            .groupby("Account", observed=False)
            .agg(
                Job_Count=("JobID", "count"),
                Avg_Allocated_VRAM=("allocated_vram", "mean"),
                Total_GPU_Hours=("gpu_hours", "sum"),
                Avg_GPUs=("GPUs", "mean"),
                Avg_Weighted_VRAM_Efficiency=("pi_weighted_vram_efficiency", "mean"),
            )
            .reset_index()
        )

        # Multiply share of total gpu hours by weighted vram efficiency to get the new metric
        inefficient_pis["Weighted_Efficiency_Contribution"] = (
            inefficient_pis["Total_GPU_Hours"]
            * inefficient_pis["Avg_Weighted_VRAM_Efficiency"]
            / inefficient_pis["Total_GPU_Hours"].sum()
        )

        # Only include PIs with at least 5 jobs
        inefficient_pis = inefficient_pis[inefficient_pis["Job_Count"] >= min_jobs]

        # Sort by the new metric ascending (lower is worse)
        inefficient_pis = inefficient_pis.sort_values(
            "Weighted_Efficiency_Contribution",
            ascending=True
        )
        return inefficient_pis
    
    def filter_jobs_by_memory_class(self, memory_gb):
        """
        Given a memory class in GB, filter jobs to only include those that match the GPU types
        associated with that memory class. Also count the number of jobs for each GPU type in that memory class.

        Args:
            memory_gb (int): Memory class in GB to filter jobs by.
        Returns:
            tuple: A tuple containing GPU jobs DataFrame and a dictionary with GPU type counts and their percentages.
        """
        gpu_types_for_mem = [k for k, v in ram_map.items() if v == memory_gb]
        print("PRINTING GPU TYPES")
        print("gpu types", gpu_types_for_mem)
        def gpu_type_matches(x):
            types = x if isinstance(x, list | np.ndarray) else [x]
            return any(str(g).strip().lower() in gpu_types_for_mem for g in types)
        filtered = self.jobs_df[self.jobs_df['GPUType'].apply(gpu_type_matches)]

        # Count jobs for each GPU type in the memory class
        def extract_gpu_types(x):
            types = x if isinstance(x, list | np.ndarray) else [x]
            return [str(g).strip().lower() for g in types if str(g).strip().lower() in gpu_types_for_mem]

        gpu_counts = (
            filtered['GPUType']
            .apply(extract_gpu_types)
            .explode()
            .value_counts()
            .to_dict()
        )

        total_jobs = sum(gpu_counts.values())
        gpu_counts_with_pct = {
            gpu: (count, round(100 * count / total_jobs, 2) if total_jobs > 0 else 0.0)
            for gpu, count in gpu_counts.items()
        }

        return filtered, gpu_counts_with_pct

        
    def aggregate_gpu_metrics_by_type(self, memory_gb=80, show_matplotlib_tables=True):
        """
        Aggregate and display metrics for each GPU type in the specified memory class.
        This method filters jobs based on the specified memory class, calculates various efficiency metrics,
        and displays a summary table of the metrics for each GPU type.
        Args:
            memory_gb (int): Memory class in GB to filter jobs by.
        show_matplotlib_tables (bool): Whether to display the summary table using matplotlib.
        Returns:    
            None
        """
      
        filtered_jobs, gpu_counts = self.filter_jobs_by_memory_class(memory_gb)
        if filtered_jobs.empty:
            print(f"No jobs found for memory class: {memory_gb}GB")
            return

        self.jobs_df = filtered_jobs
        self.calculate_efficiency_metrics()

        unique_gpu_types = gpu_counts.keys()
        if not unique_gpu_types:
            print(f"No GPU types found for memory class: {memory_gb}GB")
            return
        print("printing unique gpu types", unique_gpu_types)
        metrics = [
            "Mean GPU Memory Used (GB)",
            "Median GPU Memory Used (GB)",
            "Mean VRAM Efficiency",
            "Median VRAM Efficiency",
            "Total User GPU Hours",
            "Mean Weighted VRAM Efficiency",
            "Median Weighted VRAM Efficiency"
        ]
        results = {gpu_type.upper(): [] for gpu_type in unique_gpu_types}

        for gpu_type in unique_gpu_types:
            gpu_jobs = self.efficiency_df[self.efficiency_df['GPUType'].apply(
                lambda x, gpu_type=gpu_type: gpu_type in [str(g).strip().lower() for g in (x if isinstance(x, (list, np.ndarray)) else [x])]
            )]
            if gpu_jobs.empty:
                results[gpu_type.upper()] = [None] * len(metrics)
                continue

            results[gpu_type.upper()] = [
                round(gpu_jobs["gpu_memory_used_gb"].mean(), 2),
                round(gpu_jobs["gpu_memory_used_gb"].median(), 2),
                round(gpu_jobs["vram_efficiency"].mean(), 3),
                round(gpu_jobs["vram_efficiency"].median(), 3),
                round(gpu_jobs["gpu_hours"].sum(), 2),
                round(gpu_jobs["user_weighted_vram_efficiency"].mean(), 3),
                round(gpu_jobs["user_weighted_vram_efficiency"].median(), 3)
            ]

        summary_df = pd.DataFrame(results, index=metrics)
        print(f"\n===== Aggregated Metrics Table for {memory_gb}GB GPUs =====")
        print(summary_df)

        if show_matplotlib_tables:
            fig, ax = plt.subplots(figsize=(8, 2 + 0.5 * len(metrics)))
            ax.axis('off')
            table = ax.table(cellText=summary_df.values, rowLabels=summary_df.index, colLabels=summary_df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.5, 1.5)
            ax.set_title(f"Aggregated Metrics for {memory_gb}GB GPUs", fontweight='bold')
            plt.savefig(f"agg_metrics_{memory_gb}GB.png", bbox_inches='tight', dpi=200)
            plt.savefig(f"agg_metrics_{memory_gb}GB_table.png", bbox_inches='tight', dpi=200)
            plt.show()
            # Annotate bars with percentages
            
    def additional_metrics(self, jobs_df=None):
        """
        Add additional VRAM allocation/request metrics and categories to a jobs DataFrame.

        Args:
            jobs_df (pd.DataFrame, optional): DataFrame to compute metrics on. Defaults to self.efficiency_df.

        Returns:
            pd.DataFrame: DataFrame with additional metrics columns.
        """
        if jobs_df is None:
            jobs_df = self.efficiency_df
        metrics = jobs_df.copy()
        metrics["gpu_memory_used_gb"] = metrics['GPUMemUsage'] / (2**30)
        metrics['num_jobs'] = len(metrics)
        metrics['vram_wasted'] = metrics["allocated_vram"] - metrics["gpu_memory_used_gb"]
        metrics["request_accuracy"] = metrics["gpu_memory_used_gb"] / metrics["requested_vram"]
        metrics = metrics[metrics['request_accuracy'].notna() &
    np.isfinite(metrics['request_accuracy'])
]

        metrics["allocation_accuracy"] = metrics["gpu_memory_used_gb"] / metrics["allocated_vram"]
        metrics["request_to_allocation_ratio"] = metrics["allocated_vram"] / metrics["requested_vram"]

        metrics['allocation_efficiency_category'] = pd.cut(
            metrics['allocation_accuracy'],
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['Very Poor (<20%)', 'Poor (20-50%)', 'Fair (50-80%)', 'Good (80-100%)']
        )
        metrics['request_accuracy_category'] = pd.cut(
            metrics['request_accuracy'],
            bins=[0, 0.5, 0.8, 1.2, 2.0, float('inf')],
            labels = ['<20%', '20-50%', '50-80%', '80-100%', '>100%']
        )
        metrics['allocation_type'] = pd.cut(
            metrics['request_to_allocation_ratio'],
            bins=[0, 0.8, 1.0, 1.5, 2.0, float('inf')],
            labels=['Under-allocated (<80%)', 'Exact allocation (80-100%)',
                    'Moderate over-allocation (100-150%)', 'High over-allocation (150-200%)',
                    'Extreme over-allocation (>200%)']
        )
        metrics['request_size_category'] = pd.cut(
            metrics['requested_vram'],
            bins=[0, 8, 16, 32, 64, float('inf')],
            labels=['Small (≤8GB)', 'Medium (8-16GB)', 'Large (16-32GB)',
                    'Very Large (32-64GB)', 'Extreme (>64GB)']
        )
        self.efficiency_df = metrics
        return metrics
    def create_visualizations(self, jobs_df= None):
        sns.set(style="whitegrid", font_scale=1.1)

        fig, ax = plt.subplots(2, 1, figsize=(18, 14))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        if(jobs_df is None):
            jobs_df = self.efficiency_df

        alloc_counts = jobs_df['allocation_efficiency_category'].value_counts().sort_index()
        sns.barplot(x=alloc_counts.index.astype(str), y=alloc_counts.values, ax=ax[0], palette="Blues_d")
        ax[0].set_title('VRAM Allocation Efficiency Distribution')
        ax[0].set_xlabel('Allocation Efficiency Category')
        ax[0].set_ylabel('Number of Jobs')
        ax[0].tick_params(axis='x', rotation=15)

        req_counts = jobs_df['request_accuracy_category'].value_counts().sort_index()
        sns.barplot(x=req_counts.index.astype(str), y=req_counts.values, ax=ax[1], palette="Greens_d")
        ax[1].set_title('Request Accuracy Distribution')
        ax[1].set_xlabel('Request Accuracy Category')
        ax[1].set_ylabel('Number of Jobs')
        ax[1].tick_params(axis='x', rotation=15)

        plt.show()
    def compare_gpu_types_metrics(self, memory_gb=80, efficiency_threshold=0.3, plot_columns=None, save_dir="."):
     
        filtered_jobs, gpu_counts = self.filter_jobs_by_memory_class(memory_gb)
        if filtered_jobs.empty:
            print(f"No jobs found for memory class: {memory_gb}GB")
            return
        self.jobs_df = filtered_jobs
        self.calculate_efficiency_metrics()
        gpu_types = list(gpu_counts.keys())
        if not gpu_types:
            print(f"No GPU types found for memory class: {memory_gb}GB")
            return
        summary = []
        summary_low_eff = []
        total_jobs = len(self.efficiency_df)
        for gpu_type in gpu_types:
            jobs = self.efficiency_df[self.efficiency_df['GPUType'].apply(
                lambda x, gpu_type=gpu_type: gpu_type in [str(g).strip().lower() for g in (x if isinstance(x, (list, np.ndarray)) else [x])]
            )]
            n_jobs = len(jobs)
            pct_jobs = 100 * n_jobs / total_jobs if total_jobs else 0
            gpu_hours = jobs["gpu_hours"].sum()
            mean_eff = jobs["vram_efficiency"].mean()
            median_eff = jobs["vram_efficiency"].median()
            weighted_eff = (jobs["vram_efficiency"] * jobs["gpu_hours"]).sum() / gpu_hours if gpu_hours else 0
            weighted_time_zero = jobs.loc[jobs["vram_efficiency"] == 0, "gpu_hours"].sum()
            summary.append([
                gpu_type.upper(), n_jobs, pct_jobs, gpu_hours, mean_eff, median_eff, weighted_eff, weighted_time_zero
            ])
            low_eff = jobs[jobs["vram_efficiency"] < efficiency_threshold]
            n_jobs_low = len(low_eff)
            pct_jobs_low = 100 * n_jobs_low / n_jobs if n_jobs else 0
            gpu_hours_low = low_eff["gpu_hours"].sum()
            mean_eff_low = low_eff["vram_efficiency"].mean()
            median_eff_low = low_eff["vram_efficiency"].median()
            weighted_eff_low = (low_eff["vram_efficiency"] * low_eff["gpu_hours"]).sum() / gpu_hours_low if gpu_hours_low else 0
            weighted_time_zero_low = low_eff.loc[low_eff["vram_efficiency"] == 0, "gpu_hours"].sum()
            summary_low_eff.append([
                gpu_type.upper(), n_jobs_low, pct_jobs_low, gpu_hours_low, mean_eff_low, median_eff_low, weighted_eff_low, weighted_time_zero_low
            ])
            # Plot distributions for each group and subset
            if plot_columns is not None:
                for col in plot_columns:
                    plt.figure(figsize=(7, 4))
                    plt.hist(jobs[col].dropna(), bins=30, alpha=0.7, label=f"{gpu_type.upper()} All")
                    plt.hist(low_eff[col].dropna(), bins=30, alpha=0.7, label=f"{gpu_type.upper()} <{efficiency_threshold*100:.0f}%")
                    plt.title(f"{col} Distribution for {gpu_type.upper()} ({memory_gb}GB)")
                    plt.xlabel(col)
                    plt.ylabel("Count")
                    plt.legend()
                    fname = os.path.join(save_dir, f"{gpu_type.upper()}_{col}_dist_{memory_gb}GB.png")
                    plt.savefig(fname, bbox_inches='tight', dpi=200)
                    plt.close()
        # Output summary tables
        columns = [
            "GPU Type", "# Jobs", "% Jobs", "Total GPU Hours", "Mean Eff.", "Median Eff.", "Weighted Eff.", "Weighted Time (Eff=0)"
        ]
        df_summary = pd.DataFrame(summary, columns=columns)
        df_summary_low = pd.DataFrame(summary_low_eff, columns=columns)
        print("\n===== Aggregated Metrics by GPU Type =====")
        print(df_summary)
        print(f"\n===== Aggregated Metrics by GPU Type (Eff < {efficiency_threshold*100:.0f}%) =====")
        print(df_summary_low)
        # Save as PNG tables
        for df, tag in zip([df_summary, df_summary_low], ["all", f"loweff_{int(efficiency_threshold*100)}"]):
            fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(df)))
            ax.axis('off')
            table = ax.table(
                cellText=df.values,
                rowLabels=None,
                colLabels=df.columns,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            ax.set_title(f"Aggregated Metrics ({tag}) for {memory_gb}GB GPUs", fontweight='bold')
            plt.savefig(os.path.join(save_dir, f"agg_metrics_{memory_gb}GB_{tag}.png"), bbox_inches='tight', dpi=200)
            plt.close()
    def plot_efficiency_category_comparison(self, memory_gb=80, save_path=None):
        """
        Create a side-by-side table of efficiency categories for A100 and H100 (counts and percentages).
        Save as PNG.
        """
        filtered_jobs, gpu_counts = self.filter_jobs_by_memory_class(memory_gb)
        if filtered_jobs.empty:
            print(f"No jobs found for memory class: {memory_gb}GB")
            return
        self.jobs_df = filtered_jobs
        self.calculate_efficiency_metrics()
        # Only A100 and H100
        gpu_types = [g for g in ["a100", "h100"] if g in gpu_counts]
        if not gpu_types:
            print("No A100 or H100 jobs found.")
            return
        cat_labels = [
            "Very Low (<10%)", "Low (10-30%)", "Medium (30-60%)", "High (60-100%)"
        ]
        data_counts = {}
        data_pcts = {}
        for gpu in gpu_types:
            jobs = self.efficiency_df[self.efficiency_df['GPUType'].apply(
                lambda x, gpu=gpu: gpu in [str(g).strip().lower() for g in (x if isinstance(x, (list, np.ndarray)) else [x])]
            )]
            cat_counts = jobs["efficiency_category"].value_counts().reindex(cat_labels, fill_value=0)
            data_counts[gpu.upper()] = cat_counts.values
            total = cat_counts.sum()
            data_pcts[gpu.upper()] = [f"{(v/total*100):.1f}%" if total else "0.0%" for v in cat_counts.values]
        # Build combined table: counts and percentages
        table_data = []
        for i, cat in enumerate(cat_labels):
            row = []
            for gpu in gpu_types:
                row.append(f"{data_counts[gpu.upper()][i]}\n({data_pcts[gpu.upper()][i]})")
            table_data.append(row)
        # Plot as PNG
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        fig, ax = plt.subplots(figsize=(8, 3 + 0.7 * len(cat_labels)))
        ax.axis('off')
        table = ax.table(
            cellText=table_data,
            rowLabels=cat_labels,
            colLabels=[g.upper() for g in gpu_types],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(2.0, 2.0)
        ax.set_title(f"Efficiency Category Comparison ({memory_gb}GB)", fontweight='bold')
        if save_path is None:
            save_path = f"eff_cat_{memory_gb}GB_comparison.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"Saved efficiency category comparison table to {save_path}")
def filter_zero_vram_requested_with_gpu_allocated(df, requested_vram=0, gpus_min=1):
    """
    Return jobs where requested_vram is greater than or equal to a value (default 0) and GPUs >= gpus_min (default 1).
    
    

        
    def aggregate_metrics_by_gpu(self):
      
        # Load jobs matching the query
        

        if self.jobs_df.empty:
            return pd.DataFrame()  # Return empty DataFrame if no jobs found
        self.calculate_efficiency_metrics()
        unique_gpu_types = (
            self.jobs_df['GPUType']
            .dropna()
            .explode()
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .index
        )
        if not unique_gpu_types.any():
            return
        print("printing unique gpu types", unique_gpu_types)
        metrics = [
            "Mean GPU Memory Used (GB)",
            "Median GPU Memory Used (GB)",
            "Mean VRAM Efficiency",
            "Median VRAM Efficiency",
            "Total User GPU Hours",
            "Mean Weighted VRAM Efficiency",
            "Median Weighted VRAM Efficiency"
        ]
        results = {gpu_type.upper(): [] for gpu_type in unique_gpu_types}
        for gpu_type in unique_gpu_types:
            gpu_jobs = self.efficiency_df[self.efficiency_df['GPUType'].apply(
                lambda x, gpu_type=gpu_type: gpu_type in [str(g).strip().lower() 
                for g in (x if isinstance(x, list | np.ndarray) else [x])]
            )]
            if gpu_jobs.empty:
                results[gpu_type.upper()] = [None] * len(metrics)
                continue
            results[gpu_type.upper()] = [
                round(gpu_jobs["gpu_memory_used_gb"].mean(), 2),
                round(gpu_jobs["gpu_memory_used_gb"].median(), 2),
                round(gpu_jobs["vram_efficiency"].mean(), 3),
                round(gpu_jobs["vram_efficiency"].median(), 3),
                round(gpu_jobs["gpu_hours"].sum(), 2),
                round(gpu_jobs["user_weighted_vram_efficiency"].mean(), 3),
                round(gpu_jobs["user_weighted_vram_efficiency"].median(), 3)
            ]
        summary_df = pd.DataFrame(results, index=metrics)
        return summary_df
            
    def additional_metrics(self):
        """
        Add additional VRAM allocation/request metrics and categories to a jobs DataFrame.

        Args:
            jobs_df (pd.DataFrame, optional): DataFrame to compute metrics on. Defaults to self.efficiency_df.

        Returns:
            pd.DataFrame: DataFrame with additional metrics columns.
        """
        metrics = self.jobs_df.copy()
        metrics["gpu_memory_used_gb"] = metrics['GPUMemUsage'] / (2**30)
        metrics['num_jobs'] = len(metrics)
        metrics['vram_wasted'] = metrics["allocated_vram"] - metrics["gpu_memory_used_gb"]
        metrics["request_accuracy"] = metrics["gpu_memory_used_gb"] / metrics["vram_constraint"]
        metrics = metrics[metrics['request_accuracy'].notna() &
    np.isfinite(metrics['request_accuracy'])
]

        metrics["allocation_accuracy"] = metrics["gpu_memory_used_gb"] / metrics["allocated_vram"]
        metrics["request_to_allocation_ratio"] = metrics["allocated_vram"] / metrics["vram_constraint"]

        metrics['allocation_efficiency_category'] = pd.cut(
            metrics['allocation_accuracy'],
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['Very Poor (<20%)', 'Poor (20-50%)', 'Fair (50-80%)', 'Good (80-100%)']
        )
        metrics['request_accuracy_category'] = pd.cut(
            metrics['request_accuracy'],
            bins=[0, 0.5, 0.8, 1.2, 2.0, float('inf')],
            labels = ['<20%', '20-50%', '50-80%', '80-100%', '>100%']
        )
        metrics['allocation_type'] = pd.cut(
            metrics['request_to_allocation_ratio'],
            bins=[0, 0.8, 1.0, 1.5, 2.0, float('inf')],
            labels=['Under-allocated (<80%)', 'Exact allocation (80-100%)',
                    'Moderate over-allocation (100-150%)', 'High over-allocation (150-200%)',
                    'Extreme over-allocation (>200%)']
        )
        metrics['request_size_category'] = pd.cut(
            metrics['vram_constraint'],
            bins=[0, 8, 16, 32, 64, float('inf')],
            labels=['Small (≤8GB)', 'Medium (8-16GB)', 'Large (16-32GB)',
                    'Very Large (32-64GB)', 'Extreme (>64GB)']
        )
        self.efficiency_df = metrics
        return metrics
    def filter_jobs_for_analysis(
        self,
        vram_constraint_filter: pd.Int64Dtype = None,
        gpu_mem_usage_filter: int = None,
        allocated_vram_filter: int = None,
        gpu_count_filter: int = None,
        elapsed_seconds_min: int = DEFAULT_MIN_ELAPSED_SECONDS,
    ) -> pd.DataFrame:
        """
        Filter jobs based on VRAM constraints, GPU allocation, and usage criteria.

        Args:
            vram_constraint_filter: Filter for vram_constraint column.
            gpu_mem_usage_filter: Filter for GPUMemUsage column.
            allocated_vram_filter: Filter for allocated_vram column.
            gpu_count_filter: Filter for GPUs column.
            elapsed_seconds_min (int): Minimum elapsed time in seconds.

        Returns:
            DataFrame: Filtered jobs DataFrame based on the specified criteria.
        """
        mask = pd.Series([True] * len(self.jobs_df), index=self.jobs_df.index)

        def get_inclusive(filter_val):
            if isinstance(filter_val, dict):
                if "inclusive" not in filter_val or not isinstance(filter_val["inclusive"], bool):
                    raise ValueError("If a filter is a dict, it must include an 'inclusive' boolean key.")
                return filter_val["inclusive"]
            return None

        # vram_constraint
        if vram_constraint_filter is not None:
            col = self.jobs_df["vram_constraint"]
            if callable(vram_constraint_filter):
                mask &= col.apply(vram_constraint_filter)
            elif isinstance(vram_constraint_filter | (list, set, tuple)):
                mask &= col.isin(vram_constraint_filter)
            elif isinstance(vram_constraint_filter, dict):
                inclusive = get_inclusive(vram_constraint_filter)
                if "min" in vram_constraint_filter:
                    mask &= (
                        col.ge(vram_constraint_filter["min"]) if inclusive else col.gt(vram_constraint_filter["min"])
                    )
                if "max" in vram_constraint_filter:
                    mask &= (
                        col.le(vram_constraint_filter["max"]) if inclusive else col.lt(vram_constraint_filter["max"])
                    )
            elif vram_constraint_filter is pd.NA or (
                isinstance(vram_constraint_filter, float) and np.isnan(vram_constraint_filter)
            ):
                mask &= col.isna()
            else:
                mask &= col.eq(vram_constraint_filter)

        # GPU memory usage filter
        if gpu_mem_usage_filter is not None:
            if isinstance(gpu_mem_usage_filter, dict):
                gpu_mem_usage_inclusive = get_inclusive(gpu_mem_usage_filter)
            else:
                gpu_mem_usage_inclusive = None
            mask &= self._apply_numeric_filter(
                self.jobs_df["GPUMemUsage"], gpu_mem_usage_filter, gpu_mem_usage_inclusive
            )

        # Allocated VRAM filter
        if allocated_vram_filter is not None:
            if isinstance(allocated_vram_filter, dict):
                allocated_vram_inclusive = get_inclusive(allocated_vram_filter)
            else:
                allocated_vram_inclusive = None
            mask &= self._apply_numeric_filter(
                self.jobs_df["allocated_vram"], allocated_vram_filter, allocated_vram_inclusive
            )

        # GPU count filter
        if gpu_count_filter is not None:
            if isinstance(gpu_count_filter, dict):
                gpu_count_inclusive = get_inclusive(gpu_count_filter)
            else:
                gpu_count_inclusive = None
            mask &= self._apply_numeric_filter(self.jobs_df["GPUs"], gpu_count_filter, gpu_count_inclusive)

        return self.jobs_df[mask & (self.jobs_df["Elapsed"].dt.total_seconds() >= elapsed_seconds_min)].copy()










def get_top_n_gpus(jobs_df, n):

    gpu_types = jobs_df['GPUType'].dropna().explode()
    # Normalize to string and lowercase for consistency
    gpu_types = gpu_types.astype(str).str.strip().str.lower()
    # Get top n
    top_n = gpu_types.value_counts().head(n).index.tolist()
    return top_n

def contains_a100(gpu_array):
    if isinstance(gpu_array, list | np.ndarray):
        return any(str(gpu).strip().lower() == 'a100' for gpu in gpu_array)
    return False



def filter_a100s(jobs_df):
    a100_jobs_df = jobs_df[jobs_df['GPUType'].apply(contains_a100)]
    return a100_jobs_df



if __name__ == "__main__":
    efficiency = EfficiencyAnalysis()
    # Aggregated metrics table (per-GPU)
    efficiency.aggregate_gpu_metrics_by_type(80)
    # Detailed comparison, low-efficiency subset, and plots
    efficiency.compare_gpu_types_metrics(
        memory_gb=80,
        efficiency_threshold=0.3,
        plot_columns=["vram_efficiency", "allocated_vram", "gpu_hours"]
    )
    # Side-by-side efficiency category table for A100 and H100
    efficiency.plot_efficiency_category_comparison(memory_gb=80)

    # List inefficient PIs and users
    print("\n=== Inefficient PI Groups (weighted by hours, eff < 0.3) ===")
    pis = efficiency.find_inefficient_pis_weighted_by_hours(efficiency_threshold=0.3, min_jobs=5)
    print(pis)

    print("\n=== Inefficient Users (weighted by hours, eff < 0.3) ===")
    users = efficiency.find_inefficient_users_weighted_by_hours(efficiency_threshold=0.3, min_jobs=5)
    print(users)

    with open("inefficient_pis.txt", "w") as f:
        f.write(pis.to_string())
    with open("inefficient_users.txt", "w") as f:
        f.write(users.to_string())
    print("Saved inefficient PI groups to inefficient_pis.txt")
    print("Saved inefficient users to inefficient_users.txt")

    # Aggregate inefficient PIs and users by A100 and H100
    for gpu_type in ["a100", "h100"]:
        jobs_gpu = efficiency.efficiency_df[
            efficiency.efficiency_df['GPUType'].apply(
                lambda x, gpu_type=gpu_type: gpu_type in [
                    str(g).strip().lower()
                    for g in (x if isinstance(x, list | np.ndarray) else [x])
                ]
            )
        ]
        if not jobs_gpu.empty:
            eff_gpu = EfficiencyAnalysis()
            eff_gpu.jobs_df = jobs_gpu
            eff_gpu.efficiency_df = jobs_gpu
            pis_gpu = eff_gpu.find_inefficient_pis_weighted_by_hours(efficiency_threshold=0.3, min_jobs=5)
            users_gpu = eff_gpu.find_inefficient_users_weighted_by_hours(efficiency_threshold=0.3, min_jobs=5)
            with open(f"inefficient_pis_{gpu_type}.txt", "w") as f:
                f.write(pis_gpu.to_string())
            with open(f"inefficient_users_{gpu_type}.txt", "w") as f:
                f.write(users_gpu.to_string())
            print(f"Saved inefficient PI groups for {gpu_type.upper()} to inefficient_pis_{gpu_type}.txt")
            print(f"Saved inefficient users for {gpu_type.upper()} to inefficient_users_{gpu_type}.txt")
        else:
            print(f"No jobs found for {gpu_type.upper()}")