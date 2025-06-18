"""
Functions to analyze jobs that requested no GPUs but got GPU allocations
and used some memory. The aim is to identify potential inefficiencies
in GPU usage and notify users or PIs about these issues.
"""

import pandas as pd
from pathlib import Path
from preprocess.preprocess import preprocess_data
from database.DatabaseConnection import DatabaseConnection


def load_jobs_dataframe_from_duckdb(db_path=None, table_name="Jobs"):
    """
    Connect to the DuckDB slurm_data_small.db and return the jobs table as a pandas DataFrame.

    Args:
        db_path (str or Path, optional): Path to the DuckDB database. Defaults to 'data/slurm_data_small.db'.
        table_name (str, optional): Table name to query. Defaults to 'Jobs'.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    if db_path is None:
        db_path = Path(__file__).resolve().parents[2] / "data" / "slurm_data_small.db"
    db = DatabaseConnection(str(db_path))

    df = db.fetch_all(table_name=table_name)
    # all jobs including CPU only jobs
    data = preprocess_data(df, min_elapsed_second=0, include_failed_cancelled_jobs=False, include_CPU_only_job=True)
    db.disconnect()
    return data


def analyze_hybrid_workload_efficiency(
    df,
    requested_vram_filter=0,
    allocated_vram_greater_than=0,
    gpu_mem_usage_min=None,
    gpu_mem_usage_max=None,
    gpu_mem_usage_exact=None,
    gpus_min=1,
    elapsed_seconds_min=600, #TODO: make it a parameter
):
    """
    Analyze jobs requesting a specific VRAM, GPU allocation, and usage criteria.

    Args:
        df (DataFrame): Main job dataframe from GPUMetrics
        requested_vram_filter (int, float, or callable): Value or function to filter requested_vram
        allocated_vram_greater_than (int, float): allocated_vram greater than this value
        gpu_mem_usage_min (int, float, optional): Minimum GPUMemUsage (inclusive)
        gpu_mem_usage_max (int, float, optional): Maximum GPUMemUsage (inclusive)
        gpu_mem_usage_exact (int, float, optional): If set, select only rows where GPUMemUsage == this value
        gpus_min (int): Minimum GPUs allocated
        elapsed_seconds_min (int): Minimum elapsed time in seconds

    Returns:
        DataFrame: Filtered jobs with efficiency metrics added
    """
    df["Elapsed_seconds"] = df["Elapsed"].dt.total_seconds()
    # Flexible filter for requested_vram
    if callable(requested_vram_filter):
        mask = df["requested_vram"].apply(requested_vram_filter)
    else:
        mask = df["requested_vram"] == requested_vram_filter

    # GPU memory usage filter
    gpu_mem_mask = pd.Series([True] * len(df), index=df.index)
    if gpu_mem_usage_exact is not None:
        gpu_mem_mask &= df["GPUMemUsage"] == gpu_mem_usage_exact
    else:
        if gpu_mem_usage_min is not None:
            gpu_mem_mask &= df["GPUMemUsage"] >= gpu_mem_usage_min
        if gpu_mem_usage_max is not None:
            gpu_mem_mask &= df["GPUMemUsage"] <= gpu_mem_usage_max

    hybrid_jobs = df[
        mask
        & (df["allocated_vram"] > allocated_vram_greater_than)
        & gpu_mem_mask
        & (df["GPUs"] >= gpus_min)
        & (df["Elapsed_seconds"] >= elapsed_seconds_min)
    ].copy()

    # Calculate efficiency metrics
    hybrid_jobs["gpu_memory_used_gb"] = hybrid_jobs["GPUMemUsage"] / (2**30)
    hybrid_jobs["vram_efficiency"] = hybrid_jobs["gpu_memory_used_gb"] / hybrid_jobs["allocated_vram"]
    hybrid_jobs["gpu_hours"] = (hybrid_jobs["Elapsed_seconds"] * hybrid_jobs["GPUs"]) / 3600
    hybrid_jobs["waste_ratio"] = hybrid_jobs["allocated_vram"] / hybrid_jobs["gpu_memory_used_gb"].clip(lower=0.1)

    # Categorize by efficiency
    hybrid_jobs["efficiency_category"] = pd.cut(
        hybrid_jobs["vram_efficiency"],
        bins=[0, 0.1, 0.3, 0.6, 1.0],
        labels=["Very Low (<10%)", "Low (10-30%)", "Medium (30-60%)", "High (60-100%)"],
    )

    # Add CPU memory analysis if available
    if "CPUMemUsage" in df.columns:
        hybrid_jobs["cpu_memory_gb"] = hybrid_jobs["CPUMemUsage"] / (2**30)
        hybrid_jobs["cpu_gpu_ratio"] = hybrid_jobs["cpu_memory_gb"] / hybrid_jobs["gpu_memory_used_gb"].clip(lower=0.1)

    # Duration analysis
    hybrid_jobs["duration_hours"] = hybrid_jobs["Elapsed_seconds"] / 3600
    hybrid_jobs["duration_category"] = pd.cut(
        hybrid_jobs["duration_hours"],
        bins=[0, 1, 6, 24, 48, float("inf")],
        labels=["Short (<1h)", "Medium (1-6h)", "Long (6-24h)", "Under two days (24-48h)", "Over two days (>48h)"],
    )

    return hybrid_jobs


def evaluate_cpu_gpu_balance(hybrid_jobs_df):
    """
    Analyze CPU-GPU balance patterns to identify optimization opportunities.

    Args:
        hybrid_jobs_df (DataFrame): Output from analyze_hybrid_workload_efficiency()

    Returns:
        dict: Analysis results with balance patterns and recommendations
    """
    analysis = {}

    # Overall statistics
    analysis["total_jobs"] = len(hybrid_jobs_df)
    analysis["total_gpu_hours"] = hybrid_jobs_df["gpu_hours"].sum()
    analysis["avg_efficiency"] = hybrid_jobs_df["vram_efficiency"].mean()
    analysis["median_efficiency"] = hybrid_jobs_df["vram_efficiency"].median()

    # Efficiency distribution analysis
    efficiency_analysis = (
        hybrid_jobs_df.groupby("efficiency_category", observed=False)
        .agg(
            {
                "JobID": "count",
                "gpu_hours": "sum",
                "vram_efficiency": "mean",
                "waste_ratio": "mean",
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
        "Avg_Waste_Ratio",
        "Avg_Allocated_GB",
        "Avg_Used_GB",
    ]
    efficiency_analysis["Percentage_of_Hours"] = (
        efficiency_analysis["GPU_Hours"] / analysis["total_gpu_hours"] * 100
    ).round(1)
    analysis["efficiency_patterns"] = efficiency_analysis

    # User behavior analysis (now sorted by lowest efficiency, not highest usage)
    min_jobs = 5  # Only include users with at least 5 jobs to avoid noise
    user_analysis = hybrid_jobs_df.groupby("User").agg(
        {
            "JobID": "count",
            "gpu_hours": "sum",
            "vram_efficiency": "mean",
            "waste_ratio": "mean",
            "allocated_vram": "mean",
        }
    )
    user_analysis.columns = ["Job_Count", "GPU_Hours", "Avg_Efficiency", "Avg_Waste_Ratio", "Avg_Allocated_VRAM"]
    inefficient_users = user_analysis[user_analysis["Job_Count"] >= min_jobs].sort_values(
        "Avg_Efficiency", ascending=True
    )
    analysis["top_inefficient_users"] = inefficient_users.head(20)

    # CPU-GPU balance analysis (if CPU data available)
    if "cpu_gpu_ratio" in hybrid_jobs_df.columns:
        # Categorize workloads by CPU-GPU balance
        hybrid_jobs_df["workload_type"] = pd.cut(
            hybrid_jobs_df["cpu_gpu_ratio"],
            bins=[0, 1, 5, 20, float("inf")],
            labels=[
                "GPU-intensive (CPU<GPU)",
                "Balanced (CPU≈GPU)",
                "CPU-heavy (CPU>GPU)",
                "Very CPU-heavy (CPU>>GPU)",
            ],
        )

        balance_analysis = hybrid_jobs_df.groupby("workload_type", observed=False).agg(
            {"JobID": "count", "gpu_hours": "sum", "vram_efficiency": "mean", "cpu_gpu_ratio": "mean"}
        )
        analysis["cpu_gpu_balance"] = balance_analysis

    # Over-allocation analysis
    high_waste_jobs = hybrid_jobs_df[hybrid_jobs_df["waste_ratio"] > 10]
    analysis["high_waste_jobs"] = len(high_waste_jobs)
    analysis["high_waste_gpu_hours"] = high_waste_jobs["gpu_hours"].sum()
    analysis["high_waste_percentage"] = analysis["high_waste_gpu_hours"] / analysis["total_gpu_hours"] * 100

    # Duration vs efficiency correlation
    duration_efficiency = hybrid_jobs_df.groupby("duration_category", observed=False).agg(
        {"JobID": "count", "vram_efficiency": "mean", "gpu_hours": "sum"}
    )
    analysis["duration_efficiency_patterns"] = duration_efficiency

    # Generate recommendations
    recommendations = []

    low_efficiency_hours = efficiency_analysis.loc[
        efficiency_analysis.index.isin(["Very Low (<10%)", "Low (10-30%)"]), "GPU_Hours"
    ].sum()
    low_efficiency_percentage = low_efficiency_hours / analysis["total_gpu_hours"] * 100

    if low_efficiency_percentage > 50:
        recommendations.append("CRITICAL: >50% of GPU hours have <30% efficiency - immediate optimization needed")
    elif low_efficiency_percentage > 30:
        recommendations.append("HIGH PRIORITY: Significant inefficiency detected - user education campaign needed")

    if analysis["high_waste_percentage"] > 25:
        recommendations.append(
            "MAJOR OVER-ALLOCATION: >25% of jobs have >10x waste ratio - implement allocation limits"
        )

    if analysis["avg_efficiency"] < 0.3:
        recommendations.append(
            "POOR AVERAGE EFFICIENCY: Overall efficiency <30% - systematic resource optimization needed"
        )

    analysis["recommendations"] = recommendations

    return analysis


def filter_zero_vram_requested_with_gpu_allocated(df, requested_vram=0, gpus_min=1):
    """
    Return jobs where requested_vram is greater than or equal to a value (default 0) and GPUs >= gpus_min (default 1).

    Args:
        df (pd.DataFrame): The jobs DataFrame.
        requested_vram (int, float): Value to filter requested_vram
        gpus_min (int): Minimum GPUs allocated

    Returns:
        pd.DataFrame: Filtered DataFrame with jobs matching the criteria.
    """
    return df[(df["requested_vram"] >= requested_vram) & (df["GPUs"] >= gpus_min)]
