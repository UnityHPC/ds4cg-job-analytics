"""
Shared utilities for generating user reports.

This module contains common functions used by both the efficiency analysis
and report generation scripts to avoid code duplication.
"""

import pandas as pd
import subprocess
from typing import Any

from ..config.enum_constants import EfficiencyCategoryEnum, TimeUnitEnum
from src.database import DatabaseConnection


def extract_gpu_type(gpu_type_val: str | list | dict) -> str:
    """
    Extract GPU type from various data formats.

    Args:
        gpu_type_val: GPU type value in various formats

    Returns:
        String representation of the GPU type.
    """
    if isinstance(gpu_type_val, list) and len(gpu_type_val) > 0:
        return gpu_type_val[0]
    elif isinstance(gpu_type_val, dict) and len(gpu_type_val) > 0:
        return list(gpu_type_val.keys())[0]
    elif isinstance(gpu_type_val, str):
        return gpu_type_val
    else:
        return "unknown"


def calculate_analysis_period(jobs_df: pd.DataFrame) -> str:
    """
    Calculate the analysis period based on the job data as a descriptive string.

    Args:
        jobs_df: DataFrame with job data containing StartTime column

    Returns:
        String describing the analysis period
    """
    if "StartTime" not in jobs_df.columns:
        return "All time"

    start_date = jobs_df["StartTime"].min()
    end_date = jobs_df["StartTime"].max()

    if not (pd.notna(start_date) and pd.notna(end_date)):
        return "All time"

    duration = end_date - start_date

    if duration.days > 365:
        years = duration.days // 365
        return f"{years} year{'s' if years > 1 else ''}"
    elif duration.days > 30:
        months = duration.days // 30
        return f"{months} month{'s' if months > 1 else ''}"
    else:
        return f"{duration.days} day{'s' if duration.days > 1 else ''}"


def calculate_time_usage_metrics(user_jobs: pd.DataFrame) -> tuple[float, str]:
    """
    Calculate time usage percentage and estimation category.

    Args:
        user_jobs: DataFrame with user job data

    Returns:
        Tuple of (time_usage_percentage, time_estimate_category)
    """
    avg_elapsed_time = pd.Timedelta(user_jobs["Elapsed"].mean())
    avg_time_limit = (
        pd.Timedelta(user_jobs["TimeLimit"].mean())
        if "TimeLimit" in user_jobs.columns
        else pd.Timedelta(0)
    )

    if avg_time_limit.total_seconds() > 0:
        time_usage_pct = (avg_elapsed_time.total_seconds() / avg_time_limit.total_seconds()) * 100
    else:
        time_usage_pct = 0.0

    # Time estimation category
    if time_usage_pct > 95:
        time_estimate = "Underestimated"
    elif time_usage_pct < 60:
        time_estimate = "Overestimated"
    else:
        time_estimate = "Well estimated"

    return float(time_usage_pct), time_estimate


def calculate_gpu_type_metrics(user_jobs: pd.DataFrame) -> float:
    """
    Calculate A100 usage percentage.

    Args:
        user_jobs: DataFrame with user job data

    Returns:
        A100 usage percentage
    """
    if "GPUType" not in user_jobs.columns:
        return 0.0

    total_jobs = len(user_jobs)
    if total_jobs == 0:
        return 0.0

    user_jobs_copy = user_jobs.copy()
    user_jobs_copy["primary_gpu_type"] = user_jobs_copy["GPUType"].apply(extract_gpu_type)
    a100_count = user_jobs_copy["primary_gpu_type"].str.contains("a100", case=False, na=False).sum()
    a100_pct = (a100_count / total_jobs) * 100

    return float(a100_pct)


def calculate_cpu_memory_metrics(user_jobs: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Calculate CPU memory related metrics.

    Args:
        user_jobs: DataFrame with user job data

    Returns:
        Tuple of (avg_cpu_mem_req, avg_cpu_mem_used, cpu_mem_usage_pct, cpu_gpu_ratio)
    """
    avg_cpu_mem_req = user_jobs["allocated_cpu_mem_gib"].mean() if "allocated_cpu_mem_gib" in user_jobs.columns else 0
    avg_cpu_mem_used = user_jobs["used_cpu_mem_gib"].mean() if "used_cpu_mem_gib" in user_jobs.columns else 0
    avg_vram_used_gb = user_jobs["used_vram_gib"].mean()

    cpu_mem_usage_pct = (avg_cpu_mem_used / avg_cpu_mem_req) * 100 if avg_cpu_mem_req > 0 else 0
    cpu_gpu_ratio = avg_cpu_mem_used / avg_vram_used_gb if avg_vram_used_gb > 0 else 0

    return float(avg_cpu_mem_req), float(avg_cpu_mem_used), float(cpu_mem_usage_pct), float(cpu_gpu_ratio)


def calculate_zero_usage_metrics(user_jobs: pd.DataFrame) -> tuple[int, float]:
    """
    Calculate zero GPU usage job metrics.

    Args:
        user_jobs: DataFrame with user job data

    Returns:
        Tuple of (zero_usage_jobs_count, zero_usage_percentage)
    """
    total_jobs = len(user_jobs)
    zero_usage_jobs = len(user_jobs[user_jobs["used_vram_gib"] < 0.1])
    zero_usage_pct = (zero_usage_jobs / total_jobs) * 100 if total_jobs > 0 else 0

    return int(zero_usage_jobs), float(zero_usage_pct)


def calculate_summary_statistics(
    user_jobs: pd.DataFrame,
    all_jobs_count: int = 0,
    analysis_period: str = ''
) -> pd.DataFrame:
    """
    Calculate comprehensive summary statistics for a user.

    Args:
        user_jobs: DataFrame with user job data
        all_jobs_count: Total number of jobs in the system (for percentage calculation)
        analysis_period: Analysis period string (if not provided, will be calculated)

    Returns:
        DataFrame with summary statistics
    """
    # Basic metrics
    total_jobs = len(user_jobs)
    avg_vram_requested = user_jobs["allocated_vram"].mean()
    avg_vram_used_gb = user_jobs["used_vram_gib"].mean()
    vram_efficiency = (
        user_jobs["requested_vram_efficiency"].mean() 
        if "requested_vram_efficiency" in user_jobs.columns 
        else 0
    )

    # Calculate job percentage
    if all_jobs_count == 0:
        job_percentage = 0.0  # Cannot calculate without total
    else:
        job_percentage = (total_jobs / all_jobs_count) * 100 if all_jobs_count > 0 else 0

    # Calculate analysis period if not provided
    if analysis_period == '':
        analysis_period = calculate_analysis_period(user_jobs)

    # Time metrics
    time_usage_pct, time_estimate = calculate_time_usage_metrics(user_jobs)

    # GPU type metrics
    a100_pct = calculate_gpu_type_metrics(user_jobs)

    # CPU memory metrics
    avg_cpu_mem_req, avg_cpu_mem_used, cpu_mem_usage_pct, cpu_gpu_ratio = calculate_cpu_memory_metrics(user_jobs)

    # Zero usage metrics
    zero_usage_jobs, zero_usage_pct = calculate_zero_usage_metrics(user_jobs)

    # Efficiency category
    efficiency_category = EfficiencyCategoryEnum.get_efficiency_category(vram_efficiency)

    # Create summary statistics DataFrame
    summary_stats = pd.DataFrame({
        "Metric": [
            "Total number of GPU jobs",
            f"Percentage of all jobs in {analysis_period}",
            "Average GPU VRAM requested (GiB)",
            "Average GPU VRAM used (GiB)",
            "Average GPU VRAM efficiency",
            "A100 usage percentage",
            "Average CPU memory requested (GiB)",
            "Average CPU memory used (GiB)",
            "CPU memory usage efficiency",
            "CPU/GPU memory ratio",
            "Zero GPU usage jobs",
            "Efficiency category",
            "Time estimation",
        ],
        "Value": [
            f"{total_jobs:,}",
            f"{job_percentage:.4f}%",
            f"{avg_vram_requested:.2f}",
            f"{avg_vram_used_gb:.2f}",
            f"{vram_efficiency:.2f} ({vram_efficiency * 100:.2f}%)",
            f"{a100_pct:.2f}%",
            f"{avg_cpu_mem_req:.2f}",
            f"{avg_cpu_mem_used:.2f}",
            f"{cpu_mem_usage_pct:.2f}%",
            f"{cpu_gpu_ratio:.2f}",
            f"{zero_usage_jobs} ({zero_usage_pct:.2f}%)" if zero_usage_jobs > 0 else "0",
            efficiency_category,
            time_estimate,
        ],
    })

    return summary_stats


def calculate_comparison_statistics(
    user_id: str,
    user_jobs: pd.DataFrame,
    all_jobs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate comparison statistics between the user and all other users.

    Args:
        user_id: The user ID
        user_jobs: DataFrame with user job data
        all_jobs_df: DataFrame with all jobs data

    Returns:
        DataFrame with comparison statistics
    """
    # Calculate averages for all other users from the jobs data (not user aggregates)
    other_users_jobs = all_jobs_df[all_jobs_df["User"] != user_id]

    if len(other_users_jobs) > 0:
        # Use requested_vram_efficiency from individual jobs, not expected values
        avg_vram_eff_others = (
            other_users_jobs["requested_vram_efficiency"].mean() * 100
            if "requested_vram_efficiency" in other_users_jobs.columns
            else 30.0
        )
        avg_vram_used_others = other_users_jobs["used_vram_gib"].sum()

        if "TimeLimit" in other_users_jobs.columns:
            avg_time_usage_others = (
                other_users_jobs["Elapsed"].dt.total_seconds() /
                other_users_jobs["TimeLimit"].dt.total_seconds()
            ).mean() * 100
        else:
            avg_time_usage_others = 75.0
    else:
        avg_vram_eff_others = 30.0
        avg_vram_used_others = user_jobs["used_vram_gib"].sum() * 0.8
        avg_time_usage_others = 75.0

    # User metrics - use requested_vram_efficiency from individual jobs
    vram_efficiency = (
        user_jobs["requested_vram_efficiency"].mean() * 100
        if "requested_vram_efficiency" in user_jobs.columns
        else 30.0
    )
    time_usage_pct, _ = calculate_time_usage_metrics(user_jobs)

    comparison_stats = pd.DataFrame({
        "Category": ["VRAM Efficiency", "Time Usage", "Total GPU Memory"],
        "Your_Value": [
            vram_efficiency,
            time_usage_pct,
            user_jobs["used_vram_gib"].sum()
        ],
        "Average_Value": [
            avg_vram_eff_others,
            avg_time_usage_others,
            avg_vram_used_others
        ]
    })

    return comparison_stats


def calculate_time_series_data(user_jobs: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Calculate time series data for the user's jobs.

    Args:
        user_jobs: DataFrame with user job data

    Returns:
        List of time series data dictionaries
    """
    time_series_data = []
    try:
        # Import here to avoid circular import
        from ..analysis.frequency_analysis import FrequencyAnalysis

        frequency_analyzer = FrequencyAnalysis(user_jobs)
        monthly_data = frequency_analyzer.prepare_time_series_data(
            users=[user_jobs["User"].iloc[0]] if len(user_jobs) > 0 else [],
            metric="requested_vram_efficiency",
            time_unit=TimeUnitEnum.MONTHS,
            remove_zero_values=False
        )

        for _, row in monthly_data.iterrows():
            # Handle NaN values safely
            efficiency = row["requested_vram_efficiency"]
            vram_hours = row["GPUHours"]
            job_count = row["JobCount"]

            time_series_data.append({
                "period": row["TimeGroup"].strftime("%Y-%m"),
                "job_count": int(job_count) if pd.notna(job_count) else 0,
                "efficiency": float(efficiency) if pd.notna(efficiency) else 0.0,
                "vram_hours": float(vram_hours) if pd.notna(vram_hours) else 0.0,
                "metric_type": "monthly_summary"
            })
    except Exception as e:
        print(f"Error generating time series data: {e}")

    return time_series_data


def calculate_gpu_type_data(user_jobs: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Calculate GPU type distribution data.

    Args:
        user_jobs: DataFrame with user job data

    Returns:
        List of GPU type data dictionaries
    """
    gpu_type_data = []
    if 'GPUType' in user_jobs.columns:
        try:
            user_jobs_copy = user_jobs.copy()
            user_jobs_copy['gpu_type_clean'] = user_jobs_copy['GPUType'].apply(extract_gpu_type)
            gpu_counts = user_jobs_copy['gpu_type_clean'].value_counts()

            for gpu_type, count in gpu_counts.items():
                gpu_type_data.append({
                    "gpu_type": gpu_type,
                    "job_count": int(count),
                    "percentage": float(count / len(user_jobs) * 100)
                })
        except Exception as e:
            print(f"Error processing GPU type data: {e}")

    return gpu_type_data


def generate_recommendations(user_jobs: pd.DataFrame, user_data: pd.Series = None) -> list[str]:
    """
    Generate personalized recommendations for the user.

    Args:
        user_jobs: DataFrame with user job data
        user_data: Optional user-level metrics

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # VRAM efficiency recommendations
    vram_efficiency = (
        user_jobs["requested_vram_efficiency"].mean()
        if "requested_vram_efficiency" in user_jobs.columns
        else 0.3  # Default fallback
    )
    if vram_efficiency < EfficiencyCategoryEnum.LOW_THRESHOLD.value:
        recommendations.append(
            "**Optimize VRAM Usage**: Your GPU memory utilization is low. Consider using smaller models, reducing "
            "batch sizes, or using mixed precision to more efficiently use allocated GPU memory."
        )

    # Time allocation recommendations
    time_usage_pct, _ = calculate_time_usage_metrics(user_jobs)
    if time_usage_pct < 50:
        recommendations.append(
            "**Adjust Time Limits**: You're using significantly less time than requested. Consider reducing your "
            "job time limits to improve job scheduling and resource allocation."
        )
    elif time_usage_pct > 95:
        recommendations.append(
            "**Increase Time Limits**: Your jobs are using almost all allocated time. Consider increasing time limits "
            "slightly to prevent job termination."
        )

    # CPU/GPU balance recommendations
    _, _, _, cpu_gpu_ratio = calculate_cpu_memory_metrics(user_jobs)
    if cpu_gpu_ratio > 2.0:
        recommendations.append(
            "**Review CPU vs GPU Workload**: Your CPU memory usage is significantly higher than GPU memory usage. "
            "Some of your workloads might be more suitable for CPU-only jobs, or you might need to optimize your "
            "code to offload more computation to GPUs."
        )

    # A100 recommendations
    a100_pct = calculate_gpu_type_metrics(user_jobs)
    if a100_pct > 60 and vram_efficiency < 0.4:
        recommendations.append(
            "**Reconsider A100 Usage**: You're using A100 GPUs but with relatively low memory efficiency. "
            "Consider using other GPU types for jobs that don't require A100's specific capabilities "
            "or memory capacity."
        )

    # Zero usage recommendations
    zero_usage_jobs, zero_usage_pct = calculate_zero_usage_metrics(user_jobs)
    if zero_usage_jobs > 0 and zero_usage_pct > 10:
        recommendations.append(
            "**Check Zero-Usage Jobs**: A significant portion of your jobs are not using GPU memory. "
            "Verify that these jobs actually need GPUs or if there might be an issue with your code."
        )

    # Requested VRAM efficiency (if user_data is provided)
    eff = user_data["requested_vram_efficiency"]
    if user_data is not None and (eff.mean() if hasattr(eff, 'mean') else float(eff)) < 0.5:
        recommendations.append(
            "📊 **Resource Planning**: You consistently request more VRAM than you use. "
            "Consider reducing your VRAM requests."
        )

    if not recommendations:
        recommendations = ["Your GPU usage patterns are generally efficient. Keep up the good work!"]

    return recommendations


def run_quarto_report(
    template_file: str,
    output_file: str,
    pickle_file: str,
    output_format: str = "html",
    working_directory: str = "reports"
) -> tuple[bool, str]:
    """
    Run Quarto to generate a report.

    Args:
        template_file: Path to the Quarto template file
        output_file: Output file name
        pickle_file: Path to the pickle file with data
        output_format: Output format ('html' or 'pdf')
        working_directory: Working directory for Quarto execution

    Returns:
        Tuple of (success, error_message)
    """
    cmd = ["quarto", "render", template_file, "--to", output_format, "-o", output_file, "--execute"]
    cmd.extend(["-P", f"pickle_file:{pickle_file}"])

    try:
        print("Starting Quarto rendering...")
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=working_directory,
        )

        if result.returncode != 0:
            error_msg = f"Quarto rendering failed with return code {result.returncode}\n"
            error_msg += f"STDOUT: {result.stdout}\n"
            error_msg += f"STDERR: {result.stderr}"
            return False, error_msg

        return True, ""

    except subprocess.CalledProcessError as e:
        return False, f"Quarto execution failed: {e}"
    except Exception as e:
        return False, f"Unexpected error during Quarto rendering: {e}"


def get_total_job_count(db: DatabaseConnection, analysis_period: tuple[str, str] = None) -> int:
    """
    Get the total count of all jobs in the database, optionally filtered by analysis period.

    Args:
        db (DatabaseConnection): Database connection object
        analysis_period (tuple[str, str]): Optional tuple (start_date, end_date) as strings 'YYYY-MM-DD'

    Returns:
        int: total number of jobs in the database (optionally within the period)
    """
    try:
        if analysis_period and isinstance(analysis_period, tuple) and len(analysis_period) == 2:
            start_date, end_date = analysis_period
            query = """
                SELECT COUNT(*) as total_jobs FROM Jobs
                WHERE StartTime >= ? AND StartTime <= ?;
            """
            result = db.fetch_query(query, (start_date, end_date))
        else:
            result = db.fetch_query("SELECT COUNT(*) as total_jobs FROM Jobs;")
        return result.iloc[0]['total_jobs'] if len(result) > 0 else 0
    except Exception as e:
        print(f"Error getting total job count: {e}")
        return 0
