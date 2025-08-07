"""
Generate GPU usage reports for inefficient users using Quarto.

This script analyzes GPU job data, identifies inefficient users, and generates
personalized HTML reports with usage statistics and recommendations.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# Add the project root directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path setup
from src.analysis.efficiency_analysis import EfficiencyAnalysis  # noqa: E402
from src.database.database_connection import DatabaseConnection  # noqa: E402
from src.preprocess.preprocess import preprocess_data  # noqa: E402


def load_and_preprocess_data(db_path: str, specific_users: list = None) -> pd.DataFrame:
    """
    Load and preprocess the job data from the database.

    Args:
        db_path: Path to the database file
        specific_users: Optional list of specific users to filter by

    Returns:
        Preprocessed DataFrame with job data
    """
    print(f"Loading data from {db_path}...")

    # Connect to the database
    try:
        db = DatabaseConnection(db_path)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return pd.DataFrame()

    # Query jobs - filter by specific users if provided
    try:
        if specific_users and len(specific_users) > 0:
            # Create quoted strings for each user
            quoted_users = [f"'{user}'" for user in specific_users]
            users_list = ", ".join(quoted_users)
            query = f"SELECT * FROM Jobs WHERE GPUs > 0 AND User IN ({users_list});"
            job_df = db.fetch_query(query)
        else:
            job_df = db.fetch_query("SELECT * FROM Jobs WHERE GPUs > 0;")
    except Exception as e:
        print(f"Error executing database query: {e}")
        return pd.DataFrame()

    # Preprocess the data
    processed_df = preprocess_data(
        job_df, min_elapsed_seconds=0, include_failed_cancelled_jobs=False, include_cpu_only_jobs=False
    )

    print(f"Loaded and preprocessed {len(processed_df)} jobs.")
    return processed_df


def identify_inefficient_users(
    jobs_df: pd.DataFrame, efficiency_threshold: float = 0.3, min_jobs: int = 10, top_n: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identify the most inefficient users based on VRAM usage.

    Args:
        jobs_df: Preprocessed job DataFrame
        efficiency_threshold: Maximum efficiency threshold for inefficient users
        min_jobs: Minimum number of jobs a user must have
        top_n: Number of top inefficient users to return

    Returns:
        Tuple of (inefficient_users, job_metrics, user_metrics)
    """
    print(f"Identifying inefficient users (efficiency < {efficiency_threshold}, min jobs = {min_jobs})...")

    # Initialize the efficiency analyzer
    analyzer = EfficiencyAnalysis(jobs_df=jobs_df)

    # Filter jobs for analysis (only GPU jobs with allocated VRAM > 0)
    filtered_jobs = analyzer.filter_jobs_for_analysis(
        gpu_count_filter={"min": 1, "max": np.inf, "inclusive": True},
        vram_constraint_filter=None,
        allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
        gpu_mem_usage_filter=None,
    )

    print(f"Filtered {len(filtered_jobs)} jobs for analysis out of {len(jobs_df)} total jobs")

    # Calculate job efficiency metrics - this adds all the efficiency columns
    job_metrics = analyzer.calculate_job_efficiency_metrics(filtered_jobs=filtered_jobs)

    # Calculate user efficiency metrics
    user_metrics = analyzer.calculate_user_efficiency_metrics()

    # Find inefficient users
    inefficient_users = analyzer.find_inefficient_users_by_alloc_vram_efficiency(
        alloc_vram_efficiency_filter={"min": 0, "max": efficiency_threshold, "inclusive": True}, min_jobs=min_jobs
    )

    # Return top N inefficient users
    top_users = inefficient_users.head(top_n)

    print(f"Found {len(top_users)} inefficient users.")
    return top_users, job_metrics, user_metrics


def calculate_analysis_period(jobs_df: pd.DataFrame) -> str:
    """
    Calculate the analysis period based on the job data.

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
        years = duration.days / 365
        return f"Last {years:.1f} years"
    elif duration.days > 30:
        months = duration.days / 30
        return f"Last {months:.1f} months"
    else:
        return f"Last {duration.days} days"


def generate_user_report(
    user_id: str,
    user_data: pd.DataFrame,
    job_data: pd.DataFrame,
    all_users_data: pd.DataFrame,
    output_dir: str,
    template_path: str,
) -> str:
    """
    Generate a report for a specific user using Quarto.

    Args:
        user_id: User ID to generate report for
        user_data: DataFrame row with user metrics
        job_data: DataFrame with all job metrics
        all_users_data: DataFrame with all users metrics
        output_dir: Directory to save the report
        template_path: Path to the Quarto template file

    Returns:
        Path to the generated report
    """
    print(f"Generating report for user {user_id}...")

    # Filter jobs for this specific user
    user_jobs = job_data[job_data["User"] == user_id].copy()

    if len(user_jobs) == 0:
        print(f"No jobs found for user {user_id}")
        return None

    # Create output filename
    output_filename = f"user_{user_id}_report.html"
    output_file = os.path.join(output_dir, output_filename)

    # Check that the template exists
    if not os.path.exists(template_path):
        abs_template_path = os.path.abspath(template_path)
        if os.path.exists(abs_template_path):
            template_path = abs_template_path
        else:
            print(f"Error: Template file not found at: {template_path}")
            print(f"Absolute path also not found: {abs_template_path}")
            return None

    # Calculate dates for the report
    start_date = job_data["StartTime"].min().strftime("%Y-%m-%d")
    end_date = job_data["StartTime"].max().strftime("%Y-%m-%d")

    # Calculate analysis period
    analysis_period = calculate_analysis_period(job_data)

    # === CALCULATE SUMMARY STATISTICS ===
    total_jobs = len(user_jobs)
    total_all_jobs = len(job_data)
    job_percentage = (total_jobs / total_all_jobs) * 100 if total_all_jobs > 0 else 0

    # VRAM metrics
    avg_vram_requested = user_jobs["allocated_vram"].mean()
    avg_vram_used_gb = user_jobs["used_vram_gib"].mean()
    vram_usage_pct = (avg_vram_used_gb / avg_vram_requested) * 100 if avg_vram_requested > 0 else 0
    vram_efficiency = user_jobs["alloc_vram_efficiency"].mean() if "alloc_vram_efficiency" in user_jobs.columns else 0

    # GPU Type analysis
    def extract_gpu_type(gpu_type_val: str | list | dict) -> str:
        """
        Extract GPU type from various data formats.

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

    if "GPUType" in user_jobs.columns:
        user_jobs["primary_gpu_type"] = user_jobs["GPUType"].apply(extract_gpu_type)
        modal_gpu_type = user_jobs["primary_gpu_type"].mode()[0] if len(user_jobs) > 0 else "unknown"
        a100_count = user_jobs["primary_gpu_type"].str.contains("a100", case=False, na=False).sum()
        a100_pct = (a100_count / total_jobs) * 100 if total_jobs > 0 else 0
    else:
        modal_gpu_type = "unknown"
        a100_pct = 0

    # Job status
    modal_job_status = (
        user_jobs["Status"].mode()[0] if len(user_jobs) > 0 and "Status" in user_jobs.columns else "unknown"
    )

    # Time metrics
    avg_time_limit = user_jobs["TimeLimit"].mean() if "TimeLimit" in user_jobs.columns else pd.Timedelta(0)
    avg_elapsed_time = user_jobs["Elapsed"].mean() if "Elapsed" in user_jobs.columns else pd.Timedelta(0)
    time_usage_pct = (avg_elapsed_time / avg_time_limit) * 100 if avg_time_limit.total_seconds() > 0 else 0

    # CPU memory metrics
    avg_cpu_mem_req = user_jobs["allocated_cpu_mem_gib"].mean() if "allocated_cpu_mem_gib" in user_jobs.columns else 0
    avg_cpu_mem_used = user_jobs["used_cpu_mem_gib"].mean() if "used_cpu_mem_gib" in user_jobs.columns else 0
    cpu_mem_usage_pct = (avg_cpu_mem_used / avg_cpu_mem_req) * 100 if avg_cpu_mem_req > 0 else 0

    # CPU/GPU memory ratio
    cpu_gpu_ratio = avg_cpu_mem_used / avg_vram_used_gb if avg_vram_used_gb > 0 else 0

    # Zero usage jobs
    zero_usage_jobs = len(user_jobs[user_jobs["used_vram_gib"] < 0.1])
    zero_usage_pct = (zero_usage_jobs / total_jobs) * 100 if total_jobs > 0 else 0

    # Efficiency category
    if vram_efficiency < 0.10:
        efficiency_category = "Low Efficiency"
    elif vram_efficiency < 0.30:
        efficiency_category = "Moderate Efficiency"
    elif vram_efficiency < 0.70:
        efficiency_category = "Good Efficiency"
    else:
        efficiency_category = "Excellent Efficiency"

    # Time estimation
    if time_usage_pct > 95:
        time_estimate = "underestimated"
    elif time_usage_pct < 60:
        time_estimate = "overestimated"
    else:
        time_estimate = "well estimated"

    # Create summary statistics DataFrame
    summary_stats = pd.DataFrame({
        "Metric": [
            "Total number of jobs",
            f"Percentage of all jobs in {analysis_period}",
            "Average GPU VRAM requested (GiB)",
            "Average GPU VRAM used (GiB)",
            "Average VRAM efficiency",
            "Modal GPU Type",
            "A100 usage percentage",
            "Modal Job Status",
            "Average time limit",
            "Average elapsed time",
            "Time usage efficiency",
            "Average CPU memory requested (GiB)",
            "Average CPU memory used (GiB)",
            "CPU memory usage efficiency",
            "CPU/GPU memory ratio",
            "Zero usage jobs",
            "Efficiency category",
            "Time estimation",
        ],
        "Value": [
            f"{total_jobs:,}",
            f"{job_percentage:.2f}%",
            f"{avg_vram_requested:.2f}",
            f"{avg_vram_used_gb:.2f} ({vram_usage_pct:.2f}%)",
            f"{vram_efficiency:.2f} ({vram_efficiency * 100:.2f}%)",
            f"{modal_gpu_type} ({a100_pct:.2f}% A100)" if a100_pct > 0 else modal_gpu_type,
            f"{a100_pct:.2f}%",
            modal_job_status,
            str(avg_time_limit),
            f"{avg_elapsed_time} ({time_usage_pct:.2f}%)",
            f"{time_usage_pct:.2f}%",
            f"{avg_cpu_mem_req:.2f}",
            f"{avg_cpu_mem_used:.2f} ({cpu_mem_usage_pct:.2f}%)",
            f"{cpu_mem_usage_pct:.2f}%",
            f"{cpu_gpu_ratio:.2f}",
            f"{zero_usage_jobs} ({zero_usage_pct:.2f}%)" if zero_usage_jobs > 0 else "0",
            efficiency_category,
            time_estimate,
        ],
    })

    # === CALCULATE COMPARISON STATISTICS ===
    # Calculate all-user averages for comparison
    all_avg_vram_efficiency = all_users_data["expected_value_alloc_vram_efficiency"].mean() * 100
    all_avg_elapsed_pct = (
        job_data["Elapsed"].dt.total_seconds() / job_data["TimeLimit"].dt.total_seconds()
    ).mean() * 100
    all_avg_total_vram = job_data.groupby("User")["used_vram_gib"].sum().mean()
    user_total_vram = user_jobs["used_vram_gib"].sum()

    comparison_stats = pd.DataFrame({
        "Category": ["VRAM Efficiency", "Time Usage", "Total GPU Memory"],
        "Your_Value": [vram_efficiency * 100, time_usage_pct, user_total_vram],
        "Average_Value": [all_avg_vram_efficiency, all_avg_elapsed_pct, all_avg_total_vram],
    })

    # === PREPARE TIME SERIES DATA ===
    from src.analysis.frequency_analysis import FrequencyAnalysis
    from src.config.enum_constants import TimeUnitEnum

    try:
        freq_analyzer = FrequencyAnalysis(job_data)
        time_series_data = freq_analyzer.prepare_time_series_data(
            users=[user_id], metric="alloc_vram_efficiency", time_unit=TimeUnitEnum.WEEKS, remove_zero_values=False
        )
    except Exception as e:
        print(f"Warning: Could not prepare time series data: {e}")
        time_series_data = pd.DataFrame()

    # === PREPARE GPU TYPE DATA ===
    if "GPUType" in user_jobs.columns and "primary_gpu_type" in user_jobs.columns:
        gpu_type_data = user_jobs["primary_gpu_type"].value_counts().reset_index()
        gpu_type_data.columns = ["gpu_type", "job_count"]
    else:
        gpu_type_data = pd.DataFrame(columns=["gpu_type", "job_count"])

    # === GENERATE RECOMMENDATIONS ===
    recommendations = []

    # VRAM efficiency recommendations
    if vram_efficiency < 0.3:
        recommendations.append(
            "**Optimize VRAM Usage**: Your GPU memory utilization is low. Consider using smaller models, reducing "
            "batch sizes, or using mixed precision to more efficiently use allocated GPU memory."
        )

    # Time allocation recommendations
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
    if cpu_gpu_ratio > 2.0:
        recommendations.append(
            "**Review CPU vs GPU Workload**: Your CPU memory usage is significantly higher than GPU memory usage. Some"
            "of your workloads might be more suitable for CPU-only jobs, or you might need to optimize your code to "
            "offload more computation to GPUs."
        )

    # A100 recommendations
    if a100_pct > 60 and vram_efficiency < 0.4:
        recommendations.append(
            "**Reconsider A100 Usage**: You're using A100 GPUs but with relatively low memory efficiency. "
            "Consider using other GPU types for jobs that don't require A100's specific capabilities "
            "or memory capacity."
        )

    # Zero usage recommendations
    if zero_usage_jobs > 0 and zero_usage_pct > 10:
        recommendations.append(
            "**Check Zero-Usage Jobs**: A significant portion of your jobs are not using GPU memory. "
            "Verify that these jobs actually need GPUs or if there might be an issue with your code."
        )

    if not recommendations:
        recommendations = ["Your GPU usage patterns are generally efficient. Keep up the good work!"]

    # Create a different output directory for each report to avoid conflicts
    user_output_dir = os.path.join(output_dir, f"user_{user_id}")
    os.makedirs(user_output_dir, exist_ok=True)

    # Use a simple output filename
    simple_output_filename = "report.html"
    final_output_path = os.path.join(output_dir, output_filename)

    # Create a parameters dictionary to pass to Quarto (used later)

    # Get template filename for running from reports directory
    template_filename = os.path.basename(template_path)

    # Run quarto render command with execute-params
    cmd = ["quarto", "render", template_filename, "--to", "html", "-o", simple_output_filename, "--execute"]

    # Save the DataFrames to temporary CSV files for Quarto to load
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Check if temp directory exists
    if not os.path.exists(temp_dir):
        print(f"Error: Temp directory does not exist: {temp_dir}")
        return None

    # === SAVE DATA TO JSON FOR QUARTO ===

    # Create a temporary JSON file with all the data
    temp_data = {
        "user_id": user_id,
        "start_date": start_date,
        "end_date": end_date,
        "analysis_period": analysis_period,
        "summary_stats": summary_stats.to_dict("records"),
        "comparison_stats": comparison_stats.to_dict("records"),
        "time_series_data": time_series_data.to_dict("records"),
        "gpu_type_data": gpu_type_data.to_dict("records"),
        "recommendations": recommendations,
        "user_data": user_data.to_dict() if hasattr(user_data, "to_dict") else {},
        "user_jobs": user_jobs.to_dict("records"),
    }

    # Save to a temporary JSON file
    temp_json_file = os.path.join(temp_dir, f"{user_id}_data.json")
    with open(temp_json_file, "w") as f:
        json.dump(temp_data, f, default=str)  # default=str handles datetime objects

    # Create a separate working directory for this user's report
    import shutil

    # get abs path to the template in the reports folder
    template_file = os.path.join(project_root, "reports", "user_report_template.qmd")

    # Use simple filenames in the working directory
    data_file = f"{user_id}_data.json"

    # Update command to use the working template
    cmd = ["quarto", "render", template_file, "--to", "html", "-o", simple_output_filename, "--execute"]

    # Add parameters to quarto command
    cmd.extend(["-P", f"data_file:{data_file}"])  # Execute the command
    try:
        # Run Quarto in the working directory
        print("CMD\n", " ".join(cmd))
        # Run from the reports directory to avoid path issues
        reports_dir = os.path.dirname(template_path)
        result = subprocess.run(
            cmd,
            check=False,  # Don't raise exception, handle it manually
            capture_output=True,
            text=True,
            cwd=reports_dir,
        )

        # Check if command was successful
        if result.returncode != 0:
            print(f"Error running Quarto command (return code {result.returncode})")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return None

        # Check if output file was created (in the reports directory)
        reports_dir = os.path.dirname(template_path)
        output_file_path = os.path.join(reports_dir, simple_output_filename)
        if not os.path.exists(output_file_path):
            print(f"Error: Output file not created at {output_file_path}")
            return None

        # Copy the generated report to final location
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        shutil.copy2(output_file_path, final_output_path)
        print(f"Report generated for {user_id}: {final_output_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error generating report for {user_id}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error generating report for {user_id}: {e}")
        return None

    # Clean up temporary files
    import contextlib

    with contextlib.suppress(Exception):
        os.remove(temp_json_file)

    return output_file


def check_directory_writable(directory_path: str) -> tuple[bool, str | None]:
    """
    Check if a directory exists and is writable. Create it if it doesn't exist.

    Args:
        directory_path: Path to the directory to check

    Returns:
        Tuple of (bool, str) indicating success and an error message if applicable
    """
    try:
        # Make sure the directory exists
        os.makedirs(directory_path, exist_ok=True)

        # Check if we can write to it
        test_file = os.path.join(directory_path, "__write_test.tmp")
        with open(test_file, "w") as f:
            f.write("test")

        # Clean up
        os.remove(test_file)
        return True, None
    except Exception as e:
        return False, str(e)


def generate_reports_for_specific_users(
    user_list: list,
    db_path: str = "./slurm_data.db",
    output_dir: str = "./reports/user_reports",
    template_path: str = "./reports/user_report_template.qmd",
    min_jobs: int = 1,
) -> list[str]:
    """
    Generate reports for specific users by their user IDs.

    Args:
        user_list: List of user IDs to generate reports for
        db_path: Path to the database file
        output_dir: Directory to save the reports
        template_path: Path to the Quarto template file
        min_jobs: Minimum number of jobs a user must have

    Returns:
        List of paths to the generated reports
    """
    if not user_list:
        print("No users specified. Exiting.")
        return []

    print(f"Generating reports for {len(user_list)} specific users: {', '.join(user_list)}")

    # Check if the template file exists
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        return []

    # Check if output directory is writable
    writable, error = check_directory_writable(output_dir)
    if not writable:
        print(f"Error: Output directory is not writable: {output_dir}")
        return []

    # Load and preprocess data, filtering for specific users
    jobs_df = load_and_preprocess_data(db_path, specific_users=user_list)

    if len(jobs_df) == 0:
        print("No job data found for the specified users.")
        return []

    # Initialize the efficiency analyzer
    analyzer = EfficiencyAnalysis(jobs_df=jobs_df)

    # Filter jobs for analysis (only GPU jobs with allocated VRAM > 0)
    filtered_jobs = analyzer.filter_jobs_for_analysis(
        gpu_count_filter={"min": 1, "max": np.inf, "inclusive": True},
        vram_constraint_filter=None,
        allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
        gpu_mem_usage_filter=None,
    )

    # Calculate job efficiency metrics
    job_metrics = analyzer.calculate_job_efficiency_metrics(filtered_jobs=filtered_jobs)

    # Calculate user efficiency metrics
    all_users = analyzer.calculate_user_efficiency_metrics()

    # Filter users who have minimum required jobs
    filtered_users = all_users[all_users["job_count"] >= min_jobs]

    # Get data only for the users in our list
    users_to_report = filtered_users[filtered_users["User"].isin(user_list)]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate reports for each user
    report_paths = []
    for _, user_row in users_to_report.iterrows():
        user_id = user_row["User"]
        report_path = generate_user_report(
            user_id=user_id,
            user_data=user_row,
            job_data=job_metrics,
            all_users_data=all_users,
            output_dir=output_dir,
            template_path=template_path,
        )
        if report_path:
            report_paths.append(report_path)

    print(f"Generated {len(report_paths)} reports.")
    return report_paths


def generate_all_reports(
    n_top_ineff: int = 5,  # Changed default from 50 to 5
    efficiency_threshold: float = 0.3,
    min_jobs: int = 10,
    db_path: str = "./slurm_data.db",
    output_dir: str = "./reports/user_reports",
    template_path: str = "./reports/user_report_template.qmd",
) -> list[str]:
    """
    Generate reports for the top N inefficient users.

    Args:
        n_top_ineff: Number of top inefficient users to generate reports for (default: 5)
        efficiency_threshold: Maximum efficiency threshold for inefficient users
        min_jobs: Minimum number of jobs a user must have
        db_path: Path to the database file
        output_dir: Directory to save the reports
        template_path: Path to the Quarto template file

    Returns:
        List of paths to the generated reports
    """
    # Check if the template file exists
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        return []

    # Check if output directory is writable
    writable, error = check_directory_writable(output_dir)
    if not writable:
        print(f"Error: Output directory is not writable: {output_dir}")
        return []

    # Load and preprocess data
    jobs_df = load_and_preprocess_data(db_path)

    # Check if we have any data
    if len(jobs_df) == 0:
        print("No job data found. Cannot generate reports.")
        return []

    # Identify inefficient users
    inefficient_users, job_metrics, all_users = identify_inefficient_users(
        jobs_df, efficiency_threshold=efficiency_threshold, min_jobs=min_jobs, top_n=n_top_ineff
    )

    # Generate reports for each user
    report_paths = []
    for _, user_row in inefficient_users.iterrows():
        user_id = user_row["User"]
        report_path = generate_user_report(
            user_id=user_id,
            user_data=user_row,
            job_data=job_metrics,
            all_users_data=all_users,
            output_dir=output_dir,
            template_path=template_path,
        )
        if report_path:
            report_paths.append(report_path)

    print(f"Generated {len(report_paths)} reports.")
    return report_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate GPU usage reports for inefficient users")

    # Add a subparser for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Parser for "top" mode (generates reports for top inefficient users)
    top_parser = subparsers.add_parser("top", help="Generate reports for top inefficient users")
    top_parser.add_argument("--n-top", type=int, default=5, help="Number of top inefficient users (default: 5)")
    top_parser.add_argument(
        "--efficiency", type=float, default=0.3, help="Maximum efficiency threshold (0-1) (default: 0.3)"
    )
    top_parser.add_argument(
        "--min-jobs", type=int, default=10, help="Minimum number of jobs a user must have (default: 10)"
    )

    # Parser for "users" mode (generates reports for specific users)
    users_parser = subparsers.add_parser("users", help="Generate reports for specific users")
    users_parser.add_argument(
        "--users", type=str, required=True, help="Comma-separated list of user IDs to generate reports for"
    )
    users_parser.add_argument(
        "--min-jobs", type=int, default=1, help="Minimum number of jobs a user must have (default: 1)"
    )

    # Common arguments for both modes
    parser.add_argument(
        "--db-path",
        type=str,
        default="./slurm_data_new.db",
        help="Path to the database file (default: ./slurm_data.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports/user_reports",
        help="Directory to save the reports (default: ./reports/user_reports)",
    )
    parser.add_argument(
        "--template", type=str, default="./reports/user_report_template.qmd", help="Path to the Quarto template file"
    )

    args = parser.parse_args()

    # For backward compatibility, if no mode is specified, default to "top"
    if args.mode is None:
        args.mode = "top"
        # Use the default values for top mode
        args.n_top = 5
        args.efficiency = 0.3
        args.min_jobs = 10

    # Generate reports based on the selected mode
    if args.mode == "top":
        print(f"Generating reports for top {args.n_top} inefficient users...")
        report_paths = generate_all_reports(
            n_top_ineff=args.n_top,
            efficiency_threshold=args.efficiency,
            min_jobs=args.min_jobs,
            db_path=args.db_path,
            output_dir=args.output_dir,
            template_path=args.template,
        )
    elif args.mode == "users":
        # Parse the comma-separated list of users
        user_list = [user.strip() for user in args.users.split(",")]
        print(f"Generating reports for {len(user_list)} specific users: {', '.join(user_list)}")
        report_paths = generate_reports_for_specific_users(
            user_list=user_list,
            db_path=args.db_path,
            output_dir=args.output_dir,
            template_path=args.template,
            min_jobs=args.min_jobs,
        )

    print(f"Reports saved to {args.output_dir}")
    print(f"Generated {len(report_paths)} reports")
