"""
Generate GPU usage reports for inefficient users using Quarto.

This script analyzes GPU job data, identifies inefficient users, and generates
personalized HTML reports with usage statistics and recommendations.
"""

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

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
# from src.analysis.frequency_analysis import FrequencyAnalysis  # noqa: E402
from src.config.enum_constants import EfficiencyCategoryEnum  # noqa: E402
from src.analysis.user_comparison import UserComparison  # noqa: E402
from src.utilities.report_generation import (  # noqa: E402
    calculate_analysis_period,
    calculate_summary_statistics,
    calculate_comparison_statistics,
    generate_recommendations,
    get_total_job_count,
)


def generate_user_report(
    user_id: str,
    user_data: pd.DataFrame | pd.Series,
    job_data: pd.DataFrame | pd.Series,
    output_dir: str,
    template_path: str,
    output_format: str = "html",
    db: DatabaseConnection | None = None,
    user_comparison: UserComparison | None = None,
) -> str | None:
    """
    Generate a report for a specific user using Quarto.

    Args:
        user_id: User ID to generate report for
        user_data: DataFrame row with user metrics
        job_data: DataFrame with all job metrics
        output_dir: Directory to save the report
        template_path: Path to the Quarto template file
        output_format: Output format ('html' or 'pdf')
        db: Database connection object (optional, for comparison stats)
        user_comparison: UserComparison instance (optional, for efficient comparison stats)

    Returns:
        Path to the generated report
    """
    # efficiency_metric = "requested_vram_efficiency"

    # Filter jobs for this specific user
    user_jobs = job_data[job_data["User"] == user_id].copy()

    if len(user_jobs) == 0:
        print(f"      ❌ No jobs found for {user_id}")
        return None

    # Create a different output directory for each report to avoid conflicts
    # TODO (Ayush): Remove this
    if "ppt" in template_path:
        user_output_dir = os.path.join(output_dir, "presentation_reports", f"user_{user_id}")
    else:
        user_output_dir = os.path.join(output_dir, f"user_{user_id}")

    os.makedirs(user_output_dir, exist_ok=True)

    output_file = f"{user_id}_report.{output_format}"
    final_output_path = os.path.join(user_output_dir, output_file)

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

    # Calculate analysis period using shared utility
    analysis_period = calculate_analysis_period(user_jobs)

    # === CALCULATE SUMMARY STATISTICS USING SHARED UTILITIES ===
    # Use the get_total_job_count function if db connection is provided
    if db:
        total_all_jobs = get_total_job_count(db, analysis_period)
    else:
        total_all_jobs = len(job_data)  # Fallback to current dataset size

    # Use shared utility for summary statistics
    summary_stats = calculate_summary_statistics(
        user_jobs=user_jobs,
        all_jobs_count=total_all_jobs,
        analysis_period=analysis_period
    )

    # === CALCULATE COMPARISON STATISTICS ===
    # Use shared utility for comparison statistics
    if user_comparison and user_comparison._cached_all_users_metrics is not None:
        # Use the cached all users data for comparison
        all_jobs_df = user_comparison._cached_all_users_metrics
        comparison_stats = calculate_comparison_statistics(
            user_id=user_id,
            user_jobs=user_jobs,
            all_jobs_df=all_jobs_df
        )
    else:
        # Fallback - create basic comparison using just user's data
        comparison_stats = calculate_comparison_statistics(
            user_id=user_id,
            user_jobs=user_jobs,
            all_jobs_df=user_jobs  # Fallback to user's own data
        )

    # === PREPARE TIME SERIES DATA ===
    time_series_data = pd.DataFrame()
    # try:
    #     freq_analyzer = FrequencyAnalysis(pd.DataFrame(job_data))
    #     print(job_data.columns)
    #     time_series_data = freq_analyzer.prepare_time_series_data(
    #         users=[user_id], metric=efficiency_metric, time_unit=TimeUnitEnum.WEEKS, remove_zero_values=False
    #     )
    # except Exception as e:
    #     print(f"Warning: Could not prepare time series data: {e}")

    # === PREPARE GPU TYPE DATA ===
    if "GPUType" in user_jobs.columns and "primary_gpu_type" in user_jobs.columns:
        gpu_type_data = user_jobs["primary_gpu_type"].value_counts().reset_index()
        gpu_type_data.columns = ["gpu_type", "job_count"]
    else:
        gpu_type_data = pd.DataFrame(columns=["gpu_type", "job_count"])

    # === PREPARE ALL USERS DATA FOR ROC ANALYSIS ===
    # TODO (Ayush): Work on using existing data so that preprocess/analysis doesn't run multiple times
    all_users_job_metrics = None
    if user_comparison and user_comparison._cached_all_users_metrics is not None:
        # Get the full job metrics data (this already has efficiency calculations)
        all_users_job_metrics = user_comparison._cached_all_users_metrics
    
    # Generate recommendations using shared utility
    recommendations = generate_recommendations(user_jobs, user_data)

    # === SAVE DATA TO JSON FOR QUARTO ===

    # Get all users data for ROC analysis in template
    all_users_data = None
    if user_comparison and user_comparison._cached_all_users_metrics is not None:
        # Only include essential columns to reduce size
        essential_cols = ['User', 'alloc_vram_efficiency', 'used_vram_gib', 'allocated_vram', 'GPUCount']
        all_users_df = user_comparison._cached_all_users_metrics
        available_cols = [col for col in essential_cols if col in all_users_df.columns]
        if available_cols:
            all_users_data = all_users_df[available_cols]

    temp_data_pkl = {
        "user_id": user_id,
        "start_date": start_date,
        "end_date": end_date,
        "analysis_period": analysis_period,
        "summary_stats": summary_stats,
        "comparison_stats": comparison_stats,
        "time_series_data": time_series_data,
        "gpu_type_data": gpu_type_data,
        "all_users_job_metrics": all_users_job_metrics,
        "all_users_data": all_users_data,
        "recommendations": recommendations,
        "user_data": pd.DataFrame(user_data),
        "user_jobs": pd.DataFrame(user_jobs),
    }

    # get abs path to the template in the reports folder
    template_file = os.path.join(project_root, template_path)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(temp_data_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp_pickle_path = f.name
    
    permanent_path = os.path.join(project_root, "reports", "test_data.pkl")
    shutil.copy(temp_pickle_path, permanent_path)

    # Use absolute path for output file to avoid path resolution issues
    output_file_abs = os.path.abspath(final_output_path)
    
    cmd = ["quarto", "render", template_file, "--to", output_format, "-o", output_file, "--execute"]
    cmd.extend(["-P", f"pickle_file:{temp_pickle_path}"])
    try:
        # Run Quarto in the reports directory (full path)
        reports_dir = os.path.join(project_root, "reports")
        print("Starting Quarto rendering...")
        print(f"Output file: {output_file_abs}")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            check=False,  # Don't raise exception, handle it manually
            capture_output=True,
            text=True,
            cwd=reports_dir,
        )

        # Check if command was successful
        if result.returncode != 0:
            print(f"      ❌ Quarto rendering failed (code {result.returncode})")
            if result.stderr:
                print(f"         Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"         Output: {result.stdout.strip()}")
            return None

        # Check if output file was created
        if not os.path.exists(output_file_abs):
            print("      ❌ Output file not created")
            print(f"         Expected at: {output_file_abs}")
            return None

        print(f"      ✅ Report saved: {os.path.basename(final_output_path)}")
        return final_output_path

    except subprocess.CalledProcessError as e:
        print(f"      ❌ Process error: {e}")
        return None
    except Exception as e:
        print(f"      ❌ Unexpected error: {e}")
        return None


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
    output_format: str = "html",
) -> list[str]:
    """
    Generate reports for specific users by their user IDs.

    Args:
        user_list: List of user IDs to generate reports for
        db_path: Path to the database file
        output_dir: Directory to save the reports
        template_path: Path to the Quarto template file
        min_jobs: Minimum number of jobs a user must have
        output_format: Output format ('html' or 'pdf')

    Returns:
        List of paths to the generated reports
    """
    if not user_list:
        print("No users specified. Exiting.")
        return []

    # Check if the template file exists
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        return []

    # Check if output directory is writable
    writable, error = check_directory_writable(output_dir)
    if not writable:
        print(f"Error: Output directory is not writable: {output_dir}")
        return []

    # Initialize database connection and UserComparison
    db = DatabaseConnection(db_path)
    user_comparison = UserComparison(db)

    print("\n🔬 ANALYZING USERS")

    # Generate reports for each user
    report_paths = []
    for i, user_id in enumerate(user_list, 1):
        print(f"   [{i}/{len(user_list)}] {user_id}")

        # Get user metrics
        user_jobs = user_comparison.get_user_metrics(user_id)

        if len(user_jobs) < min_jobs:
            print(f"      ❌ {user_id}: Not enough jobs ({len(user_jobs)} found, minimum required: {min_jobs})")
            continue

        # Generate report
        report_path = generate_user_report(
            user_id=user_id,
            user_data=user_jobs,
            job_data=user_jobs,
            output_dir=output_dir,
            template_path=template_path,
            output_format=output_format,
            db=db,
            user_comparison=user_comparison,  # Pass the UserComparison instance for efficient comparison
        )
        if report_path:
            report_paths.append(report_path)

    print("\n✅ REPORTS COMPLETED")
    print(f"   Generated: {len(report_paths)} reports")
    return report_paths


def generate_all_reports(
    n_top_ineff: int = 5,
    efficiency_threshold: float = 0.3,
    min_jobs: int = 10,
    db_path: str = "./slurm_data.db",
    output_dir: str = "./reports/user_reports",
    template_path: str = "./reports/user_report_template.qmd",
    output_format: str = "html",
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
        output_format: Output format ('html' or 'pdf')

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

    # Initialize database connection and UserComparison (this loads all data once)
    db = DatabaseConnection(db_path)
    user_comparison = UserComparison(db)

    # Get all cached job data from UserComparison
    jobs_df = user_comparison._cached_all_users_metrics

    # Check if we have any data
    if jobs_df is None or len(jobs_df) == 0:
        print("No job data found. Cannot generate reports.")
        return []

    # Define toggle for efficiency metric
    use_requested_vram = True
    
    # Initialize the efficiency analyzer
    analyzer = EfficiencyAnalysis(jobs_df=jobs_df)
    
    # Filter jobs for analysis (only GPU jobs with allocated VRAM > 0)
    filtered_jobs = analyzer.filter_jobs_for_analysis(
        gpu_count_filter={"min": 1, "max": np.inf, "inclusive": True},
        vram_constraint_filter=None,
        allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
        gpu_mem_usage_filter=None,
    )

    # Calculate job efficiency metrics to prepare data for user analysis
    analyzer.calculate_job_efficiency_metrics(filtered_jobs=filtered_jobs)
    
    # Calculate user efficiency metrics to prepare user-level data
    analyzer.calculate_user_efficiency_metrics()
    
    # Find inefficient users
    if use_requested_vram:
        inefficient_users = analyzer.find_inefficient_users_by_requested_vram_efficiency(
            requested_vram_efficiency_filter={"min": 0, "max": efficiency_threshold, "inclusive": True},
            min_jobs=min_jobs
        )
    else:
        inefficient_users = analyzer.find_inefficient_users_by_alloc_vram_efficiency(
            alloc_vram_efficiency_filter={"min": 0, "max": efficiency_threshold, "inclusive": True},
            min_jobs=min_jobs
        )

    # Get top N inefficient users
    inefficient_users = inefficient_users.head(n_top_ineff)

    # Generate reports for each user using the efficient UserComparison
    report_paths = []
    for _, user_row in inefficient_users.iterrows():
        user_id = user_row["User"]

        # Get user-specific data efficiently from UserComparison
        user_jobs = user_comparison.get_user_metrics(user_id)

        if len(user_jobs) == 0:
            print(f"      ❌ No jobs found for {user_id}")
            continue

        report_path = generate_user_report(
            user_id=user_id,
            user_data=user_row,
            job_data=user_jobs,  # Pass user-specific jobs instead of all job_metrics
            output_dir=output_dir,
            template_path=template_path,
            output_format=output_format,
            db=db,  # Pass the database connection for comparison stats
            user_comparison=user_comparison,  # Pass the UserComparison instance for efficient comparison
        )
        if report_path:
            report_paths.append(report_path)

    print("\n✅ REPORTS COMPLETED")
    print(f"   Generated: {len(report_paths)} reports")

    return report_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GPU usage reports for inefficient users")

    # Add a subparser for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Parser for "top" mode (generates reports for top inefficient users)
    top_parser = subparsers.add_parser("top", help="Generate reports for top inefficient users")
    top_parser.add_argument("--n-top", type=int, default=5, help="Number of top inefficient users (default: 5)")
    top_parser.add_argument(
        "--efficiency",
        type=float,
        default=EfficiencyCategoryEnum.LOW_THRESHOLD.value,
        help="Maximum efficiency threshold (0-1) (default: 0.3)"
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
        "--min-jobs", type=int, default=10, help="Minimum number of jobs a user must have (default: 10)"
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
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "pdf"],
        default="html",
        help="Output format for the reports (default: html)",
    )

    args = parser.parse_args()
    print("Using:", args.template)

    # For backward compatibility, if no mode is specified, default to "top"
    if args.mode is None:
        args.mode = "top"
        # Use the default values for top mode
        args.n_top = 5
        args.efficiency = 0.3
        args.min_jobs = 10

    # Generate reports based on the selected mode
    if args.mode == "top":
        print(f"🎯 ANALYZING TOP {args.n_top} INEFFICIENT USERS")
        print(f"   Efficiency threshold: ≤ {args.efficiency:.0%}")
        print(f"   Minimum jobs required: {args.min_jobs}")
        report_paths = generate_all_reports(
            n_top_ineff=args.n_top,
            efficiency_threshold=args.efficiency,
            min_jobs=args.min_jobs,
            db_path=args.db_path,
            output_dir=args.output_dir,
            template_path=args.template,
            output_format=args.format,
        )
    elif args.mode == "users":
        # Parse the comma-separated list of users
        user_list = [user.strip() for user in args.users.split(",")]
        print("👥 GENERATING REPORTS FOR SPECIFIC USERS")
        print(f"   Users: {', '.join(user_list)}")
        print(f"   Minimum jobs required: {args.min_jobs}")
        report_paths = generate_reports_for_specific_users(
            user_list=user_list,
            db_path=args.db_path,
            output_dir=args.output_dir,
            template_path=args.template,
            min_jobs=args.min_jobs,
            output_format=args.format,
        )

    print("\n🎉 ALL TASKS COMPLETED!")
    print(f"   Output directory: {args.output_dir}")
    print("   Check the individual function outputs above for details.")