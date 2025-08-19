"""
Generate PI Group reports using Quarto.

This script analyzes PI group performance, identifies inefficient PI groups, and generates
personalized HTML reports with group statistics, user performance analysis, and recommendations.
"""

import argparse
import os
import subprocess
import sys
import shutil
import tempfile
import pickle

import numpy as np
import pandas as pd

# Add the project root directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.database_connection import DatabaseConnection  # noqa: E402
from src.analysis.pi_group_analysis import PIGroupAnalysis  # noqa: E402
from src.config.enum_constants import EfficiencyCategoryEnum  # noqa: E402


def generate_pi_group_report(
    pi_account: str,
    pi_analyzer: PIGroupAnalysis,
    output_dir: str,
    template_path: str,
    output_format: str = "html",
) -> str | None:
    """
    Generate a report for a specific PI group using Quarto.

    Args:
        pi_account: PI account name to generate report for
        pi_analyzer: PIGroupAnalysis instance with loaded data
        output_dir: Directory to save the report
        template_path: Path to the Quarto template file
        output_format: Output format ('html' or 'pdf')

    Returns:
        Path to the generated report
    """
    print(f"   📊 Generating report for PI group: {pi_account}")

    # Get PI group data
    pi_metrics = pi_analyzer.get_pi_group_metrics(pi_account)
    if len(pi_metrics) == 0:
        print(f"      ❌ No metrics found for PI group {pi_account}")
        return None

    pi_users = pi_analyzer.get_pi_group_users(pi_account)
    if len(pi_users) == 0:
        print(f"      ❌ No users found for PI group {pi_account}")
        return None

    # Get worst performing users (up to 5)
    worst_users = pi_analyzer.get_worst_users_in_pi_group(pi_account, n_users=5)

    # Get comparison statistics
    comparison_stats = pi_analyzer.get_pi_group_comparison_statistics(pi_account)

    # Get size comparison
    size_stats = pi_analyzer.get_pi_group_size_comparison(pi_account)

    # Calculate analysis period
    analysis_period = pi_analyzer.calculate_analysis_period()

    # Create summary statistics
    metrics_row = pi_metrics.iloc[0]
    efficiency = metrics_row["expected_value_alloc_vram_efficiency"]
    efficiency_category = EfficiencyCategoryEnum.get_efficiency_category(efficiency)

    summary_stats = pd.DataFrame({
        "Metric": [
            "PI Account",
            "Number of Users",
            "Total GPU Jobs",
            "Total GPU Hours",
            "Total VRAM Hours",
            "Average VRAM Efficiency",
            "Efficiency Category",
            "Efficiency Score",
            "Analysis Period",
        ],
        "Value": [
            pi_account,
            f"{metrics_row['user_count']}",
            f"{metrics_row['job_count']:,}",
            f"{metrics_row['pi_acc_job_hours']:.1f}",
            f"{metrics_row['pi_acc_vram_hours']:.1f}",
            f"{efficiency:.1%}",
            efficiency_category,
            f"{metrics_row['avg_alloc_vram_efficiency_score']:.2f}"
            if pd.notna(metrics_row["avg_alloc_vram_efficiency_score"])
            and metrics_row["avg_alloc_vram_efficiency_score"] != -np.inf
            else "Very Low",
            analysis_period,
        ],
    })

    # Generate recommendations
    recommendations = generate_pi_group_recommendations(pi_account, metrics_row, pi_users, worst_users)

    # Create output filename
    safe_pi_name = pi_account.replace("/", "_").replace("\\", "_").replace(" ", "_")
    output_filename = f"{safe_pi_name}_report.{output_format}"
    final_output_path = os.path.join(output_dir, output_filename)

    # Check that the template exists
    if not os.path.exists(template_path):
        abs_template_path = os.path.abspath(template_path)
        if os.path.exists(abs_template_path):
            template_path = abs_template_path
        else:
            print(f"Error: Template file not found at: {template_path}")
            return None

    # Prepare data for template
    template_data = {
        "pi_account": pi_account,
        "pi_metrics": pi_metrics,
        "pi_users": pi_users,
        "worst_users": worst_users,
        "comparison_stats": comparison_stats,
        "size_stats": size_stats,
        "analysis_period": analysis_period,
        "summary_stats": summary_stats,
        "recommendations": recommendations,
    }

    # Get absolute path to the template
    template_file = os.path.join(project_root, "reports", "pi_group_report_template.qmd")

    # Save data to pickle file for efficient transfer
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(template_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp_pickle_path = f.name

    # Simple output filename for quarto
    simple_output_filename = f"report.{output_format}"

    # Run quarto render command
    cmd = ["quarto", "render", template_file, "--to", output_format, "-o", simple_output_filename, "--execute"]
    cmd.extend(["-P", f"pickle_file:{temp_pickle_path}"])

    try:
        # Run Quarto in the reports directory
        reports_dir = os.path.dirname(template_path)
        print("      🔧 Starting Quarto rendering...")
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=reports_dir,
        )

        if result.returncode != 0:
            print(f"      ❌ Quarto rendering failed (code {result.returncode})")
            if result.stderr:
                print(f"         Error: {result.stderr.strip()}")
            return None

        # Check if output file was created
        reports_dir = os.path.dirname(template_path)
        output_file_path = os.path.join(reports_dir, simple_output_filename)
        if not os.path.exists(output_file_path):
            print("      ❌ Output file not created")
            return None

        # Copy the generated report to final location
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        shutil.copy2(output_file_path, final_output_path)
        print(f"      ✅ Report saved: {os.path.basename(final_output_path)}")

        # Clean up temporary files
        os.unlink(temp_pickle_path)

        return final_output_path

    except subprocess.CalledProcessError as e:
        print(f"      ❌ Process error: {e}")
        return None
    except Exception as e:
        print(f"      ❌ Unexpected error: {e}")
        return None


def generate_pi_group_recommendations(
    pi_account: str, metrics: pd.Series, pi_users: pd.DataFrame, worst_users: pd.DataFrame
) -> list[str]:
    """
    Generate targeted recommendations for a PI group.

    Args:
        pi_account: PI account name
        metrics: PI group metrics
        pi_users: All users in the PI group
        worst_users: Worst performing users in the group

    Returns:
        List of recommendation strings
    """
    recommendations = []

    efficiency = metrics["expected_value_alloc_vram_efficiency"]
    user_count = metrics["user_count"]

    # Efficiency-based recommendations
    if efficiency < EfficiencyCategoryEnum.VERY_LOW_THRESHOLD.value:
        recommendations.append(
            "**Critical Efficiency Alert**: Your PI group's GPU memory efficiency is extremely low. "
            "Consider organizing immediate GPU optimization training for all group members."
        )
        recommendations.append(
            "**Resource Review**: Evaluate whether all submitted jobs actually require GPUs. "
            "Many workloads might be more suitable for CPU-only execution."
        )
    elif efficiency < EfficiencyCategoryEnum.LOW_THRESHOLD.value:
        recommendations.append(
            "**Efficiency Improvement Priority**: Focus on optimizing GPU memory usage patterns "
            "across your research group. Consider implementing group-wide best practices for memory allocation."
        )

    # Group size-based recommendations
    if user_count >= 10:
        recommendations.append(
            "**Large Group Management**: With a large research group, consider establishing "
            "group-wide GPU usage guidelines and regular efficiency reviews to maintain optimal resource utilization."
        )
    elif user_count <= 2:
        recommendations.append(
            "**Small Group Optimization**: As a small research group, focus on intensive optimization "
            "of existing workflows to maximize the impact of your GPU resource usage."
        )

    # User-specific recommendations
    if len(worst_users) > 0:
        low_eff_count = len(worst_users[worst_users["expected_value_alloc_vram_efficiency"] < 0.15])
        if low_eff_count > 0:
            recommendations.append(
                f"**Targeted User Training**: {low_eff_count} users in your group show very low efficiency patterns. "(
                    "Consider providing targeted GPU optimization guidance to: "
                    f"{', '.join(worst_users.head(3)['User'].tolist())}."
                )
            )

    # Resource usage recommendations
    high_usage_threshold = 1000  # VRAM hours
    if metrics["pi_acc_vram_hours"] > high_usage_threshold:
        recommendations.append(
            "**High Resource Usage**: Your group is among the heavy GPU users. "
            "Focus on efficiency improvements to maximize the value of your substantial resource allocation."
        )

    # Best practices sharing
    if len(pi_users) > 3:
        best_users = pi_users.nlargest(2, "expected_value_alloc_vram_efficiency")
        if len(best_users) > 0 and best_users.iloc[0]["expected_value_alloc_vram_efficiency"] > 0.4:
            recommendations.append(
                f"**Best Practices Sharing**: Your most efficient users ({', '.join(best_users['User'].tolist())}) "
                f"achieve good GPU utilization. Have them share optimization strategies with the rest of the group."
            )

    # Default recommendation if none specific
    if not recommendations:
        recommendations.append(
            "**Continuous Improvement**: Maintain regular monitoring of GPU usage patterns and "
            "encourage ongoing optimization efforts across your research group."
        )

    return recommendations


def generate_reports_for_specific_pi_groups(
    pi_list: list,
    db_path: str = "./slurm_data.db",
    output_dir: str = "./reports/pi_group_reports",
    template_path: str = "./reports/pi_group_report_template.qmd",
    min_jobs: int = 10,
    output_format: str = "html",
) -> list[str]:
    """
    Generate reports for specific PI groups.

    Args:
        pi_list: List of PI account names to generate reports for
        db_path: Path to the database file
        output_dir: Directory to save the reports
        template_path: Path to the Quarto template file
        min_jobs: Minimum number of jobs a PI group must have
        output_format: Output format ('html' or 'pdf')

    Returns:
        List of paths to the generated reports
    """
    if not pi_list:
        print("No PI groups specified. Exiting.")
        return []

    # Check if the template file exists
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        return []

    # Check if output directory is writable
    from scripts.generate_user_reports import check_directory_writable

    writable, error = check_directory_writable(output_dir)
    if not writable:
        print(f"Error: Output directory is not writable: {output_dir}")
        return []

    # Initialize database connection and PI group analyzer
    db = DatabaseConnection(db_path)
    pi_analyzer = PIGroupAnalysis(db)

    print("\n🏛️ ANALYZING PI GROUPS")

    # Generate reports for each PI group
    report_paths = []
    for i, pi_account in enumerate(pi_list, 1):
        print(f"   [{i}/{len(pi_list)}] {pi_account}")

        # Get PI group metrics
        pi_metrics = pi_analyzer.get_pi_group_metrics(pi_account)

        if len(pi_metrics) == 0:
            print(f"      ❌ {pi_account}: No metrics found")
            continue

        # Check minimum jobs requirement
        if pi_metrics.iloc[0]["job_count"] < min_jobs:
            print(
                f"      ❌ {pi_account}: Not enough jobs "
                f"({pi_metrics.iloc[0]['job_count']} found, minimum required: {min_jobs})"
            )
            continue

        # Generate report
        report_path = generate_pi_group_report(
            pi_account=pi_account,
            pi_analyzer=pi_analyzer,
            output_dir=output_dir,
            template_path=template_path,
            output_format=output_format,
        )
        if report_path:
            report_paths.append(report_path)

    print("\n✅ PI GROUP REPORTS COMPLETED")
    print(f"   Generated: {len(report_paths)} reports")
    return report_paths


def generate_all_pi_group_reports(
    n_top_ineff: int = 5,
    efficiency_threshold: float = 0.3,
    min_jobs: int = 50,
    min_users: int = 1,
    db_path: str = "./slurm_data.db",
    output_dir: str = "./reports/pi_group_reports",
    template_path: str = "./reports/pi_group_report_template.qmd",
    output_format: str = "html",
) -> list[str]:
    """
    Generate reports for the top N inefficient PI groups.

    Args:
        n_top_ineff: Number of top inefficient PI groups to generate reports for
        efficiency_threshold: Maximum efficiency threshold for inefficient PI groups
        min_jobs: Minimum number of jobs a PI group must have
        min_users: Minimum number of users a PI group must have
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
    from scripts.generate_user_reports import check_directory_writable

    writable, error = check_directory_writable(output_dir)
    if not writable:
        print(f"Error: Output directory is not writable: {output_dir}")
        return []

    # Initialize database connection and PI group analyzer
    db = DatabaseConnection(db_path)
    pi_analyzer = PIGroupAnalysis(db)

    # Identify inefficient PI groups
    inefficient_pis = pi_analyzer.identify_inefficient_pi_groups(
        efficiency_threshold=efficiency_threshold, min_jobs=min_jobs, min_users=min_users, top_n=n_top_ineff
    )

    if len(inefficient_pis) == 0:
        print("No inefficient PI groups found with the specified criteria.")
        return []

    print(f"\n🎯 ANALYZING TOP {len(inefficient_pis)} INEFFICIENT PI GROUPS")
    print(f"   Efficiency threshold: ≤ {efficiency_threshold:.0%}")
    print(f"   Minimum jobs required: {min_jobs}")

    # Generate reports for each inefficient PI group
    report_paths = []
    for _, pi_row in inefficient_pis.iterrows():
        pi_account = pi_row["pi_account"]

        report_path = generate_pi_group_report(
            pi_account=pi_account,
            pi_analyzer=pi_analyzer,
            output_dir=output_dir,
            template_path=template_path,
            output_format=output_format,
        )
        if report_path:
            report_paths.append(report_path)

    print("\n✅ PI GROUP REPORTS COMPLETED")
    print(f"   Generated: {len(report_paths)} reports")

    return report_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PI Group GPU usage reports")

    # Add a subparser for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Parser for "top" mode (generates reports for top inefficient PI groups)
    top_parser = subparsers.add_parser("top", help="Generate reports for top inefficient PI groups")
    top_parser.add_argument("--n-top", type=int, default=5, help="Number of top inefficient PI groups (default: 5)")
    top_parser.add_argument(
        "--efficiency",
        type=float,
        default=EfficiencyCategoryEnum.LOW_THRESHOLD.value,
        help="Maximum efficiency threshold (0-1) (default: 0.3)",
    )
    top_parser.add_argument(
        "--min-jobs", type=int, default=50, help="Minimum number of jobs a PI group must have (default: 50)"
    )
    top_parser.add_argument(
        "--min-users", type=int, default=1, help="Minimum number of users a PI group must have (default: 1)"
    )

    # Parser for "groups" mode (generates reports for specific PI groups)
    groups_parser = subparsers.add_parser("groups", help="Generate reports for specific PI groups")
    groups_parser.add_argument(
        "--groups", type=str, required=True, help="Comma-separated list of PI group names to generate reports for"
    )
    groups_parser.add_argument(
        "--min-jobs", type=int, default=10, help="Minimum number of jobs a PI group must have (default: 10)"
    )

    # Common arguments for both modes
    parser.add_argument(
        "--db-path",
        type=str,
        default="./slurm_data_new.db",
        help="Path to the database file (default: ./slurm_data_new.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports/pi_group_reports",
        help="Directory to save the reports (default: ./reports/pi_group_reports)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="./reports/pi_group_report_template.qmd",
        help="Path to the Quarto template file",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "pdf"],
        default="html",
        help="Output format for the reports (default: html)",
    )

    args = parser.parse_args()

    # For backward compatibility, if no mode is specified, default to "top"
    if args.mode is None:
        args.mode = "top"
        args.n_top = 5
        args.efficiency = 0.3
        args.min_jobs = 50

    # Generate reports based on the selected mode
    if args.mode == "top":
        print(f"🎯 ANALYZING TOP {args.n_top} INEFFICIENT PI GROUPS")
        print(f"   Efficiency threshold: ≤ {args.efficiency:.0%}")
        print(f"   Minimum jobs required: {args.min_jobs}")
        print(f"   Minimum users required: {args.min_users}")
        report_paths = generate_all_pi_group_reports(
            n_top_ineff=args.n_top,
            efficiency_threshold=args.efficiency,
            min_jobs=args.min_jobs,
            min_users=args.min_users,
            db_path=args.db_path,
            output_dir=args.output_dir,
            template_path=args.template,
            output_format=args.format,
        )
    elif args.mode == "groups":
        # Parse the comma-separated list of PI groups
        pi_list = [group.strip() for group in args.groups.split(",")]
        print("🏛️ GENERATING REPORTS FOR SPECIFIC PI GROUPS")
        print(f"   PI Groups: {', '.join(pi_list)}")
        print(f"   Minimum jobs required: {args.min_jobs}")
        report_paths = generate_reports_for_specific_pi_groups(
            pi_list=pi_list,
            db_path=args.db_path,
            output_dir=args.output_dir,
            template_path=args.template,
            min_jobs=args.min_jobs,
            output_format=args.format,
        )

    print("\n🎉 ALL PI GROUP TASKS COMPLETED!")
    print(f"   Output directory: {args.output_dir}")
    print("   Check the individual function outputs above for details.")
