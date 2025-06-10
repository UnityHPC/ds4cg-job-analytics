"""Script to identify users who habitually waste GPU hours on Unity and optionally email them a report.

This script analyzes GPU job usage, identifies users with repeated wasted GPU hours, and can generate
summary reports or personalized email notifications. It uses DuckDB for querying job data and pandas
for data manipulation.
"""

from datetime import datetime, timedelta

from fire import Fire
from gpu_metrics import GPUMetrics
import pandas as pd

import duckdb

INTRO = """Dear {name},

Over the past few months, we've noticed that all of your {jobs} jobs on Unity which requested GPU resources did not utilize the requested GPUs."""

HOURS = "{hours} unused GPU hours. The most recent jobs are the following:"


def get_job_type_breakdown(interactive, jobs):
    """Generate a summary string describing the breakdown of interactive and batch jobs.

    Parameters:
        interactive (int): Number of interactive jobs.
        jobs (int): Total number of jobs.

    Returns:
        str: Description of job types and counts.

    """
    if interactive == jobs:
        return f" These consisted of {interactive} interactive sessions, totaling "
    if not interactive:
        return " These amounted to "
    return f" These included {interactive} interactive session{'s' if interactive > 1 else ''}, as well as {jobs - interactive} batch job{'s' if jobs - interactive > 1 else ''}, totaling "


def pi_report(account, days_back=60):
    """Create an efficiency report for a given PI group, summarizing GPU usage and waste.

    Parameters:
        account (str): PI group account name.
        days_back (int, optional): Number of days to look back for jobs (default 60).

    Returns:
        None: Prints a summary report to the console.

    """
    cutoff = datetime.now() - timedelta(days=days_back)
    filtered_df = duckdb.query(
        """
        select GPUs*Elapsed/3600 as GPUHours, GPUMemUsage=0 as Wasted, GPUMemUsage, Interactive,
        User, Queued from df where Account=? and StartTime>=?
        """,
        [account, cutoff],
    ).df()
    filtered_df["Queued"] = filtered_df["Queued"].apply(lambda x: x.total_seconds() / 3600)
    filtered_df["WastedHours"] = filtered_df["GPUHours"] * filtered_df["Wasted"]
    filtered_df["Interactive"] = filtered_df["Interactive"].notna()
    filtered_df["GPUMemUsage"] /= 2**30
    gb = filtered_df.groupby(["User", "Wasted"])
    print(gb.mean()[["GPUHours", "GPUMemUsage", "Queued"]].rename(columns={"Wasted": "Fraction "}))


def main(
    dbfile="./modules/admin-resources/reporting/slurm_data.db", userlist="./users.csv", sendEmail=False, days_back=60
):
    """Print out a list of users who habitually waste GPU hours, and optionally email them a report.

    Parameters:
        dbfile (str, optional): Path to the DuckDB database file containing job data.
        userlist (str, optional): Path to the CSV file containing user information.
        sendEmail (bool, optional): Whether to send email notifications to users (default False).
        days_back (int, optional): Number of days to look back for jobs (default 60).

    Returns:
        None: Prints a summary report or sends emails to users.

    """
    metrics = GPUMetrics(metricsfile=dbfile)
    jobs = metrics.df

    cutoff = datetime(2025, 3, 1) - timedelta(days=days_back)
    df = duckdb.query(
        """
        select GPUs*Elapsed/3600 as GPUHours, GPUMemUsage=0 as Wasted, GPUMemUsage, Interactive,
        User, Queued, IsArray, Account, StartTime, JobID, Status from jobs
        where StartTime >= ? and (Status = 'COMPLETED' or Status = 'TIMEOUT')
        """,
        [cutoff],
    ).df()

    # Wasted is a boolean indicating whether the job used any GPU memory
    df["WastedGPUHours"] = df["Wasted"] * df["GPUHours"]
    df["Interactive"] = df["Interactive"].notna()
    gb = df.groupby("User")

    # sum wasted means it gets the number of jobs that had no GPU memory usage
    user_report = gb[["Wasted", "WastedGPUHours", "GPUHours", "IsArray", "Interactive"]].sum()
    user_report["TotalJob"] = gb.size()
    user_report["WasteRatio"] = user_report["WastedGPUHours"] / user_report["GPUHours"]
    user_report["PI"] = gb["Account"].first()
    user_report["LastJob"] = gb["StartTime"].max()
    # at least 3 jobs are wasted and more than 4 hours of total runtime
    mask = (user_report["Wasted"] > 3) & (user_report["WastedGPUHours"] > 4)
    filtered_report = user_report[mask].reset_index().sort_values("WasteRatio")
    sorted_report = (
        filtered_report[filtered_report["WasteRatio"] == 1]
        .sort_values("WastedGPUHours", ascending=False)
        .reset_index(drop=True)
    )
    sorted_report["H/J"] = sorted_report["WastedGPUHours"] / sorted_report["Wasted"]  # wasted GPU hours per job
    users = pd.read_csv(userlist)

    if sendEmail:
        for _, row in sorted_report.iterrows():
            interactive = int(row["Interactive"])
            jobs = int(row["Wasted"])
            template = INTRO + get_job_type_breakdown(interactive, jobs) + HOURS
            df2 = users[users["user"] == row["User"]]

            if df2.empty:
                print(f"User {row['User']} not found in user list.")
                continue
            # user_info = users.loc[row["User"]]
            # if not user_info:
            #     continue
            # name = user_info["first"]
            name = row["User"]
            email = template.format(name=name, jobs=jobs, hours=int(row["WastedGPUHours"]))
            job_samples = duckdb.query(
                """
                select JobID as Job, StartTime as Start, GPUHours from df
                where User=? order by StartTime desc limit 5
                """,
                [row["User"]],
            ).df()
            print(email + "\n")
            print(job_samples.rename(columns={"GPUHours": "Unused GPU Hours"}).to_string(index=False))
            print("\n" + "-" * 80 + "\n")
    else:
        print(
            sorted_report[["User", "Wasted", "WastedGPUHours", "H/J", "LastJob"]]
            .rename(columns={"Wasted": "Jobs", "WastedGPUHours": "Hours"})
            .to_markdown(tablefmt="grid", floatfmt=".1f")
        )


if __name__ == "__main__":
    Fire(main)
