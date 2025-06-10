from datetime import datetime, timedelta

from fire import Fire
import numpy as np

import duckdb


class CPUMetrics:
    """A class for analyzing pure CPU jobs."""

    def __init__(self, metricsfile="/modules/admin-resources/reporting/slurm_data.db", min_elapsed=600) -> None:
        """Initialize CPUMetrics with job data from a DuckDB database.

        Parameters:
            metricsfile (str, optional): Path to the DuckDB database file containing job data.
            min_elapsed (int, optional): Minimum elapsed time (in seconds) for jobs to be included.

        """
        self.con = duckdb.connect(metricsfile)
        # TODO - handle array jobs properly
        df = self.con.query(
            "select Elapsed, StartTime, CPUs,"
            "StartTime-SubmitTime as Queued, TimeLimit, Interactive, "
            "IsArray, JobID, ArrayID, Status, Constraints, Partition, User, Account from Jobs "
            f"where Elapsed>{int(min_elapsed)} "
            " and Status != 'CANCELLED' and Status != 'FAILED'"
        ).to_df()
        self.df = df

    def group_stats(self, days_back=182) -> None:
        """Print the breakdown of CPU hour usage by PI group, including quantile statistics.

        Parameters:
            days_back (int, optional): Number of days to look back for jobs.

        Returns:
            None: Prints summary statistics and quantile cutoffs to the console.

        """
        datetime.now() - timedelta(days=days_back)
        filtered_df = duckdb.query("select sum(CPUs*Elapsed/3600) as CPUHours, Account from df group by Account").df()
        print(filtered_df["CPUHours"].describe().to_markdown(tablefmt="grid", floatfmt=".1f"))
        total_cpu_hours = filtered_df["CPUHours"].sum()
        filtered_df["CPUHours"] /= total_cpu_hours
        sorted_df = filtered_df.sort_values("CPUHours").iloc[::-1]
        sorted_arr = sorted_df["CPUHours"].values
        quantiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        cutoffs = np.cumsum(sorted_arr)
        print(quantiles, np.searchsorted(cutoffs, quantiles))

    def pi_report(self, account, days_back=60) -> None:
        """Generate breakdown of CPU usage for a PI group, summarized by user.

        Parameters:
            account (str): PI group account name.
            days_back (int, optional): Number of days to look back for jobs (default 60).

        Returns:
            None: Prints a summary report to the console.

        """
        cutoff = datetime.now() - timedelta(days=days_back)
        filtered_df = duckdb.query(
            "select CPUs*Elapsed/3600 as CPUHours, CPUs, Interactive,"
            f"User, Queued from df where Account='{account}' and StartTime>='{cutoff}'"
        ).df()
        filtered_df["Queued"] = filtered_df["Queued"].apply(lambda x: x.total_seconds() / 3600)
        filtered_df["Interactive"] = filtered_df["Interactive"].notna()
        gb = filtered_df.groupby(["User"])
        print(f"CPU usage for PI group {account}")
        summary = gb[["CPUs", "Queued"]].median()
        summary["Total CPU Hours"] = gb["CPUHours"].sum()
        summary["Pct Usage"] = summary["Total CPU Hours"] / summary["Total CPU Hours"].sum() * 100
        summary["# of jobs"] = gb["CPUs"].count()
        summary["Max CPUs"] = gb["CPUs"].max()
        print(
            summary.rename(columns={"CPUs": "Median CPUs/job", "Queued": "Median queued hours"}).to_markdown(
                tablefmt="grid", floatfmt=".1f"
            )
        )


if __name__ == "__main__":
    Fire(CPUMetrics)
