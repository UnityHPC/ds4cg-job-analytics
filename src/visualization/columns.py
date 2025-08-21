"""
Visualization utilities for pre-processed Unity job data.

Provides a function to visualize and summarize each column of a DataFrame, including appropriate
plots and statistics for numeric and categorical columns.
"""

import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pydantic import ValidationError

from .models import ColumnVisualizationKwargsModel
from .visualization import DataVisualizer
from ..config.constants import VRAM_CATEGORIES


class ColumnVisualizer(DataVisualizer[ColumnVisualizationKwargsModel]):
    """A class for visualizing and summarizing columns of pre-processed data in a DataFrame."""

    def _generate_boolean_bar_plot(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        title: str,
        output_dir_path: Path | None = None,
    ) -> None:
        """
        Generate a bar plot for boolean columns.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            title (str): The title of the plot.
            output_dir_path (Path | None): The directory to save the plot.

        Returns:
            None
        """
        plt.figure(figsize=(5, 7))
        ax = sns.countplot(x=col, stat="percent", data=jobs_df)
        if isinstance(ax.containers[0], BarContainer):
            # The heights are already in percent (0-100) due to stat="percent"
            ax.bar_label(ax.containers[0], labels=[f"{h.get_height():.1f}%" for h in ax.containers[0]])
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_barplot.png", bbox_inches="tight")
        plt.show()

    def _plot_duration_histogram(self, jobs_df: pd.DataFrame, col: str, output_dir_path: Path | None = None) -> None:
        """
        Plot histogram of durations in minutes, hours, days.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.

        Returns:
            None
        """
        col_data = jobs_df[col]

        # Convert to minutes for plotting
        timelimit_minutes = col_data.dropna().dt.total_seconds() / 60

        # Define breakpoints for minutes, hours, days
        min_break = 60  # 1 hour in minutes
        hour_break = 1440  # 1 day in minutes
        max_val = timelimit_minutes.max()
        total_count = len(timelimit_minutes)

        # Prepare data for each section
        minutes_data = timelimit_minutes[(timelimit_minutes <= min_break)]
        hours_data = timelimit_minutes[(timelimit_minutes > min_break) & (timelimit_minutes <= hour_break)]
        days_data = timelimit_minutes[(timelimit_minutes > hour_break)]

        # Proportional widths: minutes (up to 1hr), hours (1hr-1d), days (>1d)
        width_minutes = len(minutes_data) / total_count if total_count > 0 else 0
        width_hours = len(hours_data) / total_count if total_count > 0 else 0
        width_days = len(days_data) / total_count if total_count > 0 else 0

        # Enforce minimum width for non-empty sections
        min_width = 0.3  # Minimum width for any section

        widths = []
        for w, d in zip([width_minutes, width_hours, width_days], [minutes_data, hours_data, days_data], strict=True):
            if not d.empty:
                widths.append(max(w, min_width))
            else:
                widths.append(0)
        # If all are empty (shouldn't happen), fallback to equal widths
        if sum(widths) == 0:
            widths = [1, 1, 1]
        # Normalize so total is 1.0
        total_width = sum(widths)
        widths = [w / total_width for w in widths]
        width_minutes, width_hours, width_days = widths

        def pct(n: int, total_count: int = total_count) -> str:
            return f"{(n / total_count * 100):.1f}%" if total_count > 0 else "0.0%"

        # Helper to reduce xticks and rotate if width is small
        def choose_ticks_and_rotation(ticks: list[int], width: float, max_labels: int = 3) -> tuple[list[int], int]:
            # Always include first and last tick, and always have at least 4 ticks
            if width < min_width and len(ticks) > max_labels:
                # Always include first and last, and evenly space the rest
                n = max(max_labels - 2, 2)
                idxs = np.linspace(1, len(ticks) - 2, n, dtype=int)
                selected = [ticks[0]] + [ticks[i] for i in idxs] + [ticks[-1]]
                # Ensure at least 4 ticks (first, last, and two in the middle if possible)
                selected = list(dict.fromkeys(selected))  # Remove duplicates if any
                if len(selected) < 4 and len(ticks) >= 4:
                    # Insert additional ticks from the middle if needed
                    mids = [ticks[len(ticks) // 2]]
                    if len(ticks) > 4:
                        mids.append(ticks[(len(ticks) - 1) // 3])
                    for mid in mids:
                        if mid not in selected:
                            selected.insert(1, mid)
                    selected = list(dict.fromkeys(selected))
                return selected, 0
            # If less than 4 ticks, just return all
            if len(ticks) < 4:
                return ticks, 0
            return ticks, 0

        # Add space between sections to avoid overlapping xticks
        section_gap = 0.05  # width of gap between sections (in axis units)
        fig = plt.figure(figsize=(8, 5))
        # Add a small gap between sections by adding blank axes
        # We'll use 5 sections: [minutes, gap, hours, gap, days]
        width_gap = section_gap
        # Dynamically build width_ratios and section indices based on non-empty sections
        section_widths = []
        section_axes = []
        section_labels = []
        # Track which sections are present and their indices in the GridSpec
        idx = 0
        if not minutes_data.empty:
            section_widths.append(width_minutes)
            section_axes.append("minutes")
            section_labels.append("minutes")
            idx += 1
        if not minutes_data.empty and not hours_data.empty:
            section_widths.append(width_gap)
            section_axes.append("gap1")
            idx += 1
        if not hours_data.empty:
            section_widths.append(width_hours)
            section_axes.append("hours")
            section_labels.append("hours")
            idx += 1
        if not hours_data.empty and not days_data.empty:
            section_widths.append(width_gap)
            section_axes.append("gap2")
            idx += 1
        if not days_data.empty:
            section_widths.append(width_days)
            section_axes.append("days")
            section_labels.append("days")
            idx += 1

        gs = GridSpec(1, len(section_widths), width_ratios=section_widths, wspace=0.0)
        ax0 = None
        # Minutes axis
        ax_idx = 0
        if not minutes_data.empty:
            ax0 = fig.add_subplot(gs[ax_idx])
            ax0.hist(minutes_data, bins=20, color="tab:blue", alpha=0.7, log=True)
            ax0.set_xlim(0, min_break)
            min_ticks = [0, 15, 30, 45, 60]
            min_ticklabels = ["0", "15m", "30m", "45m", "1h"]
            min_ticks_sel, min_rot = choose_ticks_and_rotation(min_ticks, width_minutes)
            min_ticklabels_sel = [min_ticklabels[i] for i, t in enumerate(min_ticks) if t in min_ticks_sel]
            ax0.set_xticks(min_ticks_sel)
            ax0.set_xticklabels(min_ticklabels_sel, rotation=min_rot, ha="right" if min_rot else "center")
            ax0.set_ylabel("Count (log scale)")
            ax0.set_title(f"Minutes (≤1h)\nN={len(minutes_data)} ({pct(len(minutes_data))})")
            ax0.spines["right"].set_visible(False)
            ax0.spines["top"].set_visible(False)
            ax_idx += 1
            # Blank axis for gap between minutes and hours, only if hours_data is not empty
            if not hours_data.empty:
                ax_gap1 = fig.add_subplot(gs[ax_idx], frame_on=False)
                ax_gap1.set_xticks([])
                ax_gap1.set_yticks([])
                ax_gap1.axis("off")
                ax_idx += 1

        # Hours axis (only if hours_data is not empty)
        if not hours_data.empty:
            ax1 = fig.add_subplot(gs[ax_idx], sharey=ax0)
            ax1.hist(hours_data, bins=20, color="tab:orange", alpha=0.7, log=True)
            ax1.set_xlim(min_break, hour_break)
            hour_ticks = [60, 360, 720, 1440]
            hour_ticklabels = ["1h", "6h", "12h", "1d"]
            hour_ticks_sel, hour_rot = choose_ticks_and_rotation(hour_ticks, width_hours)
            hour_ticklabels_sel = [hour_ticklabels[i] for i, t in enumerate(hour_ticks) if t in hour_ticks_sel]
            ax1.set_xticks(hour_ticks_sel)
            ax1.set_xticklabels(hour_ticklabels_sel, rotation=hour_rot, ha="right" if hour_rot else "center")
            ax1.set_title(f"Hours (1h–1d)\nN={len(hours_data)} ({pct(len(hours_data))})")
            ax1.spines["left"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            plt.setp(ax1.get_yticklabels(), visible=False)
            ax_idx += 1
            # Blank axis for gap between hours and days, only if days_data is not empty
            if not days_data.empty:
                ax_gap2 = fig.add_subplot(gs[ax_idx], frame_on=False)
                ax_gap2.set_xticks([])
                ax_gap2.set_yticks([])
                ax_gap2.axis("off")
                ax_idx += 1

        # Days axis (only if days_data is not empty)
        if not days_data.empty:
            ax2 = fig.add_subplot(gs[ax_idx], sharey=ax0)
            ax2.hist(days_data, bins=20, color="tab:green", alpha=0.7, log=True)
            ax2.set_xlim(hour_break, max_val)
            # Choose ticks for days
            if max_val > hour_break:
                day_ticks = [
                    hour_break,
                    hour_break + (max_val - hour_break) / 3,
                    hour_break + 2 * (max_val - hour_break) / 3,
                    max_val,
                ]
            day_labels = ["1d"] + [f"{int(t / 1440)}d" for t in day_ticks[1:]]
            day_ticks_sel, day_rot = choose_ticks_and_rotation(day_ticks, width_days)
            day_labels_sel = [day_labels[i] for i, t in enumerate(day_ticks) if t in day_ticks_sel]
            ax2.set_xticks(day_ticks_sel)
            ax2.set_xticklabels(day_labels_sel, rotation=day_rot, ha="right" if day_rot else "center")
            ax2.set_title(f"Days (>1d)\nN={len(days_data)} ({pct(len(days_data))})")
            ax2.spines["left"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            plt.setp(ax2.get_yticklabels(), visible=False)

        # Remove space between subplots (handled by blank axes)
        plt.subplots_adjust(wspace=0.0)
        fig.suptitle(f"Histogram of {col} (minutes, hours, days)", y=1.04)

        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_hist.png", bbox_inches="tight")
        plt.show()

    def _generate_start_time_histogram(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
        timestamp_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> None:
        """
        Generate a histogram of job start times, either by day or by hour.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.
            timestamp_range (tuple[pd.Timestamp, pd.Timestamp] | None): The time range to filter the data.

        Raises:
            ValueError: If the column does not contain valid timestamps or if no valid timestamps are found.

        Returns:
            None
        """
        col_data = jobs_df[col]
        if timestamp_range is not None:
            col_data = col_data[col_data.between(timestamp_range[0], timestamp_range[1])]
        if not pd.api.types.is_datetime64_any_dtype(col_data):
            # Convert to datetime if not already
            col_data = pd.to_datetime(col_data, errors="coerce")  # invalid timestamps will be NaT

        col_data = col_data.dropna().sort_values()
        if col_data.empty:
            raise ValueError(f"No valid timestamps in {col}. Skipping visualization.")
        min_time = col_data.min()
        max_time = col_data.max()
        total_days = (max_time - min_time).days + 1

        plt.figure(figsize=(10, 5))
        plt.tight_layout()
        # If jobs span more than 2 days, plot jobs per day
        if total_days > 2:
            # Group by date, count jobs per day
            jobs_per_day = col_data.dt.floor("D").value_counts().sort_index()
            # Trim days at start/end with zero jobs
            jobs_per_day = jobs_per_day[jobs_per_day > 0]
            # A line plot is often better for time series to show trends over days
            plt.plot(jobs_per_day.index, np.asarray(jobs_per_day.values, dtype=int), marker="o", linestyle="-")
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            plt.xlabel("Date")
            plt.ylabel("Number of jobs")
            plt.title(f"Histogram of Number of Jobs per day for {col}")
            # Only show enough x-tick labels to keep the plot readable
            tick_locs = jobs_per_day.index
            max_labels = 10
            if len(tick_locs) > max_labels:
                # Always include first and last, then evenly space the rest
                idxs = np.linspace(0, len(tick_locs) - 1, max_labels, dtype=int)
                shown_locs = tick_locs[idxs]
                shown_labels = [dt.strftime("%Y-%m-%d") for dt in shown_locs]
                ax.set_xticks(shown_locs)
                ax.set_xticklabels(shown_labels, rotation=45, ha="right")
            else:
                plt.xticks(rotation=45, ha="right")

            # --- Legend with average, median, and first date ---
            mean_count = float(jobs_per_day.mean())
            median_count = float(jobs_per_day.median())
            # Horizontal reference lines
            ax.axhline(mean_count, color="red", linestyle="--", linewidth=1, alpha=0.8)
            ax.axhline(median_count, color="purple", linestyle=":", linewidth=1, alpha=0.8)
            legend_handles = [
                Line2D([0], [0], color="red", linestyle="--", label=f"Avg: {mean_count:.1f}"),
                Line2D([0], [0], color="purple", linestyle=":", label=f"Median: {median_count:.1f}"),
                Patch(facecolor="none", edgecolor="none", label=f"Days span: {total_days}"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="best",
                frameon=True,
                fontsize=9,
                title="Summary",
                title_fontsize=10,
            )
            if output_dir_path is not None:
                plt.savefig(output_dir_path / f"{col}_days_lineplot.png", bbox_inches="tight")
        else:
            # All jobs within a couple of days: plot by hour
            jobs_per_hour = col_data.dt.floor("H").value_counts().sort_index()
            jobs_per_hour = jobs_per_hour[jobs_per_hour > 0]
            # Use line plot for time series to better show trends over hours
            plt.plot(jobs_per_hour.index, np.asarray(jobs_per_hour.values, dtype=int), marker="o", linestyle="-")
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
            plt.xlabel("Hour")
            plt.ylabel("Number of jobs")
            plt.title(f"Histogram of Number of Jobs per hour for {col}")
            plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)

            # Set x-axis labels: show date at midnight, hour otherwise
            ax = plt.gca()
            tick_locs = jobs_per_hour.index
            tick_labels = []
            for dt in tick_locs:
                if dt.hour == 0:
                    tick_labels.append(dt.strftime("%Y-%m-%d %H:00"))
                else:
                    tick_labels.append(dt.strftime("%H:00"))
            # Show at most 12 labels to avoid crowding
            step = max(1, len(tick_labels) // 12)
            ax.set_xticks(tick_locs[::step])
            ax.set_xticklabels([tick_labels[i] for i in range(0, len(tick_labels), step)], rotation=45, ha="right")

            # --- Legend with average, median, and first date ---
            mean_count = float(jobs_per_hour.mean())
            median_count = float(jobs_per_hour.median())
            ax.axhline(mean_count, color="red", linestyle="--", linewidth=1, alpha=0.8)
            ax.axhline(median_count, color="purple", linestyle=":", linewidth=1, alpha=0.8)
            legend_handles = [
                Line2D([0], [0], color="red", linestyle="--", label=f"Avg: {mean_count:.1f}"),
                Line2D([0], [0], color="purple", linestyle=":", label=f"Median: {median_count:.1f}"),
                Patch(facecolor="none", edgecolor="none", label=f"Days span: {total_days}"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                fontsize=9,
                title="Summary",
                title_fontsize=10,
            )

            if output_dir_path is not None:
                plt.savefig(output_dir_path / f"{col}_hourly_lineplot.png", bbox_inches="tight")
        plt.show()

    def _generate_interactive_pie_chart(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
    ) -> None:
        """Generate a pie chart for interactive vs non-interactive jobs.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.

        Returns:
            None
        """
        # Replace empty/NaN with 'non interactive', others as their type
        interactive_col = jobs_df[col].astype(str)
        counts = interactive_col.value_counts()
        plt.figure(figsize=(5, 7))

        # Define threshold for small slices
        threshold_pct = 5
        total = counts.sum()
        pct_values = counts.div(total).multiply(100)

        # Explode small slices to separate them visually
        explode = [max(0.15 - p / 100 * 4, 0.1) if p < threshold_pct else 0 for p in pct_values]

        # Prepare labels: only show label on pie if above threshold
        labels = [str(label) if p >= threshold_pct else "" for label, p in zip(counts.index, pct_values, strict=True)]

        # Prepare legend labels for all slices
        legend_labels = [
            f"{label}: {count} ({p:.1f}%)" for label, count, p in zip(counts.index, counts, pct_values, strict=True)
        ]

        # Create a gridspec to reserve space for the legend above the pie
        fig = plt.gcf()
        fig.clf()

        # Use 3 rows: title, legend, pie
        gs = GridSpec(3, 1, height_ratios=[0.05, 0.75, 0.25], hspace=0.0)
        ax_title = fig.add_subplot(gs[0])
        ax_pie = fig.add_subplot(gs[1])
        ax_legend = fig.add_subplot(gs[2])

        wedges, *_ = ax_pie.pie(
            counts,
            labels=labels,
            autopct=self.pie_chart_autopct_func,
            startangle=0,
            colors=sns.color_palette("pastel")[0 : len(counts)],
            explode=explode,
        )

        ax_pie.axis("equal")

        # Hide the legend and title axes
        ax_legend.axis("off")
        ax_title.axis("off")

        # Place the title in its own axis, centered
        ax_title.text(
            0.5,
            0.5,
            "Job Type Distribution (Interactive vs Non Interactive)",
            ha="center",
            va="center",
            fontsize=14,
        )

        # Place the legend below the title and above the pie chart
        ax_legend.legend(
            wedges,
            legend_labels,
            title="Job Type",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=1,
            fontsize=10,
            title_fontsize=11,
        )
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_piechart.png", bbox_inches="tight")
        plt.show()

    def _generate_status_pie_chart(self, jobs_df: pd.DataFrame, col: str, output_dir_path: Path | None = None) -> None:
        """Generate a pie chart for job statuses.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.

        Raises:
            ValueError: If the column does not contain valid statuses or if all counts are zero.

        Returns:
            None
        """
        col_data = jobs_df[col]
        # Count occurrences of each status
        exit_code_counts = col_data.value_counts()
        total_count = exit_code_counts.sum()
        if total_count == 0:
            raise ValueError(f"No valid statuses in {col}. Skipping visualization.")

        # Prepare labels and explode small slices
        threshold_pct = 5
        pct_values = exit_code_counts.div(total_count).multiply(100)
        explode = [max(0.15 - p / 100 * 4, 0.1) if p < threshold_pct else 0 for p in pct_values]

        # Prepare labels: only show label on pie if above threshold
        labels = [
            str(label) if p >= threshold_pct else ""
            for label, p in zip(exit_code_counts.index, pct_values, strict=True)
        ]

        # Prepare legend labels for all slices
        legend_labels = [
            f"{label}: {count} ({p:.1f}%)"
            for label, count, p in zip(exit_code_counts.index, exit_code_counts, pct_values, strict=True)
        ]

        # Create figure with extra horizontal space for legend on the right
        fig, ax_pie = plt.subplots(figsize=(10, 6))
        wedges, *_ = ax_pie.pie(
            exit_code_counts,
            labels=labels,
            autopct=self.pie_chart_autopct_func,
            startangle=0,
            colors=sns.color_palette("pastel")[0 : len(exit_code_counts)],
            explode=explode,
        )
        ax_pie.axis("equal")
        # Use a figure-level title so it spans the entire figure width (including legend space)
        fig.suptitle(f"Job Status Distribution ({col})", fontsize=14, y=0.98)

        # Place legend to the right of the pie chart
        legend = ax_pie.legend(
            wedges,
            legend_labels,
            title="Job Status",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=10,
            title_fontsize=11,
            borderaxespad=0.0,
            frameon=True,
        )
        # Slight transparency for legend background for readability
        legend.get_frame().set_alpha(0.9)

        # Adjust layout so legend isn't clipped
        plt.subplots_adjust(right=0.78, top=0.88)
        # tight_layout with rect leaves space for suptitle
        fig.tight_layout(rect=(0, 0, 0.98, 0.90))
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_piechart.png", bbox_inches="tight")
        plt.show()

    def _generate_gpu_type_bar_plot(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
    ) -> None:
        """
        Generate a bar plot for GPU types (GPUType column) and show number of jobs with more than one GPU type.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.

        Raises:
            ValueError: If the column does not contain lists or dictionaries.

        Returns:
            None
        """

        col_data = jobs_df[col]
        non_null = col_data.dropna()
        if not all(isinstance(x, (list, dict)) for x in non_null):
            msg = (
                f"Error: Not all entries in column '{col}' are lists or dictionaries. "
                f"Example values:\n{non_null.head()}"
            )
            print(msg)
            raise ValueError(msg)

        # Flatten all GPU types and count per job
        multi_count = sum(1 for entry in non_null if len(entry) > 1)
        single_count = len(non_null) - multi_count

        plt.figure(figsize=(7, 4))
        gpu_counts = pd.Series(sum((Counter(entry) for entry in non_null), Counter())).sort_values(ascending=False)
        ax = sns.barplot(
            x=gpu_counts.index, y=gpu_counts.values, hue=gpu_counts.index, palette="viridis", legend=False
        )

        plt.title(f"GPU Types ({col})")
        plt.xlabel("GPU type")
        plt.ylabel("Number of jobs using this GPU type")
        plt.xticks(rotation=45, ha="right")

        gpu_percents = gpu_counts / gpu_counts.sum() * 100
        tallest = gpu_counts.max()
        gap = max(2.5, tallest * 0.08)
        ax.set_ylim(0, tallest + gap)
        for i, (count, percent) in enumerate(zip(gpu_counts.values, gpu_percents.values, strict=True)):
            label_y = count + gap * 0.2
            ax.text(
                i,
                label_y,
                f"{percent:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

        info_text = f"Jobs with 1 GPU type: {single_count}\nJobs with >1 GPU type: {multi_count}"
        ax.text(
            0.98,
            0.98,
            info_text,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
            transform=ax.transAxes,
            zorder=10,
        )
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_barplot.png", bbox_inches="tight")
        plt.show()

    def _generate_qos_pie_chart(self, jobs_df: pd.DataFrame, col: str, output_dir_path: Path | None = None) -> None:
        """
        Generate a pie chart for QoS (Quality of Service) distribution.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.

        Raises:
            ValueError: If the column does not contain valid QOS or if all counts are zero.

        Returns:
            None
        """
        col_data = jobs_df[col]
        # Count occurrences of each QOS
        qos_counts = col_data.value_counts()
        total_count = qos_counts.sum()
        if total_count == 0:
            raise ValueError(f"No valid QOS in {col}. Skipping visualization.")

        # Prepare labels and explode small slices
        threshold_pct = 5
        pct_values = qos_counts.div(total_count).multiply(100)
        explode = [max(0.15 - p / 100 * 4, 0.1) if p < threshold_pct else 0 for p in pct_values]

        # Prepare labels: only show label on pie if above threshold
        labels = [
            str(label) if p >= threshold_pct else "" for label, p in zip(qos_counts.index, pct_values, strict=True)
        ]

        plt.figure(figsize=(5, 10))

        # Prepare legend labels for all slices
        legend_labels = [
            f"{label}: {count} ({p:.1f}%)"
            for label, count, p in zip(qos_counts.index, qos_counts, pct_values, strict=True)
        ]

        # Create a gridspec to reserve space for the legend above the pie
        fig = plt.gcf()
        fig.clf()

        # Use 3 rows: title, legend, pie
        gs = GridSpec(3, 1, height_ratios=[0.05, 0.75, 0.25], hspace=0.0)
        ax_title = fig.add_subplot(gs[0])
        ax_pie = fig.add_subplot(gs[1])
        ax_legend = fig.add_subplot(gs[2])

        wedges, *_ = ax_pie.pie(
            qos_counts,
            labels=labels,
            autopct=self.pie_chart_autopct_func,
            startangle=0,
            colors=sns.color_palette("pastel")[0 : len(qos_counts)],
            explode=explode,
        )

        ax_pie.axis("equal")

        # Hide the legend and title axes
        ax_legend.axis("off")
        ax_title.axis("off")

        ax_title.text(
            0.5,
            0.5,
            f"Job QOS Distribution ({col})",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        # Add a legend below the pie chart
        ax_legend.legend(
            wedges,
            legend_labels,
            title="Job QOS",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            fontsize=10,
            title_fontsize=11,
        )
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_piechart.png", bbox_inches="tight")
        plt.show()

    def _generate_gpu_count_pie_chart(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
    ) -> None:
        """
        Generate a pie chart for GPU counts.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.

        Raises:
            ValueError: If the column does not contain valid GPU counts or if all counts are zero

        Returns:
            None
        """
        # Count occurrences of each GPU count
        gpu_counts = jobs_df[col].value_counts().sort_index()

        # Prepare labels and explode small slices
        total_count = gpu_counts.sum()
        if total_count == 0:
            raise ValueError(f"No valid GPU counts in {col}. Skipping visualization.")

        threshold_pct = 5
        pct_values = gpu_counts.div(total_count).multiply(100)
        # Explode values increase with index: first group (0), second (small), ..., last (largest)
        explode = [i * 0.04 for i in range(len(gpu_counts))]

        # Prepare labels: only show label on pie if above threshold
        labels = [
            str(label) if p >= threshold_pct else "" for label, p in zip(gpu_counts.index, pct_values, strict=True)
        ]

        # Prepare legend labels for all slices (limit to top 10 for readability)
        legend_labels = [
            f"{label} GPU{'s' if int(label) != 1 else ''}: {count} ({p:.1f}%)"
            for label, count, p in zip(gpu_counts.index, gpu_counts, pct_values, strict=True)
        ][:10]

        # Build figure with room on right for legend
        fig, ax_pie = plt.subplots(figsize=(9, 5))

        # Format labels as "x GPU(s)" for each wedge (suppress small slices label text per earlier threshold rule)
        formatted_labels = [
            f"{label} GPU{'s' if int(label) != 1 else ''}" if lab != "" else ""
            for label, lab in zip(gpu_counts.index, labels, strict=True)
        ]

        wedges, *_ = ax_pie.pie(
            gpu_counts,
            labels=formatted_labels,
            autopct=self.pie_chart_autopct_func,
            startangle=0,
            colors=sns.color_palette("pastel")[0 : len(gpu_counts)],
            explode=explode,
        )
        ax_pie.axis("equal")

        # Figure-level title spanning entire width
        fig.suptitle(f"GPU Count Distribution ({col})", fontsize=14, y=0.97)

        # Legend to the right
        legend = ax_pie.legend(
            wedges,
            legend_labels,
            title="GPU Count",
            loc="center left",
            bbox_to_anchor=(0.7, 0.5),
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            borderaxespad=0.0,
        )
        legend.get_frame().set_alpha(0.9)

        # Adjust layout so legend not clipped
        plt.subplots_adjust(right=0.78, top=0.88)
        fig.tight_layout(rect=(0, 0, 0.98, 0.90))
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_piechart.png", bbox_inches="tight")
        plt.show()

    def _generate_node_list_bar_plot_combine_trailing_digits(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
        figsize: tuple[float, float] = (7, 6),
    ) -> None:
        """Generate a bar plot for node lists, combining trailing digits before counting.

        This function combines nodes with the same prefix (e.g., "gpu010" and "gpu053" become "gpu").

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.
            figsize (tuple[float, float]): The size of the figure.

        Raises:
            ValueError: If not all entries in the specified column are numpy arrays or list-like.

        Returns:
            None
        """
        # NodeList should be a numpy array or list-like per row
        # Convert all non-null entries to numpy arrays if not already
        non_null = jobs_df[col].dropna()
        if not all(isinstance(x, np.ndarray) for x in non_null):
            # Try to convert list-like to np.ndarray
            try:
                non_null = pd.Series(
                    [np.array(x) if isinstance(x, list | tuple | np.ndarray) else np.array([x]) for x in non_null],
                    index=non_null.index,
                )
            except Exception as err:
                msg = (
                    f"Error: Not all entries in column '{col}' are arrays or list-like. "
                    f"Example values:\n{non_null.head()}"
                )
                raise ValueError(msg) from err

        # For each job, create a tuple of sorted prefixes (with multiplicity)
        # Remove trailing digits from node names (combine nodes like gypsum-gpu010 and gypsumgpu053)
        # This will combine nodes with the same prefix, regardless of dashes
        nodes_clean = non_null.apply(lambda arr: tuple(sorted([re.sub(r"[-]?\d+$", "", str(x)) for x in arr])))

        # --- Count prefix combinations (with multiplicity) for jobs with >1 node ---
        multi_node_jobs = nodes_clean[nodes_clean.apply(lambda arr: len(arr) > 1)]
        prefix_combo_counts = multi_node_jobs.value_counts()
        prefix_combo_df = prefix_combo_counts.reset_index()
        prefix_combo_df.columns = ["prefix_combo", "count"]
        # Ensure each prefix_combo is a tuple of strings
        prefix_combo_df["prefix_combo"] = prefix_combo_df["prefix_combo"].apply(
            lambda x: tuple(x) if not isinstance(x, tuple) else x
        )

        # Build summary lines for each unique prefix combo
        prefix_combo_text_lines = []
        for combo, count in zip(prefix_combo_df["prefix_combo"], prefix_combo_df["count"], strict=True):
            # Count occurrences of each prefix in the combo
            prefix_counts: dict[str, int] = {}
            for prefix in combo:
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            # Format as "prefix (xN)"
            prefix_counts_str = ", ".join(f"{p} (x{c})" for p, c in sorted(prefix_counts.items()))
            prefix_combo_text_lines.append(f"Combo: [{prefix_counts_str}]  |  Jobs: {count}")

        # Limit to top 10 combos for readability, add a note if more
        max_lines = 10
        if len(prefix_combo_text_lines) > max_lines:
            prefix_combo_text = (
                "\n".join(prefix_combo_text_lines[:max_lines])
                + f"\n... ({len(prefix_combo_text_lines) - max_lines} more combos)"
            )
        else:
            prefix_combo_text = "\n".join(prefix_combo_text_lines)

        flat_nodes_clean = pd.Series(np.concatenate(nodes_clean.values))

        # Count occurrences of each node prefix
        node_counts = flat_nodes_clean.value_counts()
        node_percents = node_counts / node_counts.sum() * 100
        plt.figure(figsize=figsize, layout="constrained")
        sns.barplot(x=node_counts.index, y=node_counts.values, hue=node_counts.index, palette="viridis", legend=False)
        plt.title(f"{col} (nodes combined by removing trailing numbers)")
        # Set a long xlabel and wrap it to fit within the figure width
        xlabel = "Nodes (trailing numbers removed, e.g., gpu10,gpu50 → gpu,gpu; counts are combined)"
        plt.xlabel(xlabel)
        plt.ylabel("Number of Jobs")
        # Now wrap the xlabel based on the actual axes width in pixels
        ax = plt.gca()
        fig = plt.gcf()
        fig.canvas.draw()  # Needed to get correct bbox
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width_inch = bbox.width
        # Estimate ~10-12 characters per inch for most fonts
        wrap_width = max(30, int(width_inch * 11))
        wrapped_xlabel = "\n".join(textwrap.wrap(xlabel, width=wrap_width))
        ax.set_xlabel(wrapped_xlabel)
        plt.xticks(rotation=45, ha="right")

        ax.text(
            0.99,
            0.99,
            prefix_combo_text,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
            transform=ax.transAxes,
            zorder=10,
        )

        # Annotate bars with counts, ensuring a gap above the tallest bar label
        tallest = node_counts.max()
        gap = max(2.5, tallest * 0.2)
        ax.set_ylim(0, tallest + gap)
        for i, (pct, count) in enumerate(zip(node_percents.values, node_counts.values, strict=True)):
            label_y = count + gap * 0.2  # offset above bar, proportional to gap
            ax.text(
                i,
                label_y,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_barplot.png", bbox_inches="tight")
        plt.show()

    def _generate_partition_bar_plot(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
        figsize: tuple[float, float] = (15, 4),
    ) -> None:
        """Generate a bar plot for job partitions.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.
            figsize (tuple[int, int]): Size of the figure for the plot.

        Returns:
            None
        """
        col_data = jobs_df[col].astype(str)
        partition_counts = col_data.value_counts()
        partition_percents = partition_counts / partition_counts.sum() * 100
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            x=partition_counts.index,
            y=partition_counts,
            hue=partition_counts.index,
            palette="viridis",
            legend=False,
        )
        plt.title(f"{col}")
        plt.xlabel("Partitions")
        plt.ylabel("Number of Jobs")
        plt.xticks(rotation=45, ha="right")

        # Annotate bars with counts, ensuring a gap above the tallest bar label
        tallest = partition_counts.max()
        gap = max(10, tallest * 0.1)
        ax.set_ylim(0, tallest + gap)
        for i, (pct, count) in enumerate(zip(partition_percents.values, partition_counts.values, strict=True)):
            label_y_position = count + gap * 0.2  # offset above bar, proportional to gap
            label = "<.1%" if pct < 0.1 and pct > 0 else f"{pct:.1f}%"
            ax.text(
                i,
                label_y_position,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_barplot.png", bbox_inches="tight")
        plt.show()

    def _generate_constraints_bar_plot(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
    ) -> None:
        """Generate a bar plot for job constraints.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.

        Raises:
            ValueError: If not all entries in the specified column are ndarrays of strings.

        Returns:
            None
        """
        # Each row is an ndarray of constraints where each constraint is a string
        # Check if all non-null entries are lists or ndarray of strings
        non_null = jobs_df[col].dropna()
        if not all(isinstance(x, list) and all(isinstance(item, str) for item in x) for x in non_null):
            msg = f"Error: Not all entries in column '{col}' are lists of strings."
            raise ValueError(msg)

        # Remove beginning and trailing single quotes from each string
        non_null = non_null.apply(lambda x: tuple(str(item).strip("'") for item in x))

        # Count constraint combinations where each job has multiple constraints
        constraint_combo_counts = non_null.apply(lambda x: tuple(sorted(x))).value_counts()
        constraint_df = constraint_combo_counts.reset_index()
        constraint_df.columns = ["constraints", "count"]
        # Ensure each constraints is a tuple of strings
        constraint_df["constraints"] = constraint_df["constraints"].apply(
            lambda x: tuple(x) if not isinstance(x, tuple) else x
        )

        # Flatten all constraints into a single list across all jobs
        all_constraints = [constraint for arr in non_null for constraint in arr]
        constraint_flat_counts = pd.Series(all_constraints).value_counts()

        # --- Only plot top 20 constraints, bundle the rest ---
        top_n = 20
        if len(constraint_flat_counts) > top_n:
            top_constraints = constraint_flat_counts.iloc[:top_n]
            rest_constraints = constraint_flat_counts.iloc[top_n:]
            rest_count = rest_constraints.sum()
            rest_num = len(rest_constraints)
            # Use a placeholder label for the rest
            placeholder_label = f"{rest_num} others (<{top_constraints.iloc[-1]})"
            plot_counts = pd.concat([top_constraints, pd.Series({placeholder_label: rest_count})])
        else:
            plot_counts = constraint_flat_counts

        constraint_flat_percents = plot_counts / plot_counts.sum() * 100

        # Build summary lines for each unique constraint combo
        constraint_text_lines = []
        for combo, count in zip(constraint_df["constraints"], constraint_df["count"], strict=True):
            if len(combo) > 1:
                # Count occurrences of each constraint in the combo
                combo_counts: dict[str, int] = {}
                for constraint in combo:
                    combo_counts[constraint] = combo_counts.get(constraint, 0) + 1
                # Format as "constraint (xN)"
                constraint_counts_str = ", ".join(f"{c} (x{cnt})" for c, cnt in sorted(combo_counts.items()))
                constraint_text_lines.append(f"Combo: [{constraint_counts_str}]  |  Jobs: {count}")
        # Limit to top 10 combos for readability, add a note if more
        max_lines = 10
        if len(constraint_text_lines) > max_lines:
            constraint_text = (
                "\n".join(constraint_text_lines[:max_lines])
                + f"\n... ({len(constraint_text_lines) - max_lines} more combos)"
            )
        else:
            constraint_text = "\n".join(constraint_text_lines)

        plt.figure(figsize=(12, 5))
        ax = sns.barplot(
            x=plot_counts.index, y=plot_counts.values, hue=plot_counts.index, palette="viridis", legend=False
        )

        # Annotate bars with count values at the correct height
        tallest = plot_counts.max()
        gap = max(2.5, tallest * 0.08)
        ax.set_ylim(0, tallest + gap)
        for i, (pct, count) in enumerate(zip(constraint_flat_percents.values, plot_counts.values, strict=True)):
            label_y = count + gap * 0.2
            ax.text(
                i,
                label_y,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
        # Set y-axis label to indicate percentage
        ax.set_ylabel("Number of Jobs")
        plt.title(f"Constraint Occurrences Across All Jobs ({col})")
        plt.xlabel("Constraint")
        plt.ylabel("Number of Jobs")
        plt.xticks(rotation=45, ha="right")

        # Add constraint summary text box
        ax.text(
            0.98,
            0.98,
            constraint_text,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
            transform=ax.transAxes,
            zorder=10,
        )
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_flat_barplot.png", bbox_inches="tight")
        plt.show()

    def _generate_gpu_memory_usage_histogram_categorical_bins(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
        figsize: tuple[float, float] = (7, 4),
    ) -> None:
        """Generate a bar plot for GPU memory usage categorized by GPU type.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.
            figsize (tuple[int, int]): The size of the figure for the plot.

        Returns:
            None
        """
        # GPU memory usage should be a numpy array or list-like per row
        # Check if all non-null entries are numpy arrays
        col_data = jobs_df[col]
        # Convert to numeric if not already
        if not pd.api.types.is_numeric_dtype(col_data):
            col_data = pd.to_numeric(col_data, errors="coerce")
        col_data = col_data.dropna().div(2**30)  # Convert to GiB

        # Define the categorical VRAM bins (as categories, not intervals)
        vram_labels = [str(v) for v in VRAM_CATEGORIES]

        # Bin the data by closest category (floor to the largest category <= value)
        bins = [-0.1] + VRAM_CATEGORIES  # -0.1 to include 0 exactly
        binned = pd.cut(col_data, bins=bins, labels=vram_labels, right=False, include_lowest=True)
        binned[col_data == 0] = "0"
        binned[col_data > max(VRAM_CATEGORIES)] = str(max(VRAM_CATEGORIES))

        bin_counts = binned.value_counts(sort=False, dropna=False)
        bin_percents = bin_counts / bin_counts.sum() * 100

        plt.figure(figsize=figsize)
        ax = plt.gca()
        x_ticks = np.arange(len(vram_labels))
        bar_positions = [0] + [i + 0.5 for i in range(1, len(bin_counts))]
        bar_width = 0.8

        # Use a distinct color for 0 (e.g., red) and a colorblind-friendly palette for others
        zero_color = "#E15759"  # red, colorblind-friendly
        other_color = "#4E79A7"  # blue, colorblind-friendly

        bars = []
        # 0 GiB bar with cross-hatch
        bar0 = ax.bar(
            bar_positions[0],
            bin_percents.to_numpy()[0],
            width=bar_width,
            align="center",
            color=zero_color,
            hatch="///",
        )
        bars.append(bar0)
        # Other bars
        for i in range(1, len(vram_labels)):
            bars.append(
                ax.bar(
                    bar_positions[i], bin_percents.to_numpy()[i], width=bar_width, align="center", color=other_color
                )
            )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(vram_labels)
        ax.set_xlabel("GPU Memory (GiB)")
        ax.set_ylabel("Percentage of Jobs")
        ax.set_title("Histogram of GPU VRAM Usage")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # --- Bar labels with gap above tallest label ---
        # Find the tallest bar
        tallest = bin_percents.max()
        # Set y-limit a bit higher to leave a gap above the tallest bar label
        gap = max(2.5, tallest * 0.08)
        ax.set_ylim(0, tallest + gap)

        # Draw bar labels with bbox, always above the bar
        for i, v in enumerate(bin_percents.values):
            label_y = v + 0.7  # offset above bar
            ax.text(
                bar_positions[i],
                label_y,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

        legend_handles = [
            Patch(facecolor=zero_color, hatch="///", label="0 GiB (no GPU used)"),
            Patch(facecolor=other_color, label=">0 GiB"),
        ]
        ax.legend(handles=legend_handles, loc="upper right")
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_hist.png", bbox_inches="tight")
        plt.show()

    def _generate_non_gpu_memory_histogram(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        column_unit: str,
        output_dir_path: Path | None = None,
        figsize: tuple[float, float] = (7, 4),
    ) -> None:
        """
        Generate a histogram for non-GPU memory usage

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            column_unit (str): The unit of the memory column, which can be "B" or "KiB".
            output_dir_path (Path | None): The directory to save the plot.
            figsize (tuple[int, int]): The size of the figure for the plot.

        Raises:
            ValueError: If the column does not contain numeric data or if the unit is unsupported.

        Returns:
            None
        """
        col_data = jobs_df[col]
        # Convert to numeric if not already
        if not pd.api.types.is_numeric_dtype(col_data):
            col_data = pd.to_numeric(col_data, errors="coerce")
        if column_unit == "KiB":
            col_data = col_data.dropna().div(2**10)  # Convert KiB to MiB
            output_unit = "MiB"
        elif column_unit == "B":
            col_data = col_data.dropna().div(2**30)  # Convert B to GiB
            output_unit = "GiB"
        else:
            raise ValueError(f"Unsupported unit '{column_unit}'. Use 'B' for bytes or 'KiB' for kibibytes.")

        # Filter out non-positive values for log scale
        col_data = col_data[col_data > 0]

        plt.figure(figsize=figsize)
        ax = sns.histplot(col_data.tolist(), bins=60, kde=True, log_scale=(True, False), stat="percent")
        plt.title(f"Histogram of {col} (in {output_unit}, log scale)")
        plt.xlabel(f"{col} ({output_unit}, log scale)")
        plt.ylabel("Percent of Jobs")

        # Prepare stat summary with units
        stats = col_data.describe(percentiles=[0.25, 0.5, 0.75])
        stat_text = (
            f"Count: {int(stats['count'])} jobs\n"
            f"Mean: {stats['mean']:.1f} {output_unit}\n"
            f"Std: {stats['std']:.1f} {output_unit}\n"
            f"Min: {stats['min']:.1f} {output_unit}\n"
            f"25%: {stats['25%']:.1f} {output_unit}\n"
            f"50%: {stats['50%']:.1f} {output_unit}\n"
            f"75%: {stats['75%']:.1f} {output_unit}\n"
            f"Max: {stats['max']:.1f} {output_unit}"
        )
        # Place the box in the top right, inside the axes, with a small offset
        ax.text(
            0.98,
            0.98,
            stat_text,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
            transform=ax.transAxes,
            zorder=10,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_hist.png", bbox_inches="tight")
        plt.show()

    def _generate_exit_code_pie_chart(
        self,
        jobs_df: pd.DataFrame,
        col: str,
        output_dir_path: Path | None = None,
    ) -> None:
        """
        Generate a pie chart for job exit codes.

        Args:
            jobs_df (pd.DataFrame): The DataFrame containing job data.
            col (str): The name of the column to plot.
            output_dir_path (Path | None): The directory to save the plot.

        Raises:
            ValueError: If the column does not contain valid exit codes or if all counts are zero.

        Returns:
            None
        """
        col_data = jobs_df[col]
        # Count occurrences of each exit code
        exit_code_counts = col_data.value_counts()
        total_count = exit_code_counts.sum()
        if total_count == 0:
            raise ValueError(f"No valid exit codes in {col}. Skipping visualization.")

        # Prepare labels and explode small slices
        threshold_pct = 5
        pct_values = exit_code_counts.div(total_count).multiply(100)
        explode = [max(0.15 - p / 100 * 4, 0.2) if p < threshold_pct else 0 for p in pct_values]

        # Prepare labels: only show label on pie if above threshold
        labels = [
            str(label) if p >= threshold_pct else ""
            for label, p in zip(exit_code_counts.index, pct_values, strict=True)
        ]

        plt.figure(figsize=(5, 7))

        # Prepare legend labels for all slices
        legend_labels = [
            f"{label}: {count} ({p:.1f}%)"
            for label, count, p in zip(exit_code_counts.index, exit_code_counts, pct_values, strict=True)
        ]

        # Create a gridspec to reserve space for the legend above the pie
        fig = plt.gcf()
        fig.clf()

        # Use 3 rows: title, legend, pie
        gs = GridSpec(3, 1, height_ratios=[0.05, 0.75, 0.25], hspace=0.0)
        ax_title = fig.add_subplot(gs[0])
        ax_pie = fig.add_subplot(gs[1])
        ax_legend = fig.add_subplot(gs[2])

        wedges, *_ = ax_pie.pie(
            exit_code_counts,
            labels=labels,
            autopct=self.pie_chart_autopct_func,
            startangle=0,
            colors=sns.color_palette("pastel")[0 : len(exit_code_counts)],
            explode=explode,
        )

        ax_pie.axis("equal")

        # Hide the legend and title axes
        ax_legend.axis("off")
        ax_title.axis("off")

        ax_title.text(
            0.5,
            0.5,
            f"Job Exit Code Distribution ({col})",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        # Add a legend below the pie chart
        ax_legend.legend(
            wedges,
            legend_labels,
            title="Job Exit Codes",
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            fontsize=10,
            title_fontsize=11,
        )
        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"{col}_piechart.png", bbox_inches="tight")
        plt.show()

    @staticmethod
    def validate_columns_argument(columns: list[str] | None, jobs_df: pd.DataFrame) -> list[str] | None:
        """Validate the provided columns against the DataFrame.

        Args:
            columns (list[str]): List of column names to validate.
            jobs_df (pd.DataFrame): The DataFrame to validate against.

        Raises:
            TypeError: If 'columns' is not a list of strings or None.
            ValueError: If any column is not present in the DataFrame or if 'columns' is an empty list.

        Returns:
            list[str]: Validated list of column names.
        """

        if columns is not None and (not all(isinstance(x, str) for x in columns)):
            raise TypeError("'columns' must be a list of strings or None")

        if columns is not None and len(columns) == 0:
            raise ValueError("'columns' cannot be an empty list. 'columns' must be a list of strings or None")

        if columns is not None and jobs_df is not None and not all(col in jobs_df.columns for col in columns):
            raise ValueError("One or more specified columns are not present in the DataFrame.")
        return columns

    def validate_visualize_kwargs(
        self,
        kwargs: dict[str, Any],
        validated_jobs_df: pd.DataFrame,
        kwargs_model: type[ColumnVisualizationKwargsModel],
    ) -> ColumnVisualizationKwargsModel:
        """Validate the keyword arguments for the visualize method.

        Args:
            kwargs (dict[str, Any]): Keyword arguments to validate.
            validated_jobs_df (pd.DataFrame): The DataFrame to validate against.
            kwargs_model (type[ColumnVisualizationKwargsModel]): Pydantic model for validation.

        Raises:
            TypeError: If any keyword argument has an incorrect type.

        Returns:
            ColumnKwargsModel: A tuple with validated keyword arguments.
        """
        try:
            # Validate the kwargs using Pydantic model
            col_kwargs = kwargs_model(**kwargs)
        except ValidationError as e:
            allowed_fields = {name: str(field.annotation) for name, field in kwargs_model.model_fields.items()}
            allowed_fields_str = "\n".join(f"  {k}: {v}" for k, v in allowed_fields.items())
            raise TypeError(
                f"Invalid visualize kwargs: {e.json(indent=2)}\nAllowed fields and types:\n{allowed_fields_str}"
            ) from e

        self.validate_columns_argument(col_kwargs.columns, validated_jobs_df)
        self.validate_figsize(col_kwargs.figsize)
        self.validate_sampling_arguments(
            col_kwargs.sample_size,
            col_kwargs.random_seed,
        )
        return col_kwargs

    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        """Visualize and summarize specified columns of the data.

        Args:
            output_dir_path (Path, optional): Directory to save plots. If None, plots are displayed.
            kwargs (dict[str, Any]): Keyword arguments for visualization options.
                This can include:
                - columns (list[str], optional): Columns to visualize. If None, all columns are used.
                - sample_size (int, optional): Number of rows to sample. If None, uses all rows.
                - random_seed (int, optional): Seed for reproducible sampling.
                - summary_file_name (str): Name of the text file to save column summaries in the output
                    directory. If it already exists, the file is overwritten.
                - figsize (tuple[int, int]): Size of the figure for plots.

        Raises:
            ValueError: On invalid DataFrame, sample size, random seed, or columns.

        Returns:
            None
        """
        jobs_df = self.validate_dataframe()
        validated_kwargs = self.validate_visualize_kwargs(kwargs, jobs_df, ColumnVisualizationKwargsModel)
        columns = validated_kwargs.columns
        sample_size = validated_kwargs.sample_size
        random_seed = validated_kwargs.random_seed
        summary_file_name = validated_kwargs.summary_file_name
        figsize = validated_kwargs.figsize
        output_dir_path = self.validate_output_dir(output_dir_path)

        jobs_df = jobs_df.copy()

        # If specific columns are provided, select them
        if columns is not None:
            jobs_df = jobs_df[columns]

        # Sample the data if sample_size is specified
        if sample_size is not None:
            if len(jobs_df) < sample_size:
                raise ValueError(f"Sample size {sample_size} is larger than the DataFrame size {len(jobs_df)}.")
            jobs_df = jobs_df.sample(sample_size, random_state=random_seed)

        self._generate_summary_stats(jobs_df, output_dir_path, summary_file_name)

        # Generate visualizations for each column
        for col in jobs_df.columns:
            col_data = jobs_df[col]

            if col in ["UUID", "Preempted"]:
                # Skip these columns as it is not useful for visualization
                print("Skipping visualization for column:", col)
                continue

            # JobID and ArrayID: treat as categorical if low cardinality, else skip plot
            if col in ["JobID", "ArrayID"]:
                if col_data.nunique() < 30:
                    sns.countplot(x=col, data=jobs_df)
                    plt.title(f"Bar plot of {col}")
                    plt.xticks(rotation=45, ha="right")
                else:
                    plt.close()
                    continue

            # IsArray: bar plot of whether a job is submitted in an array
            elif col in ["IsArray"]:
                self._generate_boolean_bar_plot(
                    jobs_df, col, "Whether a job is submitted in an array", output_dir_path
                )

            # Elapsed time and TimeLimit: histogram of durations in minutes, hours, days
            elif col in ["Elapsed", "TimeLimit"]:
                self._plot_duration_histogram(jobs_df, col, output_dir_path)

            # Timestamps: plot histogram of times and durations if possible
            elif col in ["StartTime"]:
                self._generate_start_time_histogram(
                    jobs_df,
                    col,
                    output_dir_path,
                    timestamp_range=None,
                )

            # Interactive: pie chart of interactive vs non-interactive jobs
            elif col in ["Interactive"]:
                self._generate_interactive_pie_chart(jobs_df, col, output_dir_path)

            # GPUMemUsage: histogram of GPU memory usage (categorical bins)
            elif col in ["GPUMemUsage"]:
                self._generate_gpu_memory_usage_histogram_categorical_bins(jobs_df, col, output_dir_path, figsize)

            # GPUType: bar plot of GPU types arrays
            elif col in ["GPUType"]:
                self._generate_gpu_type_bar_plot(jobs_df, col, output_dir_path)

            # Partition: bar plot of job partitions
            elif col in ["Partition"]:
                self._generate_partition_bar_plot(jobs_df, col, output_dir_path, figsize)

            # Memory: histogram of memory usage (logarithmic x-axis, percent)
            elif col in ["Memory"]:
                self._generate_non_gpu_memory_histogram(
                    jobs_df,
                    col,
                    "KiB",
                    output_dir_path,
                    figsize,
                )

            # Status: pie chart of job statuses
            elif col in ["Status"]:
                self._generate_status_pie_chart(jobs_df, col, output_dir_path)

            # ExitCode: pie chart of job exit codes
            elif col in ["ExitCode"]:
                self._generate_exit_code_pie_chart(jobs_df, col, output_dir_path)

            # QOS: pie chart of job QOS
            elif col in ["QOS"]:
                self._generate_qos_pie_chart(jobs_df, col, output_dir_path)

            # GPUs: pie chart of GPU counts in jobs
            elif col in ["GPUs"]:
                self._generate_gpu_count_pie_chart(jobs_df, col, output_dir_path)

            # Nodes: bar plot of job nodes
            elif col in ["NodeList"]:
                self._generate_node_list_bar_plot_combine_trailing_digits(jobs_df, col, output_dir_path, figsize)

            # CPUMemUsage: plot histogram of CPU memory usage
            elif col in ["CPUMemUsage"]:
                self._generate_non_gpu_memory_histogram(jobs_df, col, "B", output_dir_path, figsize)

            # Constraints: plot node features selected as constraints
            elif col in ["Constraints"]:
                self._generate_constraints_bar_plot(jobs_df, col, output_dir_path)

            # Skip visualization for other column types
            else:
                print(f"Skipping visualization for column type '{col}' in DataFrame.")
            plt.close()
            continue
