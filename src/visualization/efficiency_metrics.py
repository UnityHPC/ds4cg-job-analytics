"""
Module with utilities for visualizing efficiency metrics.
"""

from abc import ABC
import math
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import ValidationError
from matplotlib.transforms import blended_transform_factory
from .models import (
    EfficiencyMetricsKwargsModel,
    JobsWithMetricsKwargsModel,
    UsersWithMetricsKwargsModel,
    UsersWithMetricsHistKwargsModel,
    PIGroupsWithMetricsKwargsModel,
)

from .visualization import DataVisualizer


class EfficiencyMetricsVisualizer(DataVisualizer[EfficiencyMetricsKwargsModel], ABC):
    """
    Abstract base class for visualizing efficiency metrics.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def validate_visualize_kwargs(
        self, kwargs: dict[str, Any], validated_jobs_df: pd.DataFrame, kwargs_model: type[EfficiencyMetricsKwargsModel]
    ) -> EfficiencyMetricsKwargsModel:
        """Validate the keyword arguments for the visualize method.

        Args:
            kwargs (dict[str, Any]): Keyword arguments to validate.
            validated_jobs_df (pd.DataFrame): The DataFrame to validate against.
            kwargs_model (type[EfficiencyMetricsKwargsModel]): Pydantic model for validation.

        Raises:
            TypeError: If any keyword argument has an incorrect type.

        Returns:
            EfficiencyMetricsKwargsModel: A tuple with validated keyword arguments.
        """
        try:
            # Validate the kwargs using Pydantic model
            col_kwargs = kwargs_model(**kwargs)
        except ValidationError as e:
            allowed_fields = {name: str(field.annotation) for name, field in kwargs_model.model_fields.items()}
            allowed_fields_str = "\n".join(f"  {k}: {v}" for k, v in allowed_fields.items())
            raise TypeError(
                f"Invalid metrics visualization kwargs: {e.json(indent=2)}\n"
                f"Allowed fields and types:\n{allowed_fields_str}"
            ) from e

        self.validate_column_argument(col_kwargs.column, validated_jobs_df)
        if hasattr(col_kwargs, "bar_label_columns") and col_kwargs.bar_label_columns is not None:
            self.validate_columns(col_kwargs.bar_label_columns, validated_jobs_df)
        self.validate_figsize(col_kwargs.figsize)
        return col_kwargs

    @staticmethod
    def _human_readable_value(val: object) -> str:
        """Format numeric values human-readably.

        Rules (assumptions where unspecified):
        - "Small" numbers (abs(value) < 1_000) -> always show two decimals, rounding UP (toward +infinity)
        e.g. 1.234 -> 1.24, 0.001 -> 0.01, -1.231 -> -1.23 (up toward +inf makes negative less negative)
        - Thousands (>= 1_000 and < 1_000_000) -> comma separated with no decimals (123,456)
        - Millions and above use suffix with two decimals: 12.35 M, 3.40 B, 1.00 T
        - Handles ints, floats, numpy numeric types; returns original repr for non-numerics.
        - NA/None -> 'NA'

        Args:
            val (object): The value to format.

        Returns:
            str: Human-readable formatted representation.
        """
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "NA"
        # numpy / pandas NA
        try:
            import pandas as _pd  # local import to avoid circular issues
            try:
                _tmp_val = val  # help type checkers
                isna_func = getattr(_pd, "isna", None)
                if callable(isna_func) and isna_func(_tmp_val):  # type: ignore[call-arg]
                    return "NA"
            except TypeError:  # Non-array-like objects may raise
                pass
        except Exception:  # pragma: no cover - defensive
            pass
        if not isinstance(val, (int, float, np.integer, np.floating)):
            return str(val)
        # Cast to float for magnitude / operations
        fval = float(val)
        abs_val = abs(fval)
        # Small number branch
        if abs_val < 1_000:
            if fval >= 0:
                up = math.ceil(fval * 100) / 100.0
            else:
                # Up toward +infinity for negatives makes value less negative
                up = -math.ceil(-fval * 100) / 100.0
            if abs(up - int(up)) < 1e-9:
                return f"{int(up)}"
            return f"{up:.2f}".rstrip("0").rstrip(".")
        # Large number branches with suffixes
        suffixes = [
            (1_000_000_000_000, "T"),
            (1_000_000_000, "B"),
            (1_000_000, "M"),
        ]
        for threshold, suffix in suffixes:
            if abs_val >= threshold:
                scaled = fval / threshold
                formatted = f"{scaled:.2f}"
                if formatted.endswith(".00"):
                    formatted = formatted[:-3]
                else:
                    # Trim a single trailing 0 if present (e.g., 1.50 -> 1.5) but keep at least one decimal
                    if formatted.endswith("0"):
                        formatted = formatted[:-1]
                return f"{formatted} {suffix}"
        # Thousands (no suffix) -> comma separated, no decimals
        return f"{int(round(fval)):,}"


class JobsWithMetricsVisualizer(EfficiencyMetricsVisualizer):
    """Visualizer for jobs with efficiency metrics.

    Visualizes jobs ranked by selected efficiency metric.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        """Visualize the efficiency metrics for jobs.

        Args:
            output_dir_path (Path | None): Path to save the output plot.
            **kwargs (dict[str, Any]): Keyword arguments for visualization.
                This can include:
                - column (str): The efficiency metric to visualize.
                - bar_label_columns (list[str] | None): Columns to use for bar labels.
                - figsize (tuple[int | float, int | float]): Size of the figure.

        Returns:
            None: Displays the bar plot of jobs ranked by the specified efficiency metric.
        """
        jobs_with_metrics_df = self.validate_dataframe()
        validated_kwargs = self.validate_visualize_kwargs(kwargs, jobs_with_metrics_df, JobsWithMetricsKwargsModel)
        validated_kwargs = cast(JobsWithMetricsKwargsModel, validated_kwargs)
        column = validated_kwargs.column
        bar_label_columns = validated_kwargs.bar_label_columns
        figsize = validated_kwargs.figsize
        output_dir_path = self.validate_output_dir(output_dir_path)
        if validated_kwargs.anonymize:
            jobs_with_metrics_df["User"] = self.anonymize_str_column(jobs_with_metrics_df["User"], "user_")
            jobs_with_metrics_df["Account"] = self.anonymize_str_column(jobs_with_metrics_df["Account"], "pi_")

        # Create y-tick labels with JobID and User
        # Only include idx in the label if there are duplicate JobID values
        jobid_counts = jobs_with_metrics_df["JobID"].value_counts()
        duplicate_jobids = set(jobid_counts[jobid_counts > 1].index)
        yticklabels = [
            (f"idx {idx}\nJobID {jid} of\n{user}" if jid in duplicate_jobids else f"JobID {jid} of\n{user}")
            for idx, jid, user in zip(
                jobs_with_metrics_df.index,
                jobs_with_metrics_df["JobID"],
                jobs_with_metrics_df["User"],
                strict=True,
            )
        ]
        plt.figure(figsize=figsize)

        xmin = jobs_with_metrics_df[column].min()
        # If the minimum value is negative, we need to adjust the heights of the bars
        # to ensure they start from zero for better visualization.
        # This is particularly useful for metrics like allocated VRAM efficiency score.
        if xmin < 0:
            col_heights = pd.Series(
                [abs(xmin)] * len(jobs_with_metrics_df[column]), index=jobs_with_metrics_df[column].index
            ) - abs(jobs_with_metrics_df[column])
            print(f"Minimum value for {column}: {xmin}")
        else:
            col_heights = jobs_with_metrics_df[column]

        plot_df = pd.DataFrame({
            "col_height": col_heights.to_numpy(),
            "job_hours": jobs_with_metrics_df["job_hours"],
            "job_index_and_username": yticklabels,
        })

        barplot = sns.barplot(
            data=plot_df,
            y="job_index_and_username",
            x="col_height",
            orient="h",
        )
        readable_column = column.replace("_", " ")
        plt.xlabel(readable_column.upper())
        plt.ylabel(r"Jobs ($\mathtt{JobID}$ of $\mathtt{User})$")
        plt.title(f"Top Inefficient Jobs by {readable_column.upper()}")

        ax = barplot
        xmax = jobs_with_metrics_df[column].max()
        # Set x-axis limit to 1.6 times the maximum value
        # This ensures that the bars do not touch the right edge of the plot
        xlim_multiplier = 1.6
        xlim = abs(xmin) * xlim_multiplier if xmin < 0 else (xmax * xlim_multiplier if xmax > 0 else 1)
        ax.set_xlim(0, xlim)
        # If the minimum value is negative, we need to adjust the x-ticks accordingly
        if xmin < 0:
            num_xticks = max(4, min(12, int(abs(xmin) // (xlim * 0.10)) + 1))
            xticks = np.linspace(xmin, 0, num=num_xticks)
            ax.set_xticks([abs(xmin) - abs(val) for val in xticks])
            ax.set_xticklabels([f"{val:.2f}" if -1 < val < 1 else f"{val:.0f}" for val in xticks], rotation=45)

        if bar_label_columns is not None:
            for i, (*label_values_columns, column_value) in enumerate(
                zip(
                    *(jobs_with_metrics_df[col] for col in bar_label_columns),
                    plot_df["col_height"],
                    strict=True,
                )
            ):

                def _format_col(col: str) -> str:
                    prefix = "expected_value_"
                    if col.startswith(prefix):
                        return f'EV("{col[len(prefix) :]}")'
                    return col

                label_lines = [
                    f"{_format_col(col)}: {EfficiencyMetricsVisualizer._human_readable_value(val)}"
                    for col, val in zip(
                        bar_label_columns,
                        label_values_columns,
                        strict=True,
                    )
                ]
                label_text = "\n".join(label_lines)
                # Calculate x position for label text
                label_offset_fraction = 0.02  # Use a small offset to avoid overlap with the bar
                xpos = column_value + xlim * label_offset_fraction
                ax.text(xpos, i, label_text, va="center", ha="left", fontsize=10, color="black", clip_on=True)

        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"jobs_ranked_by_{column}_barplot.png", bbox_inches="tight")
        plt.show()


class UsersWithMetricsVisualizer(EfficiencyMetricsVisualizer):
    """Visualizer for users with efficiency metrics.

    Visualizes users ranked by selected efficiency metric.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        """Visualize the efficiency metrics for users.

        Args:
            output_dir_path (Path | None): Path to save the output plot.
            **kwargs (dict[str, Any]): Keyword arguments for visualization.
                This can include:
                - column (str): The efficiency metric to visualize.
                - bar_label_columns (list[str] | None): Columns to use for bar labels.
                - figsize (tuple[int | float, int | float]): Size of the figure.

        Returns:
            None: Displays the bar plot of users ranked by the specified efficiency metric.
        """
        users_with_metrics_df = self.validate_dataframe()
        validated_kwargs = self.validate_visualize_kwargs(kwargs, users_with_metrics_df, UsersWithMetricsKwargsModel)
        column = validated_kwargs.column
        validated_kwargs = cast(UsersWithMetricsKwargsModel, validated_kwargs)
        bar_label_columns = validated_kwargs.bar_label_columns
        figsize = validated_kwargs.figsize
        output_dir_path = self.validate_output_dir(output_dir_path)
        if validated_kwargs.anonymize:
            users_with_metrics_df["User"] = self.anonymize_str_column(users_with_metrics_df["User"], "user_")
            users_with_metrics_df["pi_account"] = self.anonymize_str_column(users_with_metrics_df["pi_account"], "pi_")

        xmin = users_with_metrics_df[column].min()
        # If the minimum value is negative, we need to adjust the heights of the bars
        # to ensure they start from zero for better visualization.
        # This is particularly useful for metrics like allocated VRAM efficiency score.
        if xmin < 0:
            col_heights = pd.Series(
                [abs(xmin)] * len(users_with_metrics_df[column]), index=users_with_metrics_df[column].index
            ) - abs(users_with_metrics_df[column])
            print(f"Minimum value for {column}: {xmin}")
        else:
            col_heights = users_with_metrics_df[column]

        plt.figure(figsize=figsize)
        plot_df = pd.DataFrame({
            "col_height": col_heights.to_numpy(),
            "job_hours": users_with_metrics_df["user_job_hours"],
            "username": users_with_metrics_df["User"],
        })
        barplot = sns.barplot(
            data=plot_df, y="username", x="col_height", orient="h", palette="Blues_r", hue="username"
        )
        readable_column = column.replace("_", " ")
        plt.xlabel(readable_column.upper())
        plt.ylabel(f"{'Users'}")
        plt.title(f"Top Inefficient Users by {readable_column.upper()}")

        ax = barplot
        xmax = users_with_metrics_df[column].max()
        # Set x-axis limit to 1.6 times the maximum value
        # This ensures that the bars do not touch the right edge of the plot
        xlim_multiplier = 1.6
        xlim = abs(xmin) * xlim_multiplier if xmin < 0 else (xmax * xlim_multiplier if xmax > 0 else 1)
        ax.set_xlim(0, xlim)
        # If the minimum value is negative, we need to adjust the x-ticks accordingly
        if xmin < 0:
            num_xticks = max(4, min(12, int(abs(xmin) // (xlim * 0.10)) + 1))
            xticks = np.linspace(xmin, 0, num=num_xticks)
            ax.set_xticks([abs(xmin) - abs(val) for val in xticks])
            ax.set_xticklabels([f"{val:.2f}" if -1 < val < 1 else f"{val:.0f}" for val in xticks], rotation=45)

        if bar_label_columns is not None:
            for i, (*label_values_columns, column_value) in enumerate(
                zip(
                    *(users_with_metrics_df[col] for col in bar_label_columns),
                    plot_df["col_height"],
                    strict=True,
                )
            ):

                def _format_col(col: str) -> str:
                    prefix = "expected_value_"
                    if col.startswith(prefix):
                        return f'EV("{col[len(prefix) :]}")'
                    return col

                label_lines = [
                    f"{_format_col(col)}: {EfficiencyMetricsVisualizer._human_readable_value(val)}"
                    for col, val in zip(
                        bar_label_columns,
                        label_values_columns,
                        strict=True,
                    )
                ]
                label_text = "\n".join(label_lines)
                # Calculate x position for label text
                label_offset_fraction = 0.02  # Use a small offset to avoid overlap with the bar
                label_max_fraction = 0.98  # To prevent the text from being clipped at the right edge
                xpos = min(column_value + xlim * label_offset_fraction, xlim * label_max_fraction)
                ax.text(xpos, i, label_text, va="center", ha="left", fontsize=10, color="black", clip_on=True)

        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"users_ranked_by_{column}_barplot.png", bbox_inches="tight")
        plt.show()

    def visualize_metric_distribution(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        """Visualize the distribution of efficiency metrics for users.

        Args:
            output_dir_path (Path | None): Path to save the output plot.
            **kwargs (dict[str, Any]): Keyword arguments for visualization.
                This can include:
                - column (str): The efficiency metric to visualize.
                - figsize (tuple[int | float, int | float]): Size of the figure.

        Returns:
            None: Displays the distribution plot of the specified efficiency metric.
        """
        users_with_metrics_df = self.validate_dataframe()
        validated_kwargs = self.validate_visualize_kwargs(
            kwargs,
            users_with_metrics_df,
            UsersWithMetricsHistKwargsModel,
        )
        column = validated_kwargs.column
        figsize = validated_kwargs.figsize
        output_dir_path = self.validate_output_dir(output_dir_path)

        # Distribution of Avg Requested VRAM Efficiency Score (actual values; all are <= 0)
        # We keep scores as-is (negative or zero) and construct bins that respect the skew while
        # still giving higher resolution near zero using log-spaced absolute values mapped back to negatives.
        values = users_with_metrics_df[column].dropna()
        max_val = users_with_metrics_df[column].max()
        min_val = users_with_metrics_df[column].min()

        if values.empty:
            print(f"No values to plot for column '{column}'.")
            print(f"{users_with_metrics_df[column].isna().sum()} NaN values found.")
            return
        # If all scores are exactly zero, a histogram is not informative
        if (values != 0).sum() == 0:
            print(f"All values are zero for column '{column}'; histogram not informative.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate bins and set the span of x-axis limits
        min_abs = None  # initialize so later symlog logic is safe for both branches
        use_log = False  # Will be determined based on the distribution
        xlim_multiplier_right = 1.6
        xlim_multiplier_left = 0.8
        xlim_threshold = 2
        if max_val < 0:
            left, right = ax.get_xlim()
            if right < 0:
                ax.set_xlim(left, 0)
            # All values strictly negative (or zero filtered out earlier). Build log-spaced bins on abs values.
            neg_scores = values[values < 0]
            if not neg_scores.empty:
                n_bins = 100
                min_abs = neg_scores.abs().min()
                max_abs = neg_scores.abs().max()
                if min_abs == max_abs:
                    # Degenerate: every negative value identical.
                    bins = np.linspace(neg_scores.min(), 0, 20)
                else:
                    abs_edges = np.logspace(np.log10(min_abs), np.log10(max_abs), n_bins)
                    # Convert absolute edges to negative edges (descending), then append 0 as the last edge
                    neg_edges = -abs_edges[::-1]
                    bins = np.unique(np.concatenate([neg_edges, [0]]))
                    use_log = True
            else:
                # Fallback (should not normally happen because zeros-only handled earlier)
                bins = np.linspace(-1, 0, 20)
        else:
            # All scores are non-negative (>= 0). Decide between linear and log bins heuristically.
            # We'll set the upper x-limit now, and defer the lower x-limit until after we decide
            # whether a log-like scale is used so we can avoid an unnecessary empty span from 0
            # to a much larger minimum positive value.
            non_negative_values = values[values >= 0]
            if non_negative_values.empty:
                # All zeros already handled above, but keep a safe fallback.
                bins = np.linspace(0, 1, 20)
            else:
                # Heuristics:
                # - Use log bins if dynamic range spans many orders of magnitude OR high dispersion.
                # - Keep linear if range is modest or very few unique values.
                min_non_negative = non_negative_values.min()
                max_val = non_negative_values.max()
                xlim_right = max_val * xlim_multiplier_right if max_val > xlim_threshold else 1
                xlim_left = min_val * xlim_multiplier_left if min_val > 0 else 0
                ax.set_xlim(xlim_left, xlim_right)

                # Guard: if min_non_negative == max_non_negative, just create narrow linear bins.
                if min_non_negative == max_val:
                    center = min_non_negative
                    width = center * 0.05 if center != 0 else 1.0
                    bins = np.linspace(center - width, center + width, 20)
                else:
                    dynamic_range = max_val / min_non_negative
                    range_span = max_val - min_non_negative
                    mean_pos = non_negative_values.mean()
                    coef_var = non_negative_values.std() / mean_pos if mean_pos != 0 else 0
                    unique_count = non_negative_values.nunique()
                    use_log = (
                        (dynamic_range >= 50 and range_span > xlim_threshold)  # wide dynamic range
                        or (dynamic_range >= 20 and coef_var > 1.0)  # moderately wide but highly dispersed
                    ) and unique_count >= 5
                    # Choose number of bins scaling sublinearly with sample size but capped.
                    est_bins = max(20, min(100, int(np.sqrt(len(non_negative_values)) * 10)))
                    if use_log:
                        bins = np.logspace(
                            np.log10(min_non_negative) if min_non_negative > 0 else 0,
                            np.log10(max_val),
                            est_bins,
                        )
                    else:
                        bins = np.linspace(values.min(), values.max(), est_bins)

        # plot and get height of tallest bin
        sns.histplot(values, bins=bins, color="#1f77b4", ax=ax)
        ax.set_xlabel(column.replace("_", " ").upper())
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {column.replace('_', ' ').upper()}")

        # Apply symmetrical log scale to x-axis to compress the long negative tail while keeping zero.
        # linthresh defines the range around zero that stays linear; choose smallest non-zero magnitude.

        # Ensure endpoint ticks (user request): when the distribution has non-negative values (max_val >= 0),
        # guarantee that the left-most and right-most x-limits are explicitly represented as ticks.
        if use_log:
            if min_abs is not None and min_abs > 0:
                linthresh = min_abs
            else:
                # Fallback: adapt threshold to data sign & scale
                if max_val < 0:
                    # Negative-only distribution: scale threshold to magnitude of most negative value
                    linthresh = max(1e-6, abs(values.min()) * 0.01)
                else:
                    # Positive distribution: base on smallest positive value (if any)
                    non_negative_values = values[values > 0]
                    if not non_negative_values.empty:
                        linthresh = max(1e-6, non_negative_values.min() * 0.1)
                    else:
                        linthresh = 1e-6
            ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0, base=10)

        # Annotation: counts (negative & zero) and total
        neg_count = (values < 0).sum()
        zero_count = (values == 0).sum()
        positive_count = (values > 0).sum()
        count = neg_count if neg_count > 0 else positive_count
        total = len(values)
        annotation_text = (
            f"# of Users: {total}\n"
            f"Counts:\n"
            f"  {'Negative' if neg_count > 0 else 'Positive'}: {count}\n"
            f"  Zero: {zero_count}"
        )

        # Automatic placement similar in spirit to legend(loc='best'):
        # Choose left/right based on where the data mass (mean) lies; choose top/bottom based on headroom.
        data_min, data_max = bins[0], bins[-1]
        midpoint = (data_min + data_max) / 2
        mean_val = values.mean()
        # Horizontal placement: opposite side of main mass
        place_right = mean_val < midpoint  # mass on left -> annotate on right
        x_pos = 0.82 if place_right else 0.02

        # Ensure limits are updated (draw figure canvas if needed)
        ax.figure.canvas.draw_idle()

        ax.text(
            x_pos,
            0.90,
            annotation_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

        # Cumulative distribution (CDF) over actual score values
        counts, bin_edges = np.histogram(values, bins=bins)
        cdf = np.cumsum(counts) / counts.sum()
        mids = (bin_edges[1:] + bin_edges[:-1]) / 2
        ax2 = ax.twinx()
        ax2.plot(mids, cdf, color="crimson", marker="o", linestyle="-", linewidth=1, markersize=3)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Cumulative Fraction", color="crimson")
        ax2.tick_params(axis="y", colors="crimson")

        # Notes:
        # - We plot the actual (negative/zero) scores instead of absolute values.
        # - symlog x-scale provides a log-like compression for large negative magnitudes while keeping zero.
        # - linthresh picks the smallest non-zero magnitude so near-zero structure is visible.
        # - CDF is computed over actual values to show accumulation from most negative toward zero.

        plt.tight_layout()
        plt.setp(ax.get_xticklabels(), visible=True)
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"user_{column}_distribution.png", bbox_inches="tight")
        plt.show()


class PIGroupsWithMetricsVisualizer(EfficiencyMetricsVisualizer):
    """Visualizer for PI groups with efficiency metrics.

    Visualizes PI groups ranked by selected efficiency metric.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        """Visualize the efficiency metrics for PI groups.

        Args:
            output_dir_path (Path | None): Path to save the output plot.
            **kwargs (dict[str, Any]): Keyword arguments for visualization.
                This can include:
                - column (str): The efficiency metric to visualize.
                - bar_label_columns (list[str] | None): Columns to use for bar labels.
                - figsize (tuple[int | float, int | float]): Size of the figure.

        Returns:
            None: Displays the bar plot of PI groups ranked by the specified efficiency metric.
        """
        pi_groups_with_metrics_df = self.validate_dataframe()
        validated_kwargs = self.validate_visualize_kwargs(
            kwargs,
            pi_groups_with_metrics_df,
            PIGroupsWithMetricsKwargsModel,
        )
        column = validated_kwargs.column
        validated_kwargs = cast(PIGroupsWithMetricsKwargsModel, validated_kwargs)
        bar_label_columns = validated_kwargs.bar_label_columns
        figsize = validated_kwargs.figsize
        output_dir_path = self.validate_output_dir(output_dir_path)
        if validated_kwargs.anonymize:
            pi_groups_with_metrics_df["pi_account"] = self.anonymize_str_column(
                pi_groups_with_metrics_df["pi_account"], "pi_"
            )

        xmin = pi_groups_with_metrics_df[column].min()
        # If the minimum value is negative, we need to adjust the heights of the bars
        # to ensure they start from zero for better visualization.
        # This is particularly useful for metrics like allocated VRAM efficiency score.
        if xmin < 0:
            col_heights = pd.Series(
                [abs(xmin)] * len(pi_groups_with_metrics_df[column]), index=pi_groups_with_metrics_df[column].index
            ) - abs(pi_groups_with_metrics_df[column])
            print(f"Minimum value for {column}: {xmin}")
        else:
            col_heights = pi_groups_with_metrics_df[column]

        plt.figure(figsize=figsize)
        plot_df = pd.DataFrame({
            "col_height": col_heights.tolist(),
            "job_hours": pi_groups_with_metrics_df["pi_acc_job_hours"].tolist(),
            "pi_account": pi_groups_with_metrics_df["pi_account"].tolist(),
            "user_count": pi_groups_with_metrics_df.get(
                "user_count", ["-"] * len(pi_groups_with_metrics_df["user_count"])
            ),
        })
        barplot = sns.barplot(
            data=plot_df, y="pi_account", x="col_height", orient="h", palette="Blues_r", hue="pi_account"
        )

        # We'll replace the default tick labels with custom two-line labels placed OUTSIDE the left spine.

        readable_column = column.replace("_", " ")
        plt.xlabel(readable_column.upper())
        plt.ylabel("PI Accounts")
        plt.title(f"Top Inefficient PI Accounts by {readable_column.upper()}")

        ax = barplot
        ax.set_yticks(range(len(plot_df["pi_account"])))
        ax.set_yticklabels([])  # clear built-in labels

        transform = blended_transform_factory(ax.transAxes, ax.transData)  # x in axes fraction, y in data coords
        x_outside = -0.02  # negative x fraction places text just left of spine; adjust if needed
        line_gap = 0.4  # vertical separation between the two lines
        # Collect created text objects so we can measure their rendered widths and
        # place the y-axis label far enough to the left dynamically.
        label_text_objs: list[Any] = []
        for y_pos, (pi, uc) in enumerate(zip(plot_df["pi_account"], plot_df["user_count"], strict=True)):
            # First line (PI account) slightly above center
            t1 = ax.text(
                x_outside,
                y_pos - line_gap / 2,
                pi,
                ha="right",
                va="center",
                transform=transform,
                fontsize=10,
                clip_on=False,
            )
            # Second line (Users) slightly below center
            t2 = ax.text(
                x_outside,
                y_pos + line_gap / 2,
                f"# of Users: {uc}",
                ha="right",
                va="center",
                transform=transform,
                fontsize=9,
                color="dimgray",
                clip_on=False,
            )
            label_text_objs.extend([t1, t2])

        # Y-axis label: place further left than custom tick labels.
        ax.set_ylabel("PI Account", rotation=90, labelpad=20)
        # Dynamically determine how far left the ylabel should go based on the
        # widest of the custom y-axis labels we just drew. We measure text in display
        # coordinates then convert width to an axes-fraction offset.
        try:
            fig = ax.figure
            # Ensure renderer exists (needed for text bounding boxes)
            fig.canvas.draw()  # force a draw so extents are up-to-date
            # Some backends expose get_renderer; otherwise use _get_renderer (Agg) or derive via draw
            renderer = getattr(fig.canvas, "get_renderer", None)
            if renderer is not None:
                renderer = renderer()
            else:
                # Agg / fallback
                renderer = getattr(fig.canvas, "_get_renderer", None)
                if renderer is not None:
                    renderer = renderer()
                else:
                    # Last resort: trigger draw which returns a renderer for some backends
                    fig.canvas.draw()
                    renderer = getattr(fig.canvas, "renderer", None)
            min_left_axes = 0.0  # track furthest left (most negative) axes-fraction coordinate reached by labels
            for txt in label_text_objs:
                # Position given is in axes fraction for x (because of blended transform); get right edge in display
                x_axes, _ = txt.get_position()
                right_disp_x = ax.transAxes.transform((x_axes, 0))[0]
                bbox = txt.get_window_extent(renderer=renderer)
                left_disp_x = right_disp_x - bbox.width
                left_axes = ax.transAxes.inverted().transform((left_disp_x, 0))[0]
                if left_axes < min_left_axes:
                    min_left_axes = left_axes
            # Add a small margin (in axes fraction) beyond the furthest left label
            margin = 0.04
            ylabel_x = min_left_axes - margin
            ax.yaxis.set_label_coords(ylabel_x, 0.5)
        except Exception:
            # Fallback to a reasonable static offset if measurement fails
            ax.yaxis.set_label_coords(x_outside - 0.30, 0.5)

        # Hide y-axis tick labels (already blank) but keep small outward ticks if desired
        ax.tick_params(axis="y", which="both", direction="out", length=4, pad=2)
        plt.subplots_adjust(left=0.7)

        xmax = pi_groups_with_metrics_df[column].max()
        # Set x-axis limit to 1.6 times the maximum value
        # This ensures that the bars do not touch the right edge of the plot
        xlim_multiplier = 1.6
        xlim = abs(xmin) * xlim_multiplier if xmin < 0 else (xmax * xlim_multiplier if xmax > 0 else 1)
        ax.set_xlim(0, xlim)
        # If the minimum value is negative, we need to adjust the x-ticks accordingly
        if xmin < 0:
            num_xticks = max(4, min(12, int(abs(xmin) // (xlim * 0.10)) + 1))
            xticks = np.linspace(xmin, 0, num=num_xticks)
            ax.set_xticks([abs(xmin) - abs(val) for val in xticks])
            ax.set_xticklabels([f"{val:.2f}" if -1 < val < 1 else f"{val:.0f}" for val in xticks], rotation=45)

        if bar_label_columns is not None:
            for i, (*label_values_columns, column_value) in enumerate(
                zip(
                    *(pi_groups_with_metrics_df[col] for col in bar_label_columns),
                    plot_df["col_height"],
                    strict=True,
                )
            ):

                def _format_col(col: str) -> str:
                    prefix = "expected_value_"
                    if col.startswith(prefix):
                        return f'EV("{col[len(prefix) :]}")'
                    return col

                label_lines = [
                    f"{_format_col(col)}: {EfficiencyMetricsVisualizer._human_readable_value(val)}"
                    for col, val in zip(
                        bar_label_columns,
                        label_values_columns,
                        strict=True,
                    )
                ]
                label_text = "\n".join(label_lines)
                # Calculate x position for label text
                label_offset_fraction = 0.02  # Use a small offset to avoid overlap with the bar
                label_max_fraction = 0.98  # To prevent the text from being clipped at the right edge
                xpos = min(column_value + xlim * label_offset_fraction, xlim * label_max_fraction)
                ax.text(xpos, i, label_text, va="center", ha="left", fontsize=10, color="black", clip_on=True)

        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"pi_groups_ranked_by_{column}_barplot.png", bbox_inches="tight")
        plt.show()
