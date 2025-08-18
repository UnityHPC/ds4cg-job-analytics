"""
Module with utilities for visualizing efficiency metrics.
"""

from abc import ABC
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import ValidationError
from matplotlib.transforms import blended_transform_factory


from .models import EfficiencyMetricsKwargsModel, UsersWithMetricsKwargsModel, PIGroupsWithMetricsKwargsModel
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
        self.validate_columns(col_kwargs.bar_label_columns, validated_jobs_df)
        self.validate_figsize(col_kwargs.figsize)
        return col_kwargs


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
        validated_kwargs = self.validate_visualize_kwargs(kwargs, jobs_with_metrics_df, EfficiencyMetricsKwargsModel)
        column = validated_kwargs.column
        bar_label_columns = validated_kwargs.bar_label_columns
        figsize = validated_kwargs.figsize
        output_dir_path = self.validate_output_dir(output_dir_path)

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
                    f"{_format_col(col)}: {val:.2f}"
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
        bar_label_columns = validated_kwargs.bar_label_columns
        figsize = validated_kwargs.figsize
        output_dir_path = self.validate_output_dir(output_dir_path)

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
                    f"{_format_col(col)}: {val:.2f}"
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
        bar_label_columns = validated_kwargs.bar_label_columns
        figsize = validated_kwargs.figsize
        output_dir_path = self.validate_output_dir(output_dir_path)

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
        for y_pos, (pi, uc) in enumerate(zip(plot_df["pi_account"], plot_df["user_count"], strict=True)):
            # First line (PI account) slightly above center
            ax.text(
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
            ax.text(
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

        # Y-axis label: place further left than custom tick labels.
        ax.set_ylabel("PI Account", rotation=90, labelpad=20)
        # Position the label using axes fraction (x< x_outside)
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
                    f"{_format_col(col)}: {val:.2f}"
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
