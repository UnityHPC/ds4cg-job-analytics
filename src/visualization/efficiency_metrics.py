"""
Module with utilities for visualizing efficiency metrics.
"""

from .visualization import DataVisualizer
from pydantic import ValidationError
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path
from models import EfficiencyMetricsKwargsModel, UsersWithMetricsKwargsModel


class EfficiencyMetricsVisualizer(DataVisualizer[EfficiencyMetricsKwargsModel]):
    """
    Abstract base class for visualizing efficiency metrics.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def validate_visualize_kwargs(
        self,
        kwargs: dict[str, Any],
        validated_jobs_df: pd.DataFrame,
        kwargs_model: type[EfficiencyMetricsKwargsModel]
    ) -> EfficiencyMetricsKwargsModel:
        """Validate the keyword arguments for the visualize method.
        
        Args:
            kwargs (dict[str, Any]): Keyword arguments to validate.
            validated_jobs_df (pd.DataFrame): The DataFrame to validate against.

        Raises:
            TypeError: If any keyword argument has an incorrect type.

        Returns:
            EfficiencyMetricsKwargsModel: A tuple with validated keyword arguments.
        """
        try:
            # Validate the kwargs using Pydantic model
            col_kwargs = kwargs_model(**kwargs)
        except ValidationError as e:
            allowed_fields = {
                name: str(field.annotation)
                for name, field in kwargs_model.model_fields.items()
            }
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
    """ Visualizer for jobs with efficiency metrics.

    Visualizes jobs ranked by selected efficiency metric.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        """ Visualize the efficiency metrics for jobs.

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
        yticklabels = [
            f"{jid} of\n{user}"
            for jid, user in zip(
                jobs_with_metrics_df["JobID"],
                jobs_with_metrics_df["User"],
                strict=True
            )
        ]
        plt.figure(figsize=figsize)  
        barplot = sns.barplot(
            y=yticklabels,
            x=jobs_with_metrics_df[column],
            orient="h"
        )
        plt.xlabel(column.upper())
        plt.ylabel(r'Jobs ($\mathtt{JobID}$ of $\mathtt{User})$')
        plt.title(f"Top Inefficient Jobs by {column.upper()}")

        ax = barplot
        xmax = jobs_with_metrics_df[column].max()
        # Set x-axis limit to 1.6 times the maximum value
        # This ensures that the bars do not touch the right edge of the plot
        xlim_multiplier = 1.6
        xlim = xmax * xlim_multiplier if xmax > 0 else 1
        ax.set_xlim(0, xlim)

        if bar_label_columns is not None:
            for i, (*label_values_columns, column_value) in enumerate(
                zip(
                    *(jobs_with_metrics_df[col] for col in bar_label_columns),
                    jobs_with_metrics_df[column],
                    strict=True
                )
            ):
                label_lines = [
                    f"{col}: {val:.2f}" for col, val in zip(bar_label_columns, label_values_columns, strict=True)
                ]
                label_text = "\n".join(label_lines)
                # Calculate x position for label text
                label_offset_fraction = 0.02  # Use a small offset to avoid overlap with the bar
                label_max_fraction = 0.98  # To prevent the text from being clipped at the right edge
                xpos = min(column_value + xlim * label_offset_fraction, xlim * label_max_fraction)
                ax.text(
                    xpos,
                    i,
                    label_text,
                    va="center",
                    ha="left",
                    fontsize=10,
                    color="black",
                    clip_on=True
                )

        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"jobs_ranked_by_{column}_barplot.png", bbox_inches="tight")
        plt.show()


class UsersWithMetricsVisualizer(EfficiencyMetricsVisualizer):
    """ Visualizer for users with efficiency metrics.

    Visualizes users ranked by selected efficiency metric.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        """ Visualize the efficiency metrics for users.

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

        yticklabels = users_with_metrics_df["User"]
        plt.figure(figsize=figsize)  
        barplot = sns.barplot(
            y=yticklabels,
            x=users_with_metrics_df[column],
            orient="h"
        )
        plt.xlabel(column.upper())
        plt.ylabel(f"{'Users'}")
        plt.title(f"Top Inefficient Users by {column.upper()}")

        ax = barplot
        xmax = users_with_metrics_df[column].max()
        # Set x-axis limit to 1.6 times the maximum value
        # This ensures that the bars do not touch the right edge of the plot
        xlim_multiplier = 1.6
        xlim = xmax * xlim_multiplier if xmax > 0 else 1
        ax.set_xlim(0, xlim)

        if bar_label_columns is not None:
            for i, (*label_values_columns, column_value) in enumerate(
                zip(
                    *(users_with_metrics_df[col] for col in bar_label_columns),
                    users_with_metrics_df[column],
                    strict=True
                )
            ):
                label_lines = [
                    f"{col}: {val:.2f}" for col, val in zip(bar_label_columns, label_values_columns, strict=True)
                ]
                label_text = "\n".join(label_lines)
                # Calculate x position for label text
                label_offset_fraction = 0.02  # Use a small offset to avoid overlap with the bar
                label_max_fraction = 0.98  # To prevent the text from being clipped at the right edge
                xpos = min(column_value + xlim * label_offset_fraction, xlim * label_max_fraction)
                ax.text(
                    xpos,
                    i,
                    label_text,
                    va="center",
                    ha="left",
                    fontsize=10,
                    color="black",
                    clip_on=True
                )

        plt.tight_layout()
        if output_dir_path is not None:
            plt.savefig(output_dir_path / f"jobs_ranked_by_{column}_barplot.png", bbox_inches="tight")
        plt.show()
