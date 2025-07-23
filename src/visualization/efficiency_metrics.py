"""
Module with utilities for visualizing efficiency metrics.
"""

from .visualization import DataVisualizer
from pydantic import BaseModel, Field, ValidationError, ConfigDict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path


class JobsWithMetricsVisualizer(DataVisualizer):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    @staticmethod
    def validate_column_argument(column: str | None, jobs_df: pd.DataFrame) -> str | None:
        """Validate the provided column against the DataFrame.

        Args:
            column (str | None): Column name to validate.

        Raises:
            TypeError: If 'column' is not a string or None.
            ValueError: If the column is not present in the DataFrame.

        Returns:
            str | None: Validated column name.
        """

        if column is not None and not isinstance(column, str):
            raise TypeError("'column' must be a string or None")

        if column is not None and jobs_df is not None and column not in jobs_df.columns:
            raise ValueError("The specified column is not present in the DataFrame.")

        return column

    class JobsWithMetricsKwargsModel(BaseModel):
        model_config = ConfigDict(strict=True, extra='forbid')
        column: str
        bar_label_column_1: str
        bar_label_column_2: str 
        figsize: tuple[int | float, int | float] = Field(default=(8, 10))

    def validate_visualize_kwargs(
        self,
        kwargs: dict[str, Any],
        validated_jobs_df: pd.DataFrame,
    ) -> JobsWithMetricsKwargsModel:
        """Validate the keyword arguments for the visualize method.
        
        Args:
            kwargs (dict[str, Any]): Keyword arguments to validate.
            validated_jobs_df (pd.DataFrame): The DataFrame to validate against.

        Raises:
            TypeError: If any keyword argument has an incorrect type.

        Returns:
            JobsWithMetricsKwargsModel: A tuple with validated keyword arguments.
        """
        try:
            # Validate the kwargs using Pydantic model
            col_kwargs = self.JobsWithMetricsKwargsModel(**kwargs)
        except ValidationError as e:
            allowed_fields = {
                name: str(field.annotation)
                for name, field in self.JobsWithMetricsKwargsModel.model_fields.items()
            }
            allowed_fields_str = "\n".join(f"  {k}: {v}" for k, v in allowed_fields.items())
            raise TypeError(
                f"Invalid job metrics visualization kwargs: {e.json(indent=2)}\n"
                f"Allowed fields and types:\n{allowed_fields_str}"
            ) from e

        self.validate_column_argument(col_kwargs.column, validated_jobs_df)
        self.validate_figsize(col_kwargs.figsize)
 
        return col_kwargs

    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict[str, Any]) -> None:
        jobs_with_metrics_df = self.validate_dataframe()
        validated_kwargs = self.validate_visualize_kwargs(kwargs, jobs_with_metrics_df)
        column = validated_kwargs.column
        bar_label_column_1 = validated_kwargs.bar_label_column_1
        bar_label_column_2 = validated_kwargs.bar_label_column_2
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
        xlim = xmax * 1.6 if xmax > 0 else 1
        ax.set_xlim(0, xlim)

        for i, (bar_label__1, bar_label_2, column_value) in enumerate(
            zip(
                jobs_with_metrics_df[bar_label_column_1],
                jobs_with_metrics_df[bar_label_column_2],
                jobs_with_metrics_df[column],
                strict=True
            )
        ):
            xpos = min(column_value + xlim * 0.02, xlim * 0.98)
            ax.text(
                xpos,
                i,
                f"{bar_label_column_1}: {bar_label__1:.2f}\n{bar_label_column_2}: {bar_label_2:.2f}",
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




