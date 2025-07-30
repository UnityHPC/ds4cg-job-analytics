from abc import ABC
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from pydantic import ValidationError

from .models import TimeSeriesVisualizationKwargsModel
from .visualization import DataVisualizer


class TimeSeriesVisualizer(DataVisualizer[TimeSeriesVisualizationKwargsModel], ABC):
    """
    Visualizer for plotting VRAM efficiency and VRAM hours over time for users.

    Can be used standalone with a DataFrame or with an EfficiencyAnalysis instance.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def validate_visualize_kwargs(
        self,
        kwargs: dict[str, Any],
        validated_jobs_df: pd.DataFrame,
        kwargs_model: type[TimeSeriesVisualizationKwargsModel],
    ) -> TimeSeriesVisualizationKwargsModel:
        """Validate the keyword arguments for the visualize method.

        Args:
            kwargs (dict[str, Any]): Keyword arguments to validate.
            validated_jobs_df (pd.DataFrame): The DataFrame to validate against.
            kwargs_model (type[TimeSeriesVisualizationKwargsModel]): Pydantic model for validation.

        Raises:
            TypeError: If any keyword argument has an incorrect type.

        Returns:
            TimeSeriesVisualizationKwargsModel: A tuple with validated keyword arguments.
        """

        try:
            # Validate the kwargs using Pydantic model
            col_kwargs = kwargs_model(**kwargs)
        except ValidationError as e:
            allowed_fields = {name: str(field.annotation) for name, field in kwargs_model.model_fields.items()}
            allowed_fields_str = "\n".join(f"  {k}: {v}" for k, v in allowed_fields.items())
            raise TypeError(
                f"Invalid time series visualization kwargs: {e.json(indent=2)}\n"
                f"Allowed fields and types:\n{allowed_fields_str}"
            ) from e

        return col_kwargs

    def plot_vram_efficiency(
        self,
        time_series_df: pd.DataFrame,
        annotation_style: str = "hover",  # "hover", "combined", "table", "none"
        show_secondary_y: bool = False,  # Show job counts on secondary y-axis
        exclude_fields: list[str] | None = None,  # List of fields to exclude from annotation text box
    ) -> None:
        """
        Plot VRAM efficiency over time using the prepared time series DataFrame.

        Args:
            time_series_df (pd.DataFrame): DataFrame containing time series data.
            annotation_style (str, optional): Style for annotations ("hover", "combined", "table", "none").
            show_secondary_y (bool, optional): Whether to show job counts on secondary y-axis.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.

        Returns:
            None: Displays the plot.
        """
        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Prepare secondary axis if needed
        if show_secondary_y:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Job Count", color="tab:gray")
            ax2.tick_params(axis="y", labelcolor="tab:gray")

        # Store annotation data for table display
        annotation_data = []
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(time_series_df["User"].unique()))]

        if exclude_fields is None:
            exclude_fields = []

        users = time_series_df["User"].unique()
        for idx, user in enumerate(users):
            user_data = time_series_df[time_series_df["User"] == user]
            if user_data.empty:
                continue

            ax1.plot(
                user_data["TimeGroup_Datetime"],
                user_data["Efficiency"],
                marker="o",
                label=f"{user} (Efficiency)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
            )

            if show_secondary_y:
                ax2.plot(
                    user_data["TimeGroup_Datetime"],
                    user_data["Job_Count"],
                    marker="s",
                    label=f"{user} (Jobs)",
                    color=colors[idx],
                    alpha=0.6,
                    linestyle="--",
                    markersize=4,
                )

            # Annotate data points
            for _, row in user_data.iterrows():
                annotation_fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "Efficiency": f"{row['Efficiency']:.6f}",
                    "GPU_Hours": f"{row.get('GPU_Hours', 0):.1f}",
                    "Job_Count": row["Job_Count"],
                }
                for field in exclude_fields:
                    annotation_fields.pop(field, None)
                annotation_text = "\n".join([f"{k}: {v}" for k, v in annotation_fields.items()])
                if annotation_style == "hover":
                    ax1.annotate(
                        annotation_text,
                        (row["TimeGroup_Datetime"], row["Efficiency"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )
                elif annotation_style == "combined":
                    ax1.annotate(
                        annotation_text,
                        (row["TimeGroup_Datetime"], row["Efficiency"]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        fontsize=7,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )
                annotation_data.append(annotation_fields)

        ax1.set_xlabel("Time Period")
        ax1.set_ylabel("Average VRAM Efficiency")
        ax1.set_title("VRAM Efficiency Over Time")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    def plot_vram_hours(
        self,
        time_series_df: pd.DataFrame,
        show_secondary_y: bool = False,
        exclude_fields: list[str] | None = None,
    ) -> None:
        """
        Plot VRAM Hours over time using the prepared time series DataFrame.

        Args:
            time_series_df (pd.DataFrame): DataFrame containing time series data.
            show_secondary_y (bool, optional): Whether to show job counts on secondary y-axis.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.

        Returns:
            None: Displays the plot.
        """
        fig, ax1 = plt.subplots(figsize=(12, 8))
        if show_secondary_y:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Job Count", color="tab:gray")
            ax2.tick_params(axis="y", labelcolor="tab:gray")

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(time_series_df["User"].unique()))]

        if exclude_fields is None:
            exclude_fields = []

        users = time_series_df["User"].unique()
        for idx, user in enumerate(users):
            user_data = time_series_df[time_series_df["User"] == user]
            if user_data.empty:
                continue

            ax1.plot(
                user_data["TimeGroup_Datetime"],
                user_data["GPU_Hours"],
                marker="o",
                label=f"{user} (VRAM Hours)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
            )

            if show_secondary_y:
                ax2.plot(
                    user_data["TimeGroup_Datetime"],
                    user_data["Job_Count"],
                    marker="s",
                    label=f"{user} (Jobs)",
                    color=colors[idx],
                    alpha=0.6,
                    linestyle="--",
                    markersize=4,
                )

        ax1.set_xlabel("Time Period")
        ax1.set_ylabel("VRAM Hours")
        ax1.set_title("VRAM Hours Over Time")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    def _add_user_time_series_traces(
        self,
        fig: go.Figure,
        users: list[str],
        user_dfs: list[pd.DataFrame],
        hover_texts: list[list[str]],
        y_key: str,
        colors: list[str],
        job_count_trace: bool = False,
    ) -> None:
        """
        Helper to add user time series traces to a plotly figure.

        Args:
            fig: plotly figure
            users: list of usernames
            user_dfs: list of user DataFrames (one per user)
            hover_texts: list of hover text lists (one per user)
            y_key: str, column to plot on y-axis ("Efficiency" or "VRAM_Hours")
            colors: list of color hex codes
            job_count_trace: bool, whether to add job count trace
        """

        for idx, (user_df, hover_text, user) in enumerate(zip(user_dfs, hover_texts, users, strict=True)):
            if user_df is None or user_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=user_df["TimeGroup_Datetime"],  # Use datetime for proper chronological ordering
                    y=user_df[y_key],
                    mode="lines+markers",
                    name=f"{user} ({y_key.replace('_', ' ')})",
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=8),
                    hovertext=hover_text,
                    hoverinfo="text",
                ),
                secondary_y=False,
            )
            if job_count_trace:
                if user_df is None or user_df.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=user_df["TimeGroup_Datetime"],  # Use datetime for proper chronological ordering
                        y=user_df["Job_Count"],
                        mode="lines+markers",
                        name=f"{user} (Job Count)",
                        line=dict(color=colors[idx % len(colors)], width=1, dash="dash"),
                        marker=dict(size=6, symbol="square"),
                        opacity=0.6,
                        hovertext=hover_text,
                        hoverinfo="text",
                    ),
                    secondary_y=True,
                )

    def plot_vram_efficiency_interactive(
        self,
        time_series_df: pd.DataFrame,
        max_points: int = 100,
        exclude_fields: list[str] | None = None,
        job_count_trace: bool = False,
    ) -> go.Figure:
        """
        Create an interactive plot with tooltips showing detailed metrics for VRAM efficiency.

        Args:
            time_series_df (pd.DataFrame): DataFrame containing time series data.
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            job_count_trace (bool, optional): Whether to add a trace for job counts on secondary y-axis.

        Returns:
             go.Figure: The interactive plot with detailed tooltips.
        """
        from plotly.subplots import make_subplots

        if exclude_fields is None:
            exclude_fields = []

        fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=["VRAM Efficiency Over Time"])

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        users = time_series_df["User"].unique()
        user_dfs = []
        hover_texts = []

        for user in users:
            user_data = time_series_df[time_series_df["User"] == user]

            if user_data.empty:
                user_dfs.append(pd.DataFrame())
                hover_texts.append([])
                continue

            if len(user_data) > max_points:
                user_data = user_data.iloc[-max_points:]

            hover_text = []
            for _, row in user_data.iterrows():
                fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "Efficiency": f"{row['Metric']:.6f}",
                    "GPU_Hours": f"{row['GPUHours']:.1f}",
                    "Job_Count": row["JobCount"],
                }
                for field in exclude_fields:
                    fields.pop(field, None)
                hover_text.append("<br>".join([f"{k}: {v}" for k, v in fields.items()]))

            user_dfs.append(user_data)
            hover_texts.append(hover_text)

        self._add_user_time_series_traces(
            fig=fig,
            users=users,
            user_dfs=user_dfs,
            hover_texts=hover_texts,
            y_key="Metric",
            colors=colors,
            job_count_trace=job_count_trace,
        )

        fig.update_layout(
            title="Interactive VRAM Efficiency Analysis",
            xaxis_title="Time Period",
            hovermode="closest",
            width=1000,
            height=600,
        )

        fig.update_yaxes(title_text="VRAM Efficiency", secondary_y=False)
        fig.update_yaxes(title_text="Job Count", secondary_y=True)

        fig.show()
        return fig

    def plot_vram_hours_interactive(
        self,
        time_series_df: pd.DataFrame,
        max_points: int = 100,
        exclude_fields: list[str] | None = None,
        job_count_trace: bool = False,
    ) -> go.Figure:
        """
        Create an interactive plot of VRAM Hours over time for users.

        Args:
            time_series_df (pd.DataFrame): DataFrame containing time series data.
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            job_count_trace (bool, optional): Whether to add a trace for job counts on secondary y-axis.

        Returns:
             go.Figure: The interactive plot with detailed tooltips.
        """
        from plotly.subplots import make_subplots

        if exclude_fields is None:
            exclude_fields = []

        fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=["VRAM Hours Over Time"])

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        users = time_series_df["User"].unique()
        user_dfs = []
        hover_texts = []

        for user in users:
            user_data = time_series_df[time_series_df["User"] == user]

            if user_data.empty:
                user_dfs.append(pd.DataFrame())
                hover_texts.append([])
                continue

            if len(user_data) > max_points:
                user_data = user_data.iloc[-max_points:]

            hover_text = []
            for _, row in user_data.iterrows():
                fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "VRAM_Hours": f"{row['GPUHours']:.1f}",
                    "Job_Count": row["JobCount"],
                }
                for field in exclude_fields:
                    fields.pop(field, None)
                hover_text.append("<br>".join([f"{k}: {v}" for k, v in fields.items()]))

            user_dfs.append(user_data)
            hover_texts.append(hover_text)

        self._add_user_time_series_traces(
            fig=fig,
            users=users,
            user_dfs=user_dfs,
            hover_texts=hover_texts,
            y_key="GPUHours",
            colors=colors,
            job_count_trace=job_count_trace,
        )

        fig.update_layout(
            title="Interactive VRAM Hours Analysis",
            xaxis_title="Time Period",
            hovermode="closest",
            width=1000,
            height=600,
        )

        fig.update_yaxes(title_text="VRAM Hours", secondary_y=False)
        fig.update_yaxes(title_text="Job Count", secondary_y=True)

        fig.show()
        return fig
