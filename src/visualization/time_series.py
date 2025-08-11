from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pydantic import ValidationError
from pathlib import Path

from .models import TimeSeriesVisualizationKwargsModel
from .visualization import DataVisualizer


class TimeSeriesVisualizer(DataVisualizer[TimeSeriesVisualizationKwargsModel]):
    """
    Visualizer for plotting VRAM efficiency and VRAM hours over time for users.

    Can be used standalone with a DataFrame or with an EfficiencyAnalysis instance.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def visualize(self, output_dir_path: Path | None = None, **kwargs: dict) -> None:
        """
        Base visualize method - should be overridden by subclasses.

        Args:
            output_dir_path: Optional output directory path.
            **kwargs: Additional keyword arguments.
        """
        return None

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
        annotation_style: str = "hover",  # "hover", "combined", "table", "none"
        show_secondary_y: bool = False,  # Show job counts on secondary y-axis
        exclude_fields: list[str] | None = None,  # List of fields to exclude from annotation text box
        users: list[str] | None = None,  # Optional list of users to filter
    ) -> None:
        """
        Plot VRAM efficiency over time using the class-level DataFrame.

        Args:
            annotation_style (str, optional): Style for annotations ("hover", "combined", "table", "none").
            show_secondary_y (bool, optional): Whether to show job counts on secondary y-axis.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            users (list[str], optional): List of users to filter. If None, plot all users.

        Returns:
            None: Displays the plot.
        """
        # Filter users if provided
        time_series_df = self.df
        if time_series_df is None:
            print("No data available for visualization")
            return

        if users and time_series_df is not None:
            time_series_df = time_series_df[time_series_df["User"].isin(users)]

        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Prepare secondary axis if needed
        if show_secondary_y:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Job Count", color="tab:gray")
            ax2.tick_params(axis="y", labelcolor="tab:gray")
        else:
            ax2 = None

        # Store annotation data for table display
        annotation_data = []
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(time_series_df["User"].unique()))]

        if exclude_fields is None:
            exclude_fields = []

        users = time_series_df["User"].unique().tolist()
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

            # Ensure ax2 is only used if assigned
            if show_secondary_y and ax2:
                ax2.plot(
                    user_data["TimeGroup_Datetime"],
                    user_data["JobCount"],
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
                    "JobCount": row["JobCount"],
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
        show_secondary_y: bool = False,
        exclude_fields: list[str] | None = None,
        users: list[str] | None = None,  # Optional list of users to filter
    ) -> None:
        """
        Plot VRAM Hours over time using the class-level DataFrame.

        Args:
            show_secondary_y (bool, optional): Whether to show job counts on secondary y-axis.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            users (list[str], optional): List of users to filter. If None, plot all users.

        Returns:
            None: Displays the plot.
        """
        # Filter users if provided
        time_series_df = self.df
        if time_series_df is None:
            print("No data available for visualization")
            return

        if users:
            time_series_df = time_series_df[time_series_df["User"].isin(users)]

        fig, ax1 = plt.subplots(figsize=(12, 8))
        if show_secondary_y:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Job Count", color="tab:gray")
            ax2.tick_params(axis="y", labelcolor="tab:gray")
        else:
            ax2 = None

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(time_series_df["User"].unique()))]

        if exclude_fields is None:
            exclude_fields = []

        users = time_series_df["User"].unique().tolist()
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

            # Ensure ax2 is only used if assigned
            if show_secondary_y and ax2:
                ax2.plot(
                    user_data["TimeGroup_Datetime"],
                    user_data["JobCount"],
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
        users: list[str] | np.ndarray,
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
                        y=user_df["JobCount"],
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
        max_points: int = 100,
        exclude_fields: list[str] | None = None,
        job_count_trace: bool = False,
        users: list[str] | None = None,  # Optional list of users to filter
    ) -> go.Figure:
        """
        Create an interactive plot with tooltips showing detailed metrics for VRAM efficiency.

        Args:
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            job_count_trace (bool, optional): Whether to add a trace for job counts on secondary y-axis.
            users (list[str], optional): List of users to filter. If None, plot all users.

        Returns:
             go.Figure: The interactive plot with detailed tooltips.
        """
        from plotly.subplots import make_subplots

        # Filter users if provided
        time_series_df = self.df
        if time_series_df is None:
            print("No data available for visualization")
            return go.Figure()

        if users:
            time_series_df = time_series_df[time_series_df["User"].isin(users)]

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

        users = time_series_df["User"].unique().tolist()
        user_dfs = []
        hover_texts: list[list[str]] = []

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
                    "JobCount": row["JobCount"],
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
        max_points: int = 100,
        exclude_fields: list[str] | None = None,
        job_count_trace: bool = False,
        users: list[str] | None = None,  # Optional list of users to filter
    ) -> go.Figure:
        """
        Create an interactive plot of VRAM Hours over time for users.

        Args:
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            job_count_trace (bool, optional): Whether to add a trace for job counts on secondary y-axis.
            users (list[str], optional): List of users to filter. If None, plot all users.

        Returns:
             go.Figure: The interactive plot with detailed tooltips.
        """
        from plotly.subplots import make_subplots

        # Filter users if provided
        time_series_df = self.df
        if time_series_df is None:
            print("No data available for visualization")
            return go.Figure()

        if users:
            time_series_df = time_series_df[time_series_df["User"].isin(users)]

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

        users = time_series_df["User"].unique().tolist()
        user_dfs = []
        hover_texts: list[list[str]] = []

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
                    "JobCount": row["JobCount"],
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

    def plot_vram_efficiency_dot(
        self,
        exclude_fields: list[str] | None = None,
    ) -> None:
        """
        Plot VRAM efficiency over time as a dot plot.

        Each dot's:
        - Position: x = time, y = efficiency
        - Size: scaled by VRAM hours
        - Color: represents user

        Args:
            time_series_df (pd.DataFrame): DataFrame containing time series data.
            exclude_fields (list[str], optional): Fields to exclude from tooltip (unused in static plot).
        """
        time_series_df = self.df
        if time_series_df is None:
            print("No data available for visualization")
            return

        if exclude_fields is None:
            exclude_fields = []

        fig, ax = plt.subplots(figsize=(12, 8))

        users = time_series_df["User"].unique()
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(users))]

        for idx, user in enumerate(users):
            user_data = time_series_df[time_series_df["User"] == user]
            if user_data.empty:
                continue

            # Normalize size: square root scaling to keep bubble area proportional to VRAM hours
            vram_hours = user_data["GPUHours"]
            sizes = np.sqrt(vram_hours + 1) * 10  # tweak multiplier for visual clarity

            ax.scatter(
                user_data["TimeGroup_Datetime"],
                user_data["Efficiency"],
                s=sizes,
                color=colors[idx],
                alpha=0.7,
                label=user,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Time Period")
        ax.set_ylabel("VRAM Efficiency")
        ax.set_title("VRAM Efficiency Dot Plot (Size = VRAM Hours)")

        # Create legend with consistent small marker size
        legend = ax.legend(title="User", bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=0.5)
        for handle in legend.legend_handles:
            if handle is not None and hasattr(handle, "set_sizes"):
                handle.set_sizes([50])  # Set consistent small size for all legend markers

        plt.tight_layout()
        plt.show()

    def plot_vram_efficiency_per_job_dot(
        self,
        users: list[str],
        efficiency_metric: str,
        vram_metric: str = "job_hours",
        remove_zero_values: bool = True,
    ) -> None:
        """
        Dot plot of VRAM efficiency for all individual jobs (not grouped).

        Args:
            users (list[str]): List of users to include.
            efficiency_metric (str): Column name representing efficiency.
            vram_metric (str): Column name representing VRAM hours (used for dot size).
            remove_zero_values (bool): Whether to exclude jobs with zero or NaN efficiency.
        """
        df = self.df
        if df is None:
            print("No data available for visualization")
            return

        df = df.copy()

        # Filter by selected users
        df = df[df["User"].isin(users)]

        # Parse time from StartTime
        # df["JobStart"] = pd.to_datetime(df["StartTime"])
        df["JobStart"] = pd.to_datetime(df["StartTime"]).dt.to_period("M").dt.to_timestamp()

        # Filter zero or invalid values if needed
        if remove_zero_values:
            df = df[df[efficiency_metric] > 0]
            df = df[df[vram_metric] > 0]
            df = df[df[efficiency_metric].notna() & df[vram_metric].notna()]

        fig, ax = plt.subplots(figsize=(12, 8))

        users = df["User"].unique().tolist()
        cmap = plt.get_cmap("tab10")
        colors = {user: cmap(i % 10) for i, user in enumerate(users)}

        for user in users:
            user_df = df[df["User"] == user]

            sizes = np.sqrt(user_df[vram_metric] + 1) * 10  # tweak size factor

            ax.scatter(
                user_df["JobStart"],
                user_df[efficiency_metric],
                s=sizes,
                color=colors[user],
                alpha=0.7,
                label=user,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Job Start Time")
        ax.set_ylabel("VRAM Efficiency")
        ax.set_title("Per-Job VRAM Efficiency Dot Plot (Size = VRAM Hours)")

        # Create legend with consistent small marker size
        legend = ax.legend(title="User", bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=0.5)
        for handle in legend.legend_handles:
            if handle is not None and hasattr(handle, "set_sizes"):
                handle.set_sizes([50])  # Set consistent small size for all legend markers

        plt.tight_layout()
        plt.show()

    def plot_vram_efficiency_per_job_dot_interactive(
        self,
        users: list[str],
        efficiency_metric: str,
        vram_metric: str = "job_hours",
        remove_zero_values: bool = True,
        max_points: int = 1000,
        exclude_fields: list[str] | None = None,
    ) -> go.Figure:
        """
        Interactive dot plot of VRAM efficiency for all individual jobs (not grouped) using Plotly.

        Args:
            users (list[str]): List of users to include.
            efficiency_metric (str): Column name representing efficiency.
            vram_metric (str): Column name representing VRAM hours (used for dot size).
            remove_zero_values (bool): Whether to exclude jobs with zero or NaN efficiency.
            max_points (int): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from hover text.

        Returns:
            go.Figure: The interactive plot with detailed tooltips.
        """
        df = self.df
        if df is None:
            print("No data available for visualization")
            return go.Figure()

        df = df.copy()

        # Filter by selected users
        df = df[df["User"].isin(users)]

        # Parse time from StartTime
        df["JobStart"] = pd.to_datetime(df["StartTime"]).dt.to_period("M").dt.to_timestamp()

        # Filter zero or invalid values if needed
        if remove_zero_values:
            df = df[df[efficiency_metric] > 0]
            df = df[df[vram_metric] > 0]
            df = df[df[efficiency_metric].notna() & df[vram_metric].notna()]

        # Limit points to avoid memory issues
        if len(df) > max_points:
            df = df.sample(n=max_points, random_state=42)

        if exclude_fields is None:
            exclude_fields = []

        fig = go.Figure()

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        users_list = df["User"].unique().tolist()

        for idx, user in enumerate(users_list):
            user_df = df[df["User"] == user]

            if user_df.empty:
                continue

            # Normalize size: square root scaling to keep bubble area proportional to VRAM hours
            sizes = np.sqrt(user_df[vram_metric] + 1) * 5  # Adjust multiplier for Plotly

            # Create hover text with job details
            hover_text = []
            for _, row in user_df.iterrows():
                fields = {
                    "User": user,
                    "JobID": row.get("JobID", "N/A"),
                    "Job Start": row["JobStart"].strftime("%Y-%m") if pd.notnull(row["JobStart"]) else "N/A",
                    "Efficiency": f"{row[efficiency_metric]:.6f}",
                    "VRAM Hours": f"{row[vram_metric]:.1f}",
                    "GPU Type": row.get("GPUType", "N/A"),
                    "Exit Code": row.get("ExitCode", "N/A"),
                }

                # Remove excluded fields
                for field in exclude_fields:
                    fields.pop(field, None)

                hover_text.append("<br>".join([f"{k}: {v}" for k, v in fields.items()]))

            fig.add_trace(
                go.Scatter(
                    x=user_df["JobStart"],
                    y=user_df[efficiency_metric],
                    mode="markers",
                    name=user,
                    marker=dict(
                        size=sizes,
                        color=colors[idx % len(colors)],
                        opacity=0.7,
                        line=dict(width=1, color="black"),
                        sizemode="diameter",
                        sizemin=4,
                    ),
                    hovertext=hover_text,
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            title="Interactive Per-Job VRAM Efficiency Dot Plot (Size = VRAM Hours)",
            xaxis_title="Job Start Time",
            yaxis_title="VRAM Efficiency",
            hovermode="closest",
            width=1000,
            height=600,
            showlegend=True,
        )

        fig.show()
        return fig

