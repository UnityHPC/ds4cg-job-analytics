from abc import ABC
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from pydantic import ValidationError

from src.config.enum_constants import TimeUnitEnum
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
        users: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        days_back: int | None = None,
        time_unit: str | TimeUnitEnum = TimeUnitEnum.MONTHS.value,
        remove_zero_values: bool = True,
        max_points: int = 100,
        annotation_style: str = "hover",  # "hover", "combined", "table", "none"
        show_secondary_y: bool = False,  # Show job counts on secondary y-axis
        exclude_fields: list[str] | None = None,  # List of fields to exclude from annotation text box
    ) -> pd.DataFrame:
        """
        Plot VRAM efficiency over time for specific users with improved annotation options.

        Args:
            users (list[str]): List of usernames to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            days_back (int, optional): Number of days back from today to filter jobs.
            time_unit (str or TimeUnitEnum, optional): Time unit to group by ('Month', 'Week', 'Day').
            remove_zero_values (bool, optional): Whether to remove users with zero efficiency values
                from the plot.
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            annotation_style (str, optional): Style for annotations ("hover", "combined", "table", "none").
            show_secondary_y (bool, optional): Whether to show job counts on secondary y-axis.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.

        Returns:
            pd.DataFrame: DataFrame containing the metrics used for the table and annotations in the plot.
        """

        # Filter data by date range or days back
        data = self.filter_jobs_by_date_range(start_date=start_date, end_date=end_date, days_back=days_back)

        # Group data by the specified time unit
        data = self.group_jobs_by_time(data, time_unit)

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
        colors = [cmap(i % 10) for i in range(len(users))]

        if exclude_fields is None:
            exclude_fields = []

        all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict = self._prepare_time_series_data(
            data,
            users,
            "alloc_vram_efficiency",  # Metric for VRAM efficiency
            time_unit,
            remove_zero_values=remove_zero_values,
        )

        # Create a dictionary to track job counts for each time group across all users
        time_group_job_counts = {tg: 0 for tg in all_time_groups}

        # First pass: calculate job counts for all time groups
        for user in users:
            user_data = data[data["User"] == user]
            if user_data.empty:
                continue
            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                job_count = group_data["JobID"].count() if not group_data.empty else 0
                time_group_job_counts[time_group] += job_count

        # Trim leading and trailing zero-job time groups while maintaining continuity in the middle
        all_time_groups = self._trim_zero_job_time_groups(all_time_groups, time_group_job_counts, remove_zero_values)

        # Format time groups as strings appropriately based on time unit
        if time_unit == TimeUnitEnum.WEEKS.value:
            # For weeks, create a more readable format like "Week of Jun 2, 2025"
            all_time_groups_str = []
            for tg in all_time_groups:
                # Extract the start date from the period (format is like '2025-06-02/2025-06-08')
                week_start = pd.to_datetime(str(tg).split("/")[0])
                all_time_groups_str.append(f"Week of {week_start.strftime('%b %d, %Y')}")
        else:
            all_time_groups_str = [str(tg) for tg in all_time_groups]

        user_dfs = []
        any_nonzero_efficiency = False
        for user in users:
            user_data = data[data["User"] == user]
            if user_data.empty:
                user_dfs.append(pd.DataFrame())
                continue
            grouped_efficiency = []
            grouped_hours = []
            grouped_job_counts = []
            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                user_gpu_hours = group_data["job_hours"].sum() if not group_data.empty else 0
                total_gpu_hours = data[data["TimeGroup"] == time_group]["job_hours"].sum()
                if total_gpu_hours > 0 and not group_data.empty:
                    efficiency = (group_data["alloc_vram_efficiency"] * user_gpu_hours / total_gpu_hours).mean()
                else:
                    efficiency = 0
                grouped_efficiency.append(efficiency)
                grouped_hours.append(user_gpu_hours)
                grouped_job_counts.append(group_data["JobID"].count() if not group_data.empty else 0)
            user_df = pd.DataFrame({
                "TimeGroup": all_time_groups,
                "TimeGroup_Str": all_time_groups_str,
                "Efficiency": grouped_efficiency,
                "GPU_Hours": grouped_hours,
                "Job_Count": grouped_job_counts,
            })

            if not user_df.empty:
                any_nonzero_efficiency = True
            user_dfs.append(user_df)

        # If all users have zero efficiency, plot VRAM Hours instead
        if not any_nonzero_efficiency:
            print("All users have zero efficiency. Plotting VRAM Hours instead.")
            return self.plot_vram_hours(
                users=users,
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                time_unit=time_unit,
                remove_zero_values=remove_zero_values,
                max_points=max_points,
                show_secondary_y=show_secondary_y,
                exclude_fields=exclude_fields,
                annotation_style=annotation_style,
            )

        for idx, user_df in enumerate(user_dfs):
            if user_df.empty:
                continue
            user = users[idx]

            # Create mapping of time groups to their position in the x-axis
            time_group_to_index = {tg: i for i, tg in enumerate(all_time_groups)}

            # Map each data point to its correct position on the x-axis
            x_positions = [time_group_to_index[tg] for tg in user_df["TimeGroup"]]

            ax1.plot(
                x_positions,
                user_df["Efficiency"],
                marker="o",
                label=f"{user} (Efficiency)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
            )
            if show_secondary_y:
                ax2.plot(
                    x_positions,
                    user_df["Job_Count"],
                    marker="s",
                    label=f"{user} (Jobs)",
                    color=colors[idx],
                    alpha=0.6,
                    linestyle="--",
                    markersize=4,
                )
            # Annotate data points
            for _, row in user_df.iterrows():
                # Use the mapped position for annotations
                x_pos = time_group_to_index[row["TimeGroup"]]
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
                        (x_pos, row["Efficiency"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )
                elif annotation_style == "combined":
                    ax1.annotate(
                        annotation_text,
                        (x_pos, row["Efficiency"]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        fontsize=7,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )
                annotation_data.append(annotation_fields)

        ax1.set_xlabel(f"Time Period ({time_unit})")
        ax1.set_ylabel("Average VRAM Efficiency")
        ax1.set_title(f"VRAM Efficiency Over Time ({time_unit})")
        ax1.set_xticks(range(len(all_time_groups_str)))
        ax1.set_xticklabels(all_time_groups_str, rotation=45, ha="right")
        lines1, labels1 = ax1.get_legend_handles_labels()
        if show_secondary_y:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.05, 1))
        else:
            ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
        table_df = pd.DataFrame(annotation_data)
        if annotation_style == "table" and annotation_data:
            print("\n" + "=" * 80)
            print("DETAILED METRICS TABLE")
            print("=" * 80)
            if not table_df.empty:
                print(table_df.to_string(index=False))
            print("=" * 80)
        return table_df

    def plot_vram_hours(
        self,
        users: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        days_back: int | None = None,
        time_unit: str | TimeUnitEnum = TimeUnitEnum.MONTHS.value,
        remove_zero_values: bool = True,
        max_points: int = 100,
        show_secondary_y: bool = False,
        exclude_fields: list[str] | None = None,
        annotation_style: str = "hover",
    ) -> pd.DataFrame:
        """
        Plot VRAM Hours over time for specific users (non-interactive version).

        Args:
            users (list[str]): List of usernames to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            days_back (int, optional): Number of days back from today to filter jobs.
            time_unit (str or TimeUnitEnum, optional): Time unit to group by ('Month', 'Week', 'Day').
            remove_zero_values (bool, optional): Whether to remove users with zero VRAM hours from the plot.
                Note that all time periods between the first and last available data point will be
                shown regardless.
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            show_secondary_y (bool, optional): Whether to show job counts on secondary y-axis.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            annotation_style (str, optional): Style for annotations ("hover", "combined", "table", "none").

        Returns:
            pd.DataFrame: DataFrame containing the metrics used for the table and annotations in the plot.
        """

        data = self.filter_jobs_by_date_range(start_date=start_date, end_date=end_date, days_back=days_back)
        data = self.group_jobs_by_time(data, time_unit)

        fig, ax1 = plt.subplots(figsize=(12, 8))
        if show_secondary_y:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Job Count", color="tab:gray")
            ax2.tick_params(axis="y", labelcolor="tab:gray")

        annotation_data = []
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(users))]
        if exclude_fields is None:
            exclude_fields = []

        all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict = self._prepare_time_series_data(
            data,
            users,
            "vram_hours",  # Metric for VRAM hours
            time_unit,
            remove_zero_values=False,  # Don't remove zero values for VRAM hours
        )

        for idx, user in enumerate(users):
            user_data = data[data["User"] == user]
            if user_data.empty:
                continue
            grouped_hours = []
            grouped_job_counts = []
            for time_group in all_time_groups:
                group_data = user_data[user_data["TimeGroup"] == time_group]
                user_gpu_hours = group_data["job_hours"].sum() if not group_data.empty else 0
                grouped_hours.append(user_gpu_hours)
                grouped_job_counts.append(group_data["JobID"].count() if not group_data.empty else 0)
            user_df = pd.DataFrame({
                "TimeGroup": all_time_groups,
                "TimeGroup_Str": all_time_groups_str,
                "VRAM_Hours": grouped_hours,
                "Job_Count": grouped_job_counts,
            })

            # Additional filtering for VRAM_Hours if needed
            if remove_zero_values:
                user_df = user_df[user_df["VRAM_Hours"] > 0]
            if len(user_df) > max_points:
                user_df = user_df.iloc[-max_points:]
            if user_df.empty:
                continue

            # Create mapping of time groups to their position in the x-axis
            time_group_to_index = {tg: i for i, tg in enumerate(all_time_groups)}

            # Map each data point to its correct position on the x-axis
            x_positions = [time_group_to_index[tg] for tg in user_df["TimeGroup"]]

            ax1.plot(
                x_positions,
                user_df["VRAM_Hours"],
                marker="o",
                label=f"{user} (VRAM Hours)",
                color=colors[idx],
                linewidth=2,
                markersize=6,
            )
            if show_secondary_y:
                ax2.plot(
                    x_positions,
                    user_df["Job_Count"],
                    marker="s",
                    label=f"{user} (Jobs)",
                    color=colors[idx],
                    alpha=0.6,
                    linestyle="--",
                    markersize=4,
                )
            # Annotate data points
            for _, row in user_df.iterrows():
                # Use the mapped position for annotations
                x_pos = time_group_to_index[row["TimeGroup"]]
                annotation_fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "VRAM_Hours": f"{row['VRAM_Hours']:.1f}",
                    "Job_Count": row["Job_Count"],
                }
                for field in exclude_fields:
                    annotation_fields.pop(field, None)
                annotation_text = "\n".join([f"{k}: {v}" for k, v in annotation_fields.items()])
                if annotation_style == "hover":
                    ax1.annotate(
                        annotation_text,
                        (x_pos, row["VRAM_Hours"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )
                elif annotation_style == "combined":
                    ax1.annotate(
                        annotation_text,
                        (x_pos, row["VRAM_Hours"]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        fontsize=7,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )
                annotation_data.append(annotation_fields)

        ax1.set_xlabel(f"Time Period ({time_unit})")
        ax1.set_ylabel("VRAM Hours")
        ax1.set_title(f"VRAM Hours Over Time ({time_unit})")
        ax1.set_xticks(range(len(all_time_groups_str)))
        ax1.set_xticklabels(all_time_groups_str, rotation=45, ha="right")
        lines1, labels1 = ax1.get_legend_handles_labels()
        if show_secondary_y:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.05, 1))
        else:
            ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
        table_df = pd.DataFrame(annotation_data)
        if annotation_style == "table" and annotation_data:
            print("\n" + "=" * 80)
            print("DETAILED METRICS TABLE")
            print("=" * 80)
            if not table_df.empty:
                print(table_df.to_string(index=False))
            print("=" * 80)
        return table_df

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
        users: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        days_back: int | None = None,
        time_unit: TimeUnitEnum | str = TimeUnitEnum.MONTHS.value,
        remove_zero_values: bool = True,
        max_points: int = 100,
        exclude_fields: list[str] | None = None,
        job_count_trace: bool = False,
    ) -> go.Figure:
        """
        Create an interactive plot with tooltips showing detailed metrics.

        Args:
            users (list[str]): List of usernames to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            days_back (int, optional): Number of days back from today to filter jobs.
            time_unit (str or TimeUnitEnum, optional): Time unit to group by ('Month', 'Week', 'Day').
            remove_zero_values (bool, optional): Whether to remove zero efficiency values from the plot.
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            job_count_trace (bool, optional): Whether to add a trace for job counts on secondary
                y-axis. False by default.

        Returns:
             go.Figure: The interactive plot with detailed tooltips.
        """
        try:
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return self.plot_vram_efficiency(
                users,
                start_date,
                end_date,
                days_back,
                time_unit,
                remove_zero_values,
                max_points,
                annotation_style="table",
            )

        # Filter data by date range or days back
        data = self.filter_jobs_by_date_range(start_date=start_date, end_date=end_date, days_back=days_back)

        # Group data by the specified time unit
        data = self.group_jobs_by_time(data, time_unit)

        if exclude_fields is None:
            exclude_fields = []

        # Use helper function to prepare consistent time series data
        all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict = self._prepare_time_series_data(
            data, users, "alloc_vram_efficiency", time_unit, remove_zero_values
        )

        # Create subplots with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]], subplot_titles=[f"VRAM Efficiency Over Time ({time_unit})"]
        )

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

        any_nonzero_efficiency = False
        user_dfs = []
        hover_texts: list[list[str]] = []

        for user in users:
            user_df = user_dfs_dict.get(user, pd.DataFrame())

            if user_df.empty:
                user_dfs.append(pd.DataFrame())
                hover_texts.append([])
                continue

            # Check if user has any non-zero efficiency
            if user_df["Efficiency"].sum() > 0:
                any_nonzero_efficiency = True

            # Filter for remove_zero_values and max_points
            if remove_zero_values:
                user_df = user_df[user_df["Efficiency"] > 0]

            if len(user_df) > max_points:
                user_df = user_df.iloc[-max_points:]

            user_dfs.append(user_df)

            # Create hover text for each point
            hover_text = []
            for _, row in user_df.iterrows():
                fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "Efficiency": f"{row['Efficiency']:.6f}",
                    "GPU_Hours": f"{row['GPU_Hours']:.1f}",
                    "Job_Count": row["Job_Count"],
                }
                for field in exclude_fields:
                    fields.pop(field, None)
                hover_text.append("<br>".join([f"{k}: {v}" for k, v in fields.items()]))
            hover_texts.append(hover_text)

        # If all users have zero efficiency, call plot_vram_hours_interactive and return
        if not any_nonzero_efficiency:
            print("All users have zero efficiency. Plotting VRAM Hours instead.")
            return self.plot_vram_hours_interactive(
                users=users,
                start_date=start_date,
                end_date=end_date,
                days_back=days_back,
                time_unit=time_unit,
                max_points=max_points,
                exclude_fields=exclude_fields,
                job_count_trace=job_count_trace,
            )

        # Now plot only for users with nonzero efficiency
        self._add_user_time_series_traces(
            fig=fig,
            users=users,
            user_dfs=user_dfs,
            hover_texts=hover_texts,
            y_key="Efficiency",
            colors=colors,
            job_count_trace=job_count_trace,
        )

        # Update layout
        fig.update_layout(
            title=f"Interactive VRAM Efficiency Analysis ({time_unit})",
            xaxis_title=f"Time Period ({time_unit})",
            hovermode="closest",
            width=1000,
            height=600,
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="VRAM Efficiency / VRAM Hours", secondary_y=False)
        fig.update_yaxes(title_text="Job Count", secondary_y=True)

        fig.show()
        return fig

    def plot_vram_hours_interactive(
        self,
        users: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        days_back: int | None = None,
        time_unit: TimeUnitEnum | str = TimeUnitEnum.MONTHS.value,
        max_points: int = 100,
        exclude_fields: list[str] | None = None,
        job_count_trace: bool = False,
    ) -> go.Figure:
        """
        Create an interactive plot of VRAM Hours over time for users.

        Args:
            users (list[str]): List of usernames to plot.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            days_back (int, optional): Number of days back from today to filter jobs.
            time_unit (str or TimeUnitEnum, optional): Time unit to group by ('Month', 'Week', 'Day').
            max_points (int, optional): Maximum number of points to plot to avoid memory issues.
            exclude_fields (list[str], optional): List of fields to exclude from annotation text box.
            job_count_trace (bool, optional): Whether to add a trace for job counts on secondary y-axis.

        Returns:
             go.Figure: The interactive plot with detailed tooltips.
        """
        try:
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return

        data = self.filter_jobs_by_date_range(start_date=start_date, end_date=end_date, days_back=days_back)
        data = self.group_jobs_by_time(data, time_unit)

        if exclude_fields is None:
            exclude_fields = []

        # Use helper function to prepare consistent time series data
        all_time_groups, all_time_groups_str, all_time_groups_datetime, user_dfs_dict = self._prepare_time_series_data(
            data,
            users,
            "vram_hours",
            time_unit,
            remove_zero_values=False,  # Don't remove zero values for VRAM hours
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=[f"VRAM Hours Over Time ({time_unit})"])

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

        user_dfs = []
        hover_texts: list[list[str]] = []

        for user in users:
            user_df = user_dfs_dict.get(user, pd.DataFrame())

            if user_df.empty:
                user_dfs.append(pd.DataFrame())
                hover_texts.append([])
                continue

            # Limit the number of points to plot
            if len(user_df) > max_points:
                user_df = user_df.iloc[-max_points:]

            user_dfs.append(user_df)

            # Create hover text for each point
            hover_text = []
            for _, row in user_df.iterrows():
                fields = {
                    "User": user,
                    "Time": row["TimeGroup_Str"],
                    "VRAM_Hours": f"{row['VRAM_Hours']:.1f}",
                    "Job_Count": row["Job_Count"],
                }
                for field in exclude_fields:
                    fields.pop(field, None)
                hover_text.append("<br>".join([f"{k}: {v}" for k, v in fields.items()]))
            hover_texts.append(hover_text)

        # Now plot for users with data
        self._add_user_time_series_traces(
            fig=fig,
            users=users,
            user_dfs=user_dfs,
            hover_texts=hover_texts,
            y_key="VRAM_Hours",
            colors=colors,
            job_count_trace=job_count_trace,
        )

        fig.update_layout(
            title=f"Interactive VRAM Hours Analysis ({time_unit})",
            xaxis_title=f"Time Period ({time_unit})",
            hovermode="closest",
            width=1000,
            height=600,
        )
        fig.update_yaxes(title_text="VRAM Hours", secondary_y=False)
        fig.update_yaxes(title_text="Job Count", secondary_y=True)
        fig.show()
        return fig
