from pydantic import BaseModel, Field, ConfigDict


class ColumnVisualizationKwargsModel(BaseModel):
    """Model for keyword arguments used in column visualizations."""

    model_config = ConfigDict(strict=True, extra="forbid")
    columns: list[str] | None = None
    sample_size: int | None = None
    random_seed: int | None = None
    summary_file_name: str = "columns_stats_summary.txt"
    figsize: tuple[int | float, int | float] = (7, 4)


class EfficiencyMetricsKwargsModel(BaseModel):
    """Model for keyword arguments used in efficiency metrics visualizations."""

    model_config = ConfigDict(strict=True, extra="forbid")
    column: str
    bar_label_columns: list[str] | None
    figsize: tuple[int | float, int | float] = Field(default=(8, 10))


class UsersWithMetricsKwargsModel(EfficiencyMetricsKwargsModel):
    """Model for keyword arguments used in user metrics visualizations."""

    model_config = ConfigDict(strict=True, extra="forbid")
    column: str
    bar_label_columns: list[str] | None
    figsize: tuple[int | float, int | float] = Field(default=(8, 8))


class TimeSeriesVisualizationKwargsModel(BaseModel):
    """Model for keyword arguments used in time series visualizations (VRAM efficiency/hours over time)."""

    model_config = ConfigDict(strict=True, extra="forbid")
    users: list[str]
    start_date: str | None = None
    end_date: str | None = None
    days_back: int | None = None
    time_unit: str = "Months"
    remove_zero_values: bool = True
    max_points: int = 100
    annotation_style: str = "hover"  # "hover", "combined", "table", "none"
    show_secondary_y: bool = False
    exclude_fields: list[str] | None = None
