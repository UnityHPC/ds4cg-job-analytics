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

    model_config = ConfigDict(strict=True, extra="allow")
    column: str
    figsize: tuple[int | float, int | float] = Field(default=(8, 10))
    anonymize: bool = Field(default=False)


class JobsWithMetricsKwargsModel(EfficiencyMetricsKwargsModel):
    """Model for keyword arguments used in jobs with metrics visualizations."""

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


class UsersWithMetricsHistKwargsModel(EfficiencyMetricsKwargsModel):
    """Model for keyword arguments used in user metrics histogram visualizations."""

    model_config = ConfigDict(strict=True, extra="forbid")
    column: str
    figsize: tuple[int | float, int | float] = Field(default=(8, 5))


class PIGroupsWithMetricsKwargsModel(EfficiencyMetricsKwargsModel):
    """Model for keyword arguments used in PI group metrics visualizations."""

    model_config = ConfigDict(strict=True, extra="forbid")
    column: str
    bar_label_columns: list[str] | None
    figsize: tuple[int | float, int | float] = Field(default=(8, 8))
