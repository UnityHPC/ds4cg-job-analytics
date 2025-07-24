from pydantic import BaseModel, Field, ConfigDict


class ColumnVisualizationKwargsModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    columns: list[str] | None = None
    sample_size: int | None = None
    random_seed: int | None = None
    summary_file_name: str = "columns_stats_summary.txt"
    figsize: tuple[int | float, int | float] = (7, 4)


class EfficiencyMetricsKwargsModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    column: str
    bar_label_columns: list[str] | None
    figsize: tuple[int | float, int | float] = Field(default=(8, 10))


class UsersWithMetricsKwargsModel(EfficiencyMetricsKwargsModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    column: str
    bar_label_columns: list[str] | None
    figsize: tuple[int | float, int | float] = Field(default=(8, 8))