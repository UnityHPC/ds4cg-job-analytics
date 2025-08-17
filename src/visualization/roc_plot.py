# """
# Module with utilities for visualizing ROC graph.
# """

# import numpy as np
# from abc import ABC
# from pandas import DataFrame
# from .visualization import DataVisualizer
# from pydantic import ValidationError
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# from typing import Any
# from pathlib import Path
# from src.analysis.roc_plot import ROCVisualizer

# from .models import ROCVisualizationKwargsModel


# class ROCVisualizer(DataVisualizer[ROCVisualizationKwargsModel], ABC):
#     """
#     Abstract base class for ROC visualization.
#     """

#     def __init__(self, jobs_df: pd.DataFrame) -> None:
#         super().__init__(jobs_df)  # add jobs_df to self.df
#         self.df = self.validate_dataframe()  # validate that dataframe is neither None nor empty
#         self.analysis_instance = ROCVisualizer(self.df)

#     def validate_visualize_kwargs(
#         self, kwargs: dict[str, Any],
#         validated_jobs_df: pd.DataFrame, kwargs_model: type[ROCVisualizationKwargsModel]
#     ) -> ROCVisualizationKwargsModel:
#         """Validate the keyword arguments for the visualize method.

#         Args:
#             kwargs (dict[str, Any]): Keyword arguments to validate.
#             validated_jobs_df (pd.DataFrame): The DataFrame to validate against.
#             kwargs_model (type[ROCVisualizationKwargsModel]): Pydantic model for validation.

#         Raises:
#             TypeError: If any keyword argument has an incorrect type.

#         Returns:
#             ROCVisualizationKwargsModel: A tuple with validated keyword arguments.
#         """

#         try:
#             # Validate the kwargs using Pydantic model
#             col_kwargs = kwargs_model(**kwargs)
#         except ValidationError as e:
#             allowed_fields = {name: str(field.annotation) for name, field in kwargs_model.model_fields.items()}
#             allowed_fields_str = "\n".join(f"  {k}: {v}" for k, v in allowed_fields.items())
#             raise TypeError(
#                 f"Invalid metrics visualization kwargs: {e.json(indent=2)}\n"
#                 f"Allowed fields and types:\n{allowed_fields_str}"
#             ) from e

#         return col_kwargs


# class SingleROCPlot(ROCVisualizer):
#     def __init__(self, data: pd.DataFrame) -> None:
#         super().__init__(data)
