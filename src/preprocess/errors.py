from ..config.enum_constants import PreprocessingErrorTypeEnum


class JobProcessingError(ValueError):
    """
    Custom exception for errors encountered during job processing.
    """

    def __init__(self, error_type: PreprocessingErrorTypeEnum, info: str) -> None:
        self.error_type = error_type
        self.info = info
        super().__init__(f"{error_type}: {info}")
