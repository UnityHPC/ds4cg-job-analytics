"""
Declaration of some enum class such as constants values of categorical types.
"""

from enum import Enum, unique, auto


@unique
class InteractiveEnum(Enum):
    """An enumeration representing different interactive job types.

    Attributes:
        NON_INTERACTIVE: Represents non-interactive jobs.
        SHELL: Represents shell jobs.
        JUPYTER: Represents Jupyter notebook jobs.
        MATLAB: Represents MATLAB jobs.
        RSTUDIO: Represents RStudio jobs.
        DESKTOP: Represents desktop jobs.
    """

    NON_INTERACTIVE = "non-interactive"
    SHELL = "shell"
    JUPYTER = "bc_jupyter"
    MATLAB = "bc_matlab"
    RSTUDIO = "bc_rstudio"
    DESKTOP = "bc_desktop"


@unique
class QOSEnum(Enum):
    """An enumeration representing a non-exhaustive Quality of Service (QoS) types.

    Attributes:
        NORMAL: Represents normal QoS.
        UPDATES: Represents update QoS.
        SHORT: Represents short QoS.
        LONG: Represents long QoS.
    """

    NORMAL = "normal"
    UPDATES = "updates"
    SHORT = "short"
    LONG = "long"


@unique
class StatusEnum(Enum):
    """An enumeration representing different job statuses.

    Attributes:
        COMPLETED: Represents jobs that have completed successfully.
        FAILED: Represents jobs that have failed.
        CANCELLED: Represents jobs that have been cancelled.
        TIMEOUT: Represents jobs that have timed out.
        PREEMPTED: Represents jobs that have been preempted.
        OUT_OF_MEMORY: Represents jobs that have run out of memory.
        PENDING: Represents jobs that are pending.
        NODE_FAIL: Represents jobs that have failed due to node issues.
    """

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    PREEMPTED = "PREEMPTED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    PENDING = "PENDING"
    NODE_FAIL = "NODE_FAIL"


@unique
class ExitCodeEnum(Enum):
    """An enumeration representing different job exit codes.

    Attributes:
        SUCCESS: Represents a successful job exit.
        ERROR: Represents an error during job execution.
        SIGNALED: Represents a job that was signaled to terminate.
    """

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    SIGNALED = "SIGNALED"


@unique
class AdminsAccountEnum(Enum):
    """An enumeration representing admin accounts that need to be omitted from analysis.

    Attributes:
        ROOT: Represents the root admin account.
    """

    ROOT = "root"


@unique
class AdminPartitionEnum(Enum):
    """An enumeration representing partitions used by admin in maintenance that need to be omitted from analysis.

    Attributes:
        BUILDING: Represents the building partition.
    """

    BUILDING = "building"


@unique
class FilterTypeEnum(Enum):
    """
    An enumeration representing different types of filterable data structures.

    Attributes:
        DICTIONARY: Represents a Python dictionary type.
        LIST: Represents a Python list type.
        SET: Represents a Python set type.
        TUPLE: Represents a Python tuple type.
        NUMERIC_SCALAR: Represents a scalar value (e.g., int, float, str).
        PD_NA: Represents a pandas 'NA' value for missing data.
        DATE: Represents a date type (str or datetime).
    """

    DICTIONARY = auto()
    LIST = auto()
    SET = auto()
    TUPLE = auto()
    NUMERIC_SCALAR = auto()
    PD_NA = auto()
    DATE = auto()


@unique
class PartitionTypeEnum(Enum):
    """
    An enumeration representing different types of partitions.

    Attributes:
        CPU: Represents CPU partitions.
        GPU: Represents GPU partitions.
    """

    CPU = "cpu"
    GPU = "gpu"


@unique
class MetricsDataFrameNameEnum(Enum):
    """
    An enumeration representing the names of DataFrames containing efficiency metrics.

    Attributes:
        JOBS: DataFrame name for jobs with efficiency metrics.
        USERS: DataFrame name for users with efficiency metrics.
        PI_GROUPS: DataFrame name for PI accounts/groups with efficiency metrics.
    """

    JOBS = "jobs_with_efficiency_metrics"
    USERS = "users_with_efficiency_metrics"
    PI_GROUPS = "pi_accounts_with_efficiency_metrics"


@unique
class TimeUnitEnum(Enum):
    """
    An enumeration representing different time units for efficiency metrics.

    Attributes:
        SECONDS: Represents time in seconds.
        MINUTES: Represents time in minutes.
        HOURS: Represents time in hours.
        DAYS: Represents time in days.
        WEEKS: Represents time in weeks.
        MONTHS: Represents time in months.
        YEARS: Represents time in years.
    """

    SECONDS = "Seconds"
    MINUTES = "Minutes"
    HOURS = "Hours"
    DAYS = "Days"
    WEEKS = "Weeks"
    MONTHS = "Months"
    YEARS = "Years"


@unique
class ProportionMetricsEnum(Enum):
    """
    Contains metrics for calculating proportion of data on ROC plot (y-axis).

    JOB_HOURS and JOBS elements are both used for y-axis ROC plot and as metrics in EfficiencyAnalysis class.
    """

    JOB_HOURS = "job_hours"
    VRAM_HOURS = "vram_hours"
    JOBS = "job_count"
    USERS = "User"
    PI_GROUPS = "Account"


@unique
class JobEfficiencyMetricsEnum(Enum):
    """
    Contains efficiency metrics for calculating efficiency of jobs.

    These are used as jobs metrics (in EfficiencyAnalysis class) and also metrics for x-axis in ROC Plot.

    For all members, if it is similar to a member in ProportionMetricsEnum (the member can be both
        x-axis and y-axis in ROC), it must have the same value to that member in ProportionMetricsEnum.
    """

    JOB_HOURS = "job_hours"
    VRAM_HOURS = "vram_hours"
    ALLOC_VRAM_EFFICIENCY = "alloc_vram_efficiency"
    VRAM_CONSTRAINT_EFFICIENCY = "vram_constraint_efficiency"
    VRAM_CONSTRAINT_EFFICIENCY_SCORE = "vram_constraint_efficiency_score"
    ALLOC_VRAM_EFFICIENCY_SCORE = "alloc_vram_efficiency_score"
    CPU_MEM_EFFICIENCY = "cpu_mem_efficiency"
    GPU_COUNT = "gpu_count"
    USED_VRAM_GIB = "used_vram_gib"
    # CPU column
    USED_CPU_MEMORY_GIB = "used_cpu_mem_gib"
    ALLOCATED_CPU_MEM_GIB = "allocated_cpu_mem_gib"


@unique
class UserEfficiencyMetricsEnum(Enum):
    """
    Contains efficiency metrics for calculating efficiency of users.

    These are used as users metrics (in EfficiencyAnalysis class) and also metrics for x-axis in ROC Plot (for users).

    For all members, if it is similar to a member in ProportionMetricsEnum (the member can be both
        x-axis and y-axis in ROC), it must have the same value to that member in ProportionMetricsEnum.
    """

    JOB_HOURS = "job_hours"
    VRAM_HOURS = "vram_hours"
    JOBS = "job_count"
    WEIGHTED_AVG_ALLOC_VRAM_EFFICIENCY = "expected_value_alloc_vram_efficiency"
    WEIGHTED_AVG_VRAM_CONSTRAINTS_EFFICIENCY = "expected_value_vram_constraint_efficiency"
    WEIGHTED_AVG_GPU_COUNT = "expected_value_gpu_count"
    AVG_ALLOC_VRAM_EFFICIENCY_SCORE = "avg_alloc_vram_efficiency_score"
    AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE = "avg_vram_constraint_efficiency_score"


@unique
class PIEfficiencyMetricsEnum(Enum):
    """
    Contains efficiency metrics for calculating efficiency of pi groups.

    These are used as pi groups metrics (in EfficiencyAnalysis class) and also metrics for x-axis
        in ROC Plot (for pi groups).

    For all members, if it is similar to a member in ProportionMetricsEnum (the member can be both
        x-axis and y-axis in ROC), it must have the same value to that member in ProportionMetricsEnum.
    """

    JOB_HOURS = "job_hours"
    VRAM_HOURS = "vram_hours"
    JOBS = "job_count"
    USERS = "User"
    WEIGHTED_AVG_ALLOC_VRAM_EFFICIENCY = "expected_value_alloc_vram_efficiency"
    WEIGHTED_AVG_VRAM_CONSTRAINTS_EFFICIENCY = "expected_value_vram_constraint_efficiency"
    WEIGHTED_AVG_GPU_COUNT = "expected_value_gpu_count"
    AVG_ALLOC_VRAM_EFFICIENCY_SCORE = "avg_alloc_vram_efficiency_score"
    AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE = "avg_vram_constraint_efficiency_score"


@unique
class ROCPlotTypes(Enum):
    """
    Contain different plot types for ROC
    """

    JOB = "JOB"
    USER = "USER"
    PI_GROUP = "PI_GROUP"


@unique
class ErrorTypeEnum(Enum):
    """An enumeration representing different error types.

    Attributes:
        MALFORMED_CONSTRAINT: Represents errors due to malformed constraints.
        UNKNOWN_GPU_TYPE: Represents errors related to unknown GPU types specified in constraints.
    """

    MALFORMED_CONSTRAINT = "malformed_constraint"
    UNKNOWN_GPU_TYPE = "unknown_gpu_type"


@unique
class EfficiencyCategoryEnum(Enum):
    """
    An enumeration representing efficiency category thresholds and labels.

    Attributes:
        VERY_LOW_THRESHOLD: Threshold for very low efficiency (below 0.1).
        LOW_THRESHOLD: Threshold for low efficiency (below 0.3).
        GOOD_THRESHOLD: Threshold for good efficiency (below 0.7).
        VERY_LOW_LABEL: Label for very low efficiency.
        LOW_LABEL: Label for low efficiency.
        GOOD_LABEL: Label for good efficiency.
        EXCELLENT_LABEL: Label for excellent efficiency.
    """

    # Thresholds
    VERY_LOW_THRESHOLD = 0.1
    LOW_THRESHOLD = 0.3
    GOOD_THRESHOLD = 0.7

    # Labels
    VERY_LOW_LABEL = "Very Low Efficiency"
    LOW_LABEL = "Low Efficiency"
    GOOD_LABEL = "Good Efficiency"
    EXCELLENT_LABEL = "Excellent Efficiency"

    @classmethod
    def get_efficiency_category(cls, efficiency: float) -> str:
        """
        Get the efficiency category label based on the efficiency value.

        Args:
            efficiency (float): The efficiency value to categorize.

        Returns:
            str: The efficiency category label.
        """
        if efficiency < cls.VERY_LOW_THRESHOLD.value:
            return cls.VERY_LOW_LABEL.value
        elif efficiency < cls.LOW_THRESHOLD.value:
            return cls.LOW_LABEL.value
        elif efficiency < cls.GOOD_THRESHOLD.value:
            return cls.GOOD_LABEL.value
        else:
            return cls.EXCELLENT_LABEL.value

