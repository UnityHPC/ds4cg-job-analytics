"""
Declaration of some enum class such as constants values of categorical types.
"""

from enum import Enum, EnumMeta, unique, auto
from typing import Any


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
        SCALAR: Represents a scalar value (e.g., int, float, str).
        PD_NA: Represents a pandas 'NA' value for missing data.
    """

    DICTIONARY = auto()
    LIST = auto()
    SET = auto()
    TUPLE = auto()
    NUMERIC_SCALAR = auto()
    PD_NA = auto()


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
class RequiredHoardingAnalysisColumnsEnum(Enum):
    """
    An enumeration representing required columns for hoarding analysis.

    Attributes:
        USED_CPU_MEM_GIB: Represents the used CPU memory in GiB.
        ALLOCATED_CPU_MEM_GIB: Represents the allocated CPU memory in GiB.
        CPU_MEM_EFFICIENCY: Represents the CPU memory efficiency.
    """

    USED_CPU_MEM_GIB = "used_cpu_mem_gib"
    ALLOCATED_CPU_MEM_GIB = "allocated_cpu_mem_gib"
    CPU_MEM_EFFICIENCY = "cpu_mem_efficiency"
    CPU_CORE_COUNT = "cpu_core_count"


@unique
class NodeInfoKeyEnum(Enum):
    """
    An enumeration representing important keys in node information configuration.

    Attributes:
        NAME: Represents the name of the node.
        TYPE: Represents the type of the node (e.g., CPU, GPU).
        RAM: Represents the total RAM available on the node.
        COUNT: Represents the number of nodes of this type.
        GPU_COUNT: Represents the number of GPUs available on the node.
    """

    NODES = "nodes"
    RAM = "ram"
    COUNT = "count"
    GPU_COUNT = "gpu_count"
    CORE_COUNT_PER_NODE = "cores"


class MetricsDataFrameNameMeta(EnumMeta):
    """Metaclass enforcing required members and their values on concrete metrics enums."""

    _required_values: dict[str, str] = {
        "JOBS": "jobs_with_efficiency_metrics",
        "USERS": "users_with_efficiency_metrics",
        "PI_GROUPS": "pi_accounts_with_efficiency_metrics",
    }
    _required_members: set[str] = set(_required_values.keys())

    def __init__(cls, name: str, bases: tuple, namespace: dict, **kwargs: dict[str, Any]) -> None:
        """Finalize Enum subclass creation and enforce required members and values.

        Raises:
            TypeError: If required members are missing or have unexpected values.
        """
        super().__init__(name, bases, namespace, **kwargs)
        # Skip enforcement for the abstract base itself
        if name != "MetricsDataFrameNameBase":
            member_names: set[str] = set(cls.__members__.keys())
            missing = cls._required_members - member_names
            if missing:
                raise TypeError(f"{name} must define members: {cls._required_members}; missing: {sorted(missing)}")
            # Enforce exact expected string values for required members
            for req_name, expected_value in cls._required_values.items():
                actual_value = getattr(cls, req_name).value
                if actual_value != expected_value:
                    raise TypeError(f"{name}.{req_name} must equal {expected_value!r}, got {actual_value!r}")


class MetricsDataFrameNameBase(Enum, metaclass=MetricsDataFrameNameMeta):
    """Base class for all metrics DataFrame name enums."""

    pass


@unique
class MetricsDataFrameNameEnum(MetricsDataFrameNameBase):
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
class ResourceHoardingDataFrameNameEnum(MetricsDataFrameNameBase):
    """
    An enumeration representing the names of DataFrames containing resource hoarding metrics.

    Attributes:
        JOBS: DataFrame name for jobs with efficiency metrics.
        USERS: DataFrame name for users with efficiency metrics.
        PI_GROUPS: DataFrame name for PI accounts/groups with efficiency metrics.
        JOBS_WITH_RESOURCE_HOARDING_METRICS: DataFrame name for jobs with resource hoarding metrics.
        USERS_WITH_RESOURCE_HOARDING_METRICS: DataFrame name for users with resource hoarding metrics.
        PI_GROUPS_WITH_RESOURCE_HOARDING_METRICS: DataFrame name for PI accounts/groups with resource hoarding metrics.
    """

    # Reuse canonical values to avoid duplicate string literals
    JOBS = MetricsDataFrameNameEnum.JOBS.value
    USERS = MetricsDataFrameNameEnum.USERS.value
    PI_GROUPS = MetricsDataFrameNameEnum.PI_GROUPS.value
    JOBS_WITH_RESOURCE_HOARDING_METRICS = "jobs_with_resource_hoarding_metrics"
    USERS_WITH_RESOURCE_HOARDING_METRICS = "users_with_resource_hoarding_metrics"
    PI_GROUPS_WITH_RESOURCE_HOARDING_METRICS = "pi_accounts_with_resource_hoarding_metrics"


@unique
class PreprocessingErrorTypeEnum(Enum):
    """An enumeration representing different error types that could occur during preprocessing.

    Attributes:
        MALFORMED_CONSTRAINT: Represents errors due to malformed constraints.
        UNKNOWN_GPU_TYPE: Represents errors related to unknown GPU types specified in constraints.
        NO_VALID_NODES: Represents errors when no valid nodes are found for a job.
        GPU_TYPE_NULL: Represents errors when the GPU type is null in constraints.
    """

    MALFORMED_CONSTRAINT = "Malformed Constraint"
    UNKNOWN_GPU_TYPE = "Unknown GPU Type"
    NO_VALID_NODES = "No Valid Nodes"
    GPU_TYPE_NULL = "GPU type is Null"


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


@unique
class OptionalColumnsEnum(Enum):
    """
    An enumeration representing optional columns used for filtering in preprocess code.
    
    Attributes:
        STATUS: Job status column.
        ACCOUNT: Account column.
        QOS: Quality of Service column.
        ARRAY_ID: Position in job array.
        JOB_NAME: Name of job.
        IS_ARRAY: Indicator if job is part of an array.
        INTERACTIVE: Indicator if job was interactive.
        USER: Unity user.
        EXIT_CODE: Job exit code.
        TIME_LIMIT: Job time limit (seconds).
        GPU_COMPUTE_USAGE: GPU compute usage (pct).
        CPUS: Number of CPUs.
        MEMORY: Job allocated memory (bytes).
        CPU_MEM_USAGE: CPU memory usage column.
        CPU_COMPUTE_USAGE: CPU compute usage (pct).
    """

    STATUS = "Status"
    ACCOUNT = "Account"
    QOS = "QOS"
    ARRAY_ID = "ArrayID"
    JOB_NAME = "JobName"
    IS_ARRAY = "IsArray"
    INTERACTIVE = "Interactive"
    USER = "User"
    EXIT_CODE = "ExitCode"
    TIME_LIMIT = "TimeLimit"
    GPU_COMPUTE_USAGE = "GPUComputeUsage"
    CPUS = "CPUs"
    MEMORY = "Memory"
    CPU_MEM_USAGE = "CPUMemUsage"
    CPU_COMPUTE_USAGE = "CPUComputeUsage"


@unique
class RequiredColumnsEnum(Enum):
    """
    An enumeration representing required columns that must be present in the dataframe.

    Attributes:
        GPU_TYPE: GPU type column.
        CONSTRAINTS: Job constraints column.
        START_TIME: Job start time column.
        SUBMIT_TIME: Job submit time column.
        NODE_LIST: Node list column.
        GPUS: Number of GPUs column.
        GPU_MEM_USAGE: GPU memory usage column.
        PARTITION: Partition column.
        ELAPSED: Job elapsed time column.
    """

    JOB_ID = "JobID"
    GPU_TYPE = "GPUType"
    CONSTRAINTS = "Constraints"
    START_TIME = "StartTime"
    SUBMIT_TIME = "SubmitTime"
    NODE_LIST = "NodeList"
    GPUS = "GPUs"
    GPU_MEM_USAGE = "GPUMemUsage"
    ELAPSED = "Elapsed"
    PARTITION = "Partition"


@unique
class ExcludedColumnsEnum(Enum):
    """
    An enumeration representing columns that should be omitted during preprocessing.
    
    Attributes:
        UUID: Unique identifier column.
        END_TIME: Job end time column.
        NODES: Number of nodes column.
        PREEMPTED: Job preemption status column.
    """

    UUID = "UUID"
    END_TIME = "EndTime"
    NODES = "Nodes"
    PREEMPTED = "Preempted"