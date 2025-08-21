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
    GPU_TYPE_NULL = "GPU Type is Null"


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


@unique
class ProportionMetricsEnum(Enum):
    """
    Contains metrics for calculating proportion of data on ETC plot (y-axis).

    JOB_HOURS and JOBS elements are both used for y-axis ETC plot and as metrics in EfficiencyAnalysis class.
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

    These are used as jobs metrics (in EfficiencyAnalysis class) and also metrics for x-axis in ETC Plot.

    For all members, if it is similar to a member in ProportionMetricsEnum (the member can be both
        x-axis and y-axis in ETC), it must have the same value to that member in ProportionMetricsEnum.
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

    These are used as users metrics (in EfficiencyAnalysis class) and also metrics for x-axis in ETC Plot (for users).

    For all members, if it is similar to a member in ProportionMetricsEnum (the member can be both
        x-axis and y-axis in ETC), it must have the same value to that member in ProportionMetricsEnum.
    """

    JOB_HOURS = "job_hours"
    VRAM_HOURS = "vram_hours"
    JOBS = "job_count"
    EXPECTED_VALUE_ALLOC_VRAM_EFFICIENCY = "expected_value_alloc_vram_efficiency"
    EXPECTED_VALUE_VRAM_CONSTRAINTS_EFFICIENCY = "expected_value_vram_constraint_efficiency"
    EXPECTED_VALUE_GPU_COUNT = "expected_value_gpu_count"
    AVG_ALLOC_VRAM_EFFICIENCY_SCORE = "avg_alloc_vram_efficiency_score"
    AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE = "avg_vram_constraint_efficiency_score"


@unique
class PIEfficiencyMetricsEnum(Enum):
    """
    Contains efficiency metrics for calculating efficiency of pi groups.

    These are used as pi groups metrics (in EfficiencyAnalysis class) and also metrics for x-axis
        in ETC Plot (for pi groups).

    For all members, if it is similar to a member in ProportionMetricsEnum (the member can be both
        x-axis and y-axis in ETC), it must have the same value to that member in ProportionMetricsEnum.
    """

    JOB_HOURS = "job_hours"
    VRAM_HOURS = "vram_hours"
    JOBS = "job_count"
    USERS = "User"
    EXPECTED_VALUE_ALLOC_VRAM_EFFICIENCY = "expected_value_alloc_vram_efficiency"
    EXPECTED_VALUE_VRAM_CONSTRAINTS_EFFICIENCY = "expected_value_vram_constraint_efficiency"
    EXPECTED_VALUE_GPU_COUNT = "expected_value_gpu_count"
    AVG_ALLOC_VRAM_EFFICIENCY_SCORE = "avg_alloc_vram_efficiency_score"
    AVG_VRAM_CONSTRAINT_EFFICIENCY_SCORE = "avg_vram_constraint_efficiency_score"


@unique
class ETCPlotTypes(Enum):
    """
    Contain different plot types for ETC
    """

    JOB = "JOB"
    USER = "USER"
    PI_GROUP = "PI_GROUP"
