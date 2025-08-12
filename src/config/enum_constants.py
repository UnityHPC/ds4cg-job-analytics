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
class OptionalColumnsEnum(Enum):
    """
    An enumeration representing optional columns used for filtering in preprocess code.

    Attributes:
        STATUS: Job status column.
        ACCOUNT: Account column.
        PARTITION: Partition column.
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
    PARTITION = "Partition"
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
