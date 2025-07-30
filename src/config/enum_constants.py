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
    """An enumeration representing important admin accounts.

    Attributes:
        ROOT: Represents the root admin account.
    """

    ROOT = "root"


@unique
class PartitionEnum(Enum):
    """An enumeration representing different partitions on Unity."""

    BUILDING = "building"
    ARM_GPU = "arm-gpu"
    ARM_PREEMPT = "arm-preempt"
    CPU = "cpu"
    CPU_PREEMPT = "cpu-preempt"
    GPU = "gpu"
    GPU_PREEMPT = "gpu-preempt"
    POWER9 = "power9"
    POWER9_GPU = "power9-gpu"
    POWER9_GPU_PREEMPT = "power9-gpu-preempt"
    ASTROTH_CPU = "astroth-cpu"
    ASTROTH_GPU = "astroth-gpu"
    CBIO_CPU = "cbio-cpu"
    CBIO_GPU = "cbio-gpu"
    CEEWATER_CASEY_CPU = "ceewater_casey-cpu"
    CEEWATER_CJGLEASON_CPU = "ceewater_cjgleason-cpu"
    CEEWATER_KANDREAD_CPU = "ceewater_kandread-cpu"
    ECE_GPU = "ece-gpu"
    FSI_LAB = "fsi-lab"
    GAOSEISMOLAB_CPU = "gaoseismolab-cpu"
    SUPERPOD_A100 = "superpod-a100"
    GPUPOD_L40S = "gpupod-l40s"
    IALS_GPU = "ials-gpu"
    JDELHOMMELLE = "jdelhommelle"
    LAN = "lan"
    MPI = "mpi"
    POWER9_GPU_OSG = "power9-gpu-osg"
    TOLTEC_CPU = "toltec-cpu"
    UMD_CSCDR_ARM = "umd-cscdr-arm"
    UMD_CSCDR_CPU = "umd-cscdr-cpu"
    UMD_CSCDR_GPU = "umd-cscdr-gpu"
    URI_CPU = "uri-cpu"
    URI_GPU = "uri-gpu"
    URI_RICHAMP = "uri-richamp"
    VISTERRA = "visterra"
    ZHOULIN_CPU = "zhoulin-cpu"


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

    These are used as jobs metrics (in EfficiencyAnalysis class) and also metrics for x-axis in ROC Plot (for users).

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
