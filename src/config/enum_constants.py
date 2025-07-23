"""
Declaration of some enum class such as constants values of categorical types.
"""

from enum import Enum, unique, auto


@unique
class InteractiveEnum(Enum):
    NON_INTERACTIVE = "non-interactive"
    SHELL = "shell"
    JUPYTER = "bc_jupyter"
    MATLAB = "bc_matlab"
    RSTUDIO = "bc_rstudio"
    DESKTOP = "bc_desktop"


@unique
class QOSEnum(Enum):
    NORMAL = "normal"
    UPDATES = "updates"
    SHORT = "short"
    LONG = "long"


@unique
class StatusEnum(Enum):
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
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    SIGNALED = "SIGNALED"


@unique
class AdminsAccountEnum(Enum):
    ROOT = "root"


@unique
class PartitionEnum(Enum):
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
class ProportionMetricsEnum(Enum):
    """
    Contains metrics for calculating proportion of data on ROC plot (y-axis).

    JOB_HOURS and JOBS elements are both used for y-axis ROC plot and as metrics in EfficiencyAnalysis class.
    """

    JOB_HOURS = "job_hours"
    VRAM_HOURS = "vram_hours"
    JOBS = "jobs"
    USER = "User"
    PI_GROUP = "Account"


@unique
class JobEfficiencyMetricsEnum(Enum):
    """
    Contains efficiency metrics for calculating efficiency of jobs.

    These are used as jobs metrics (in EfficiencyAnalysis class) and also metrics for x-axis in ROC Plot.
    """

    ALLOC_VRAM_EFFICIENCY = "alloc_vram_efficiency"
    VRAM_CONSTRAINT_EFFICIENCY = "vram_constraint_efficiency"
    VRAM_CONSTRAINT_EFFICIENCY_SCORE = "vram_constraint_efficiency_score"
    ALLOC_VRAM_EFFICIENCY_SCORE = "alloc_vram_efficiency_score"
    CPU_MEM_EFFICIENCY = "cpu_mem_efficiency"
    JOB_HOURS = "job_hours"
    VRAM_HOURS = "vram_hours"
