"""
Declaration of some enum class such as constants values of categorical types.
"""

from enum import Enum


class InteractiveEnum(Enum):
    NON_INTERACTIVE = "non-interactive"
    SHELL = "shell"
    JUPYTER = "bc_jupyter"
    MATLAB = "bc_matlab"
    RSTUDIO = "bc_rstudio"
    DESKTOP = "bc_desktop"


class QOSEnum(Enum):
    NORMAL = "normal"
    UPDATES = "updates"
    SHORT = "short"
    LONG = "long"


class StatusEnum(Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    PREEMPTED = "PREEMPTED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    PENDING = "PENDING"
    NODE_FAIL = "NODE_FAIL"


class ExitCodeEnum(Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    SIGNALED = "SIGNALED"


class AdminsAccountEnum(Enum):
    ROOT = "root"


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


class FilterTypeEnum(Enum):
    DICTIONARY = ("dictionary",)
    LIST = ("list",)
    SET = ("set",)
    TUPLE = ("tuple",)
    SCALAR = ("scalar",)
    PD_NA = ("pd_na",)
