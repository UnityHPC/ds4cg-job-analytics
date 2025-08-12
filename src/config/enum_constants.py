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
        JOBS: DataFrame name for jobs with resource hoarding metrics.
        USERS: DataFrame name for users with resource hoarding metrics.
        PI_GROUPS: DataFrame name for PI accounts/groups with resource hoarding metrics.
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
