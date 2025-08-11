from .enum_constants import InteractiveEnum, QOSEnum, StatusEnum, ExitCodeEnum, AdminPartitionEnum, AdminsAccountEnum

# VRAM values for different GPU types in GB. Add new GPU types in lowercase as needed.
VRAM_VALUES = {
    "a100": 80,  # By default, 'a100' in constraints is considered to be the 80GiB variant
    "a100-40g": 40,  # 40GB variant of a100 that can be specified explicitly in constraints
    "a100-80g": 80,  # 80GB variant of a100 that can be specified explicitly in constraints
    "v100": 32,  # By default, 'v100' in constraints is considered to be the 32GiB variant
    "a40": 48,
    "gh200": 80,
    "rtx_8000": 48,
    "rtx8000": 48,  # Alias for RTX 8000
    "2080_ti": 11,
    "2080ti": 11,  # Alias for 2080 Ti
    "1080_ti": 11,
    "1080ti": 11,  # Alias for 1080 Ti
    "2080": 8,
    "h100": 80,
    "l4": 23,
    "m40": 23,
    "l40s": 48,
    "titan_x": 12,
    "titanx": 12,  # Alias for Titan X
    "a16": 16,
    "cpu": 0,
}

VRAM_CATEGORIES = [0, 8, 11, 12, 16, 23, 32, 40, 48, 80]

DEFAULT_MIN_ELAPSED_SECONDS = 600  # 10 minutes, used for filtering jobs with short execution times

# A map for categorical type construction, containing some values that exist in each type
ATTRIBUTE_CATEGORIES = {
    "Interactive": InteractiveEnum,
    "QOS": QOSEnum,
    "Status": StatusEnum,
    "ExitCode": ExitCodeEnum,
    "Account": AdminsAccountEnum,
    "Partition": AdminPartitionEnum,
}

# Storing GPU names that have multiple vram options
# This is used to determine which GPU variant a job is using based on the VRAM usage
MULTIVALENT_GPUS = {"a100": [40, 80], "v100": [16, 32]}

# Mapping partitions to GPU types for specific constraints to calculate requested VRAM
# Update this map as new partitions are added or existing ones change
PARTITION_TO_GPU_MAP = {
    "superpod-a100": "a100-80g",
    "umd-cscdr-gpu": "a100-80g",
    "uri-gpu": "a100-80g",
    "cbio-gpu": "a100-80g",
    "power9-gpu": "v100",
    "power9-gpu-preempt": "v100",
    "ials-gpu": "2080_ti",
    "ece-gpu": "a100-40g",
    "lan": "a40",
    "astroth-gpu": "2080",
    "gpupod-l40s": "l40s",
}
