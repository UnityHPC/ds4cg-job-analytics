from .enum_constants import InteractiveEnum, QOSEnum, StatusEnum, ExitCodeEnum, PartitionEnum, AdminsAccountEnum

VRAM_VALUES = {
    "a100": 80,  # By default, a100 is considered to be the 80GB variant
    "a100-40g": 40,  # 40GB variant of a100 that can be specified explicitly in constraints
    "a100-80g": 80,  # 80GB variant of a100 that can be specified explicitly in constraints
    "v100": 32, # By default, v100 is considered to be the 32GB variant
    "a40": 48,
    "gh200": 80,
    "rtx_8000": 48,
    "2080_ti": 11,
    "1080_ti": 11,
    "2080": 8,
    "h100": 80,
    "l4": 23,
    "m40": 23,
    "l40s": 48,
    "titan_x": 12,
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
    "Partition": PartitionEnum,
}

# Storing GPU names that have multiple vram options
# This is used to determine which GPU variant a job is using based on the VRAM usage
MULTIVALENT_GPUS = {"a100": [40, 80], "v100": [16, 32]}
