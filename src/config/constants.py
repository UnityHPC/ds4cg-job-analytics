from .enum_constants import InteractiveEnum, QOSEnum, StatusEnum, ExitCodeEnum, PartitionEnum, AdminsAccountEnum

RAM_MAP = {
    "a100": 40,  # have 80GB and 40GB variants, default is setted to 40GB
    "a100-40g": 40,
    "a100-80g": 80,
    "v100": 16,  # have 32GB and 16GB variants, default is setted to 16GB
    "a40": 48,
    "gh200": 95,
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

VARIABLE_GPUS = {"a100": [40, 80], "v100": [16, 32]}  # contains GPUs that have multiple memory variants

DEFAULT_MIN_ELAPSED_SECONDS = 600

# A map for categorical type construction, containing some values that exist in each type
ATTRIBUTE_CATEGORIES = {
    "Interactive": InteractiveEnum,
    "QOS": QOSEnum,
    "Status": StatusEnum,
    "ExitCode": ExitCodeEnum,
    "Account": AdminsAccountEnum,
    "Partition": PartitionEnum,
}
