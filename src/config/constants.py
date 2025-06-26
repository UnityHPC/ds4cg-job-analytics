from .enum_constants import InteractiveEnum, QOSEnum, StatusEnum, ExitCodeEnum, PartitionEnum, AdminsAccountEnum

RAM_MAP = {
    "a100": 80,  # have 80GB and 40GB variants, variants will be added in another PR
    "v100": 16,  # have 32GB and 16GB variants, variants will be added in another PR
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
VRAM_CATEGORIES = [0, 8, 11, 12, 16, 23, 32, 40, 48, 80]
MIN_ELAPSED_SECONDS = 600  # Minimum elapsed time in seconds for analysis
