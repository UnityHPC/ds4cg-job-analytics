from .enum_constants import InteractiveEnum, QOSEnum, StatusEnum, ExitCodeEnum, PartitionEnum, AdminsAccountEnum

VRAM_VALUES = {
    "a100": 40,  # Default VRAM for a100 is 40GB, but we check usage to see which variant they want
    "a100-40g": 40,  # 40GB variant of a100 that can be specified explicitly in constraints
    "a100-80g": 80,  # 80GB variant of a100 that can be specified explicitly in constraints
    "v100": 16,
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

# Storing GPU names that have multiple memory sizes
MULTIVALENT_GPUS = {"a100": [40, 80], "v100": [16, 32]}


# Storing columns that are used for filtering in preprocess code
# This is used to check if all these columns exist in the dataframe for proper warning and handling
ESSENTIAL_COLUMNS = {
    "GPUType",
    "GPUs",
    "Status",
    "Elapsed",
    "Account",
    "Partition",
    "QOS",
    "Constraints",
    "StartTime",
    "SubmitTime",
    "NodeList",
    "GPUMemUsage",
}

# can add any columns below that we want to raise error anytime it is not there
ENFORCE_COLUMNS = {"GPUType", "Constraints", "StartTime", "SubmitTime", "NodeList", "GPUs", "GPUMemUsage", "Elapsed"}
