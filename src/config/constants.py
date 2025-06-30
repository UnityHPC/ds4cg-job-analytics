from .enum_constants import InteractiveEnum, QOSEnum, StatusEnum, ExitCodeEnum, PartitionEnum, AdminsAccountEnum
import re

RAM_MAP = {
    "a100": 80,
    "a100-40g": 40,
    "a100-80g": 80,
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

# Storing GPUs that have variable memory sizes
VARIABLE_GPUS = {"a100": [40, 80], "v100": [16, 32]}

# calculate specific VRAM based on node name
GET_VRAM_FROM_NODE = {
    "a100": lambda node: 40
    if node.startswith("ece-gpu")
    else 80
    if re.match("^(gpu0(1[3-9]|2[0-4]))|(gpu042)|(umd-cscdr-gpu00[1-2])|(uri-gpu00[1-8])$", node)
    else 0,
    "v100": lambda node: 16
    if re.match("^(gpu00[1-7])|(power9-gpu009)|(power9-gpu01[0-6])$", node)
    else 32
    if re.match("^(gpu01[1-2])|(power9-gpu00[1-8])$", node)
    else 0,
}
