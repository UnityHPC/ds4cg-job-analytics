from .enum_constants import InteractiveEnum, QOSEnum, StatusEnum, ExitCodeEnum, PartitionEnum, AdminsAccountEnum
import re

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

# Storing GPU names that have multiple memory sizes
MULTIVALENT_GPUS = {"a100": [40, 80], "v100": [16, 32]}

# Map to calculate specific VRAM based on node name.
# This list is only for nodes that have multiple VRAM sizes for the same GPU type.
# based on https://docs.unity.rc.umass.edu/documentation/cluster_specs/nodes/
# TODO: read this from a config file or database
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
