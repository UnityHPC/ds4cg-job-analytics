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


CATEGORY_PARTITION = [
    "building",
    "arm-gpu",
    "arm-preempt",
    "cpu",
    "cpu-preempt",
    "gpu",
    "gpu-preempt",
    "power9",
    "power9-gpu",
    "power9-gpu-preempt",
    "astroth-cpu",
    "astroth-gpu",
    "cbio-cpu",
    "cbio-gpu",
    "ceewater_casey-cpu",
    "ceewater_cjgleason-cpu",
    "ceewater_kandread-cpu",
    "ece-gpu",
    "fsi-lab",
    "gaoseismolab-cpu",
    "superpod-a100",
    "gpupod-l40s",
    "ials-gpu",
    "jdelhommelle",
    "lan",
    "mpi",
    "power9-gpu-osg",
    "toltec-cpu",
    "umd-cscdr-arm",
    "umd-cscdr-cpu",
    "umd-cscdr-gpu",
    "uri-cpu",
    "uri-gpu",
    "uri-richamp",
    "visterra",
    "zhoulin-cpu",
]

DEFAULT_MIN_ELAPSED_SECONDS = 600

# A map for categorical type construction, containing some values that exist in each type
ATTRIBUTE_CATEGORIES = {
    "Interactive": ["non-interactive", "shell"],
    "QOS": ["normal", "updates", "short", "long"],
    "Status": [
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "PREEMPTED",
        "OUT_OF_MEMORY",
        "PENDING",
        "NODE_FAIL",
    ],
    "ExitCode": ["SUCCESS", "ERROR", "SIGNALED"],
    "Account": ["root"],
    "Partition": CATEGORY_PARTITION,
}
