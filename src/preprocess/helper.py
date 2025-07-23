from scipy.optimize import linprog

from ..config.constants import MULTIVALENT_GPUS


def estimate_vram_allocation(
    multivalent: dict[str, int],
    gpu_mem_usage: float | int,
    allocated_vram: int,
    multivalent_gpus: dict[str, list[int]] = MULTIVALENT_GPUS,
) -> dict[tuple[str, int], int] | None:
    """
    Estimate total VRAM allocation given GPU counts and multivalent variants.

    This ensures that the estimate exceeds both allocated_vram and gpu_mem_usage constraints.

    Args:
        multivalent (dict[str, int]): Dict of GPU types to their counts, e.g. {'A100': 3, 'V100': 2}.
        gpu_mem_usage (float | int): Actual observed GPU memory usage in GB.
        allocated_vram (int): Conservative VRAM estimate in GB (e.g., min VRAM * count).
        multivalent_gpus (dict[str, list[int]], Optional): Dict mapping GPU types to list of variant VRAM sizes, e.g.
                          {'A100': [40, 80], 'V100': [16, 32]}.

    Returns:
        Dict mapping (GPU_type, variant_size) to allocated count,
        or None if no valid allocation found.
    """
    print("Helper function is running", allocated_vram, gpu_mem_usage)

    variant_sizes = []
    variant_labels = []
    for gpu, _count in multivalent.items():
        for size in multivalent_gpus[gpu]:
            variant_sizes.append(size)
            variant_labels.append((gpu, size))

    # Objective: minimize total VRAM (for LP feasibility)
    c = [size for size in variant_sizes]

    # Equality constraints: sum of variants per GPU type == GPU count
    a_eq = []
    b_eq = []
    cursor = 0
    for gpu, count in multivalent.items():
        num_variants = len(multivalent_gpus[gpu])
        row = [0] * len(variant_sizes)
        for i in range(num_variants):
            row[cursor + i] = 1
        a_eq.append(row)
        b_eq.append(count)
        cursor += num_variants

    # Inequality constraint: total VRAM > gpu_mem_usage
    total_vram_needed = gpu_mem_usage
    a_ub = [[-size for size in variant_sizes]]
    b_ub = [-total_vram_needed]

    bounds = [(0, None)] * len(variant_sizes)

    res = linprog(
        c=c,
        A_eq=a_eq,
        b_eq=b_eq,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None

    # Round LP solution to integer counts
    allocation = {}
    for label, val in zip(variant_labels, res.x, strict=True):
        count = int(round(val))
        if count > 0:
            allocation[label] = count

    return allocation
