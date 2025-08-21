# Analysis Module Documentation

This module contains advanced analysis tools for GPU job efficiency analysis, with a focus on the **ETC (Efficiency Threshold Curve)** visualization system.

## Overview

The analysis module provides sophisticated tools for understanding GPU resource utilization patterns through statistical analysis and data visualization. The primary contribution is the ETC plotting system, which enables researchers to analyze efficiency thresholds and their impact on resource consumption.

## ETC (Efficiency Threshold Curve) Feature

### What is ETC?

The **Efficiency Threshold Curve (ETC)** is a visualization technique originally implemented as "ROC Plot" but renamed for accuracy. ETC plots show the **cumulative resource consumption by efficiency threshold**, answering the critical question: *"What percentage of total resources is consumed by users/jobs below a given efficiency threshold?"*

**Key Insight**: ETC plots reveal how much GPU memory and compute time is wasted by inefficient users, enabling data-driven resource allocation policies.

### Core Functionality

The `ETCVisualizer` class (in `etc_plot.py`) provides three types of efficiency threshold analysis:

1. **Job-Level Analysis**: Analyze individual job efficiency vs. resource consumption
2. **User-Level Analysis**: Aggregate user behavior across all their jobs  
3. **PI Group Analysis**: Research group efficiency patterns

### Technical Implementation

#### Main Components

- **`ETCVisualizer` Class**: Core visualization engine inheriting from `EfficiencyAnalysis`
- **Plot Types**: `ETCPlotTypes.JOB`, `ETCPlotTypes.USER`, `ETCPlotTypes.PI_GROUP`
- **Metrics System**: Supports multiple efficiency and proportion metrics
- **Data Validation**: Robust filtering of invalid records (NaN, -inf values)

#### Key Methods

```python
# Primary plotting interface
plot_etc(
    plot_type: ETCPlotTypes,
    threshold_metric: EfficiencyMetricEnum,
    proportion_metric: ProportionMetricsEnum,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    threshold_step: float = 0.001
)

# Multi-line comparison plots
plot_multiple_etc_lines(
    plot_object_list: list[str],
    object_column_type: Literal["USERS", "PI_GROUPS"]
)
```

#### Supported Metrics

**Threshold Metrics (X-axis)**:
- `ALLOC_VRAM_EFFICIENCY`: GPU memory allocation efficiency 
- `VRAM_CONSTRAINT_EFFICIENCY`: Memory constraint efficiency  
- `ALLOC_VRAM_EFFICIENCY_SCORE`: Logarithmic allocated vram efficiency score
- `VRAM_CONSTRAINT_EFFICIENCY_SCORE`: Logarithmic vram constraint efficiency score
- And more (check [`enum_constants.py`](../config/enum_constants.py) at the EfficiencyMetrics enum for more information)
- User/PI aggregated versions of above metrics

**Proportion Metrics (Y-axis)**:
- `VRAM_HOURS`: Total GPU memory × time consumed
- `JOB_HOURS`: Total compute time consumed
- `JOBS`: Count of jobs
- `USERS`: Count of unique users
- `PI_GROUPS`: Count of research groups

## Limitations

### Current Implementation Issues

1. **Visualization & Calculation relies in the same class**
   Unlike other modules on the main branch, both analysis and visualization components are tightly coupled within the `ETCVisualizer` class,
   which reduces maintainability and violates separation of concerns principles. 

2. **Inconsistent Data Filtering in PI Group Aggregation**
   - **Issue**: When plotting PI group metrics with proportion metrics (`vram_hours`, `job_hours`, `jobs`, `users`) on the y-axis, the aggregation includes data from users whose threshold metrics contain invalid values (NULL/-inf)
   - **Impact**: This creates inconsistency where some users contribute to the PI group's proportion metric but not to the threshold metric calculation
   - **Example**: For a PI group plot with:
     - Y-axis: `vram_hours`
     - X-axis: `EXPECTED_VALUE_VRAM_CONSTRAINT_EFFICIENCY`
     
     Users with NULL `EXPECTED_VALUE_VRAM_CONSTRAINT_EFFICIENCY` are excluded from the PI's threshold calculation but their `vram_hours` are still included in the proportion calculation
   
   - **Current Workaround**: Partially addressed for User plots in lines 387-416 of [`_etc_calculate_proportion()`](etc_plot.py#L387-L416)
   - **Fix Needed**: Implement unified filtering logic that handles both User and PI Group plots consistently

## References

- Original implementation in `src/analysis/etc_plot.py`
- Configuration enums in `src/config/enum_constants.py`
- Example usage in `notebooks/ETC Plot.ipynb`
- Base efficiency analysis in `src/analysis/efficiency_analysis.py`
