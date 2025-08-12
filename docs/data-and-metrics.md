# Data and Efficiency Metrics

This page provides comprehensive documentation about the data structure and efficiency metrics available in the DS4CG Unity Job Analytics project.

## Data Structure

The project works with job data from the Unity cluster's Slurm scheduler. After preprocessing, the data contains the following key attributes:

### Job Identification

- **JobID**: Unique identifier for each job
- **ArrayID**: Array job identifier (-1 for non-array jobs)
- **User**: Username of the job submitter
- **Account**: Account/group associated with the job

### Time Attributes

- **StartTime**: When the job started execution (datetime)
- **SubmitTime**: When the job was submitted (datetime)
- **Elapsed**: Total runtime duration (timedelta)
- **TimeLimit**: Maximum allowed runtime (timedelta)

### Resource Allocation

- **GPUs**: Number of GPUs allocated
- **GPUType**: Type of GPU allocated (e.g., "v100", "a100", or "cpu" for CPU-only jobs)
- **Nodes**: Number of nodes allocated
- **CPUs**: Number of CPU cores allocated
- **ReqMem**: Requested memory

### Job Status

- **Status**: Final job status (e.g., "COMPLETED", "FAILED", "CANCELLED")
- **ExitCode**: Job exit code
- **QOS**: Quality of Service level
- **Partition**: Cluster partition used

### Resource Usage

- **CPUTime**: Total CPU time used
- **CPUTimeRAW**: Raw CPU time measurement

### Constraints and Configuration

- **Constraints**: Hardware constraints specified
- **Interactive**: Whether the job was interactive ("interactive" or "non-interactive")

## Efficiency Metrics

The project provides several efficiency metrics to analyze resource utilization:

### VRAM Efficiency Metrics

#### Allocated VRAM Efficiency Score
```python
alloc_vram_efficiency_score = calculated based on allocated memory per GPU
```
Measures how effectively allocated VRAM was utilized based on the job's resource allocation.

#### VRAM Constraint Efficiency
```python
vram_constraint_efficiency = efficiency based on specified VRAM constraints
```
Measures utilization against specified VRAM constraints.

### Time-based Metrics

#### Job Hours
```python
job_hours = Elapsed.total_seconds() / 3600
```
Total runtime in hours.

#### VRAM Hours
```python
vram_hours = job_hours * allocated_vram_per_gpu
```
Total VRAM-hours consumed (job hours × allocated VRAM).

### CPU Efficiency Metrics

#### CPU Time Efficiency
```python
cpu_efficiency = (CPUTime / (Elapsed * CPUs)) * 100
```
Percentage of allocated CPU time actually used.

## Preprocessing Criteria

### Records Included

- Jobs with elapsed time above minimum threshold (default: 60 seconds)
- Jobs from non-root accounts
- Jobs not from building partition
- Jobs not using updates QOS

### Records Excluded (Optional)

- CPU-only jobs (can be included with `include_cpu_only_jobs=True`)
- Failed or cancelled jobs (can be included with `include_failed_cancelled_jobs=True`)

### Data Transformations

#### Null Value Handling

- **ArrayID**: Set to -1 for non-array jobs
- **Interactive**: Set to "non-interactive" if null
- **Constraints**: Set to empty array if null
- **GPUs**: Set to 0 for CPU jobs
- **GPUType**: Set to ["cpu"] for CPU jobs

#### Type Conversions

- **StartTime, SubmitTime**: Converted to datetime
- **TimeLimit, Elapsed**: Converted to timedelta
- **Categorical fields**: Status, ExitCode, QOS, Partition, Account

---

## Usage Examples

### Basic Efficiency Analysis

```python
from src.analysis.efficiency_analysis import EfficiencyAnalysis

# Initialize analyzer
analyzer = EfficiencyAnalysis(processed_df)

# Filter jobs
filtered_jobs = analyzer.filter_jobs_for_analysis(
    gpu_count_filter=1,
    vram_constraint_filter=None,
    allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
    gpu_mem_usage_filter={"min": 0, "max": np.inf, "inclusive": False}
)

# Calculate metrics
job_metrics = analyzer.calculate_job_efficiency_metrics(filtered_jobs=filtered_jobs)
```

### Time Series Analysis

```python
from src.analysis.frequency_analysis import FrequencyAnalysis
from src.config.enum_constants import TimeUnitEnum

# Prepare time series data
freq_analyzer = FrequencyAnalysis(job_metrics)
time_series_df = freq_analyzer.prepare_time_series_data(
    users=["user1", "user2"],
    metric="alloc_vram_efficiency_score",
    time_unit=TimeUnitEnum.MONTHS
)
```

### Memory Efficiency Categories
The system categorizes jobs into efficiency ranges:

- **Excellent**: > 80%
- **Good**: 60-80%
- **Fair**: 40-60%
- **Poor**: 20-40%
- **Very Poor**: < 20%

## Reference Documentation

For detailed API documentation of classes and methods:

- [Efficiency Analysis](analysis/efficiency_analysis.md)
- [Preprocessing](preprocess.md)
- [Visualization Models](visualization/models.md)
