# Getting Started

This guide will help you set up and start using the DS4CG Unity Job Analytics project.

## Getting the Libraries

To get started with the project, clone the repository. Since the data is stored on the Unity cluster, we recommend cloning directly on Unity for best performance:

```bash
git clone https://github.com/Unity-HPC/ds4cg-job-analytics.git
cd ds4cg-job-analytics
```

## Dependencies

This project is compatible with **Python 3.10+**. We recommend first installing Python and then setting up a virtual environment for the project.

### Setting Up Virtual Environment

To set up a virtual environment, run the following commands:

```bash
# Create virtual environment
python -m venv duckdb

# Activate virtual environment
# On Linux/Mac:
source duckdb/bin/activate
# On Windows:
duckdb\Scripts\activate

# Install required libraries
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

### Required Libraries

The main dependencies include:

- pandas for data manipulation
- plotly and matplotlib for visualization
- duckdb for database operations
- pydantic for data validation
- mkdocs for documentation

## Data Retrieval and Preprocessing

The project provides streamlined functions to connect to the database and preprocess data:

### Database Connection

```python
from src.database.database_connection import DatabaseConnection

# Connect to the Slurm database
db = DatabaseConnection("path/to/slurm_data.db")

# Query GPU jobs
gpu_df = db.fetch_query("SELECT * FROM Jobs WHERE GPUs > 0")
```

### Data Preprocessing

The preprocessing pipeline handles data cleaning, type conversion, and filtering:

```python
from src.preprocess.preprocess import preprocess_data

# Preprocess raw job data
processed_df = preprocess_data(
    gpu_df,
    min_elapsed_seconds=600,
    include_failed_cancelled_jobs=False,
    include_cpu_only_jobs=True
)
```

For detailed preprocessing criteria, see the [Data and Efficiency Metrics](data-and-metrics.md) section.

## Getting Efficiency Metrics

The analysis workflow follows a specific order as demonstrated in our notebooks. Here's the complete process:

### Step 1: Initialize the Efficiency Analyzer

```python
from src.analysis.efficiency_analysis import EfficiencyAnalysis

# Initialize efficiency analyzer
efficiency_analyzer = EfficiencyAnalysis(jobs_df=processed_df)
```

### Step 2: Filter Jobs for Analysis

```python
import numpy as np

# Filter jobs based on specific criteria
filtered_jobs = efficiency_analyzer.filter_jobs_for_analysis(
    gpu_count_filter=1,
    vram_constraint_filter=None,
    allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
    gpu_mem_usage_filter={"min": 0, "max": np.inf, "inclusive": False}
)
```

### Step 3: Calculate Metrics

```python
# Calculate job-level efficiency metrics
job_metrics = efficiency_analyzer.calculate_job_efficiency_metrics(filtered_jobs=filtered_jobs)

# Calculate user-level efficiency metrics  
user_metrics = efficiency_analyzer.calculate_user_efficiency_metrics()

# Find inefficient users
inefficient_users = efficiency_analyzer.find_inefficient_users_by_alloc_vram_efficiency(
    alloc_vram_efficiency_filter={"min": 0, "max": 0.3, "inclusive": False}, 
    min_jobs=5
)
```

### Step 4: Prepare Time Series Data

```python
from src.analysis.frequency_analysis import FrequencyAnalysis

# Initialize frequency analyzer
frequency_analyzer = FrequencyAnalysis(job_metrics)

# Prepare time series data for visualization
time_series_data = frequency_analyzer.prepare_time_series_data(
    users=inefficient_users["User"].tolist(),
    time_unit="Months",
    metric="alloc_vram_efficiency_score",
    remove_zero_values=False
)
```

**📚 Complete Example**: See [Frequency Analysis Demo](../notebooks/Frequency Analysis/) for a full walkthrough.

For detailed information about available metrics, see [Efficiency Metrics](visualization/efficiency_metrics.md).

## Visualizing Job Analysis

The project offers both static and interactive visualization capabilities with a specific workflow:

### Step 1: Initialize Time Series Visualizer

```python
from src.visualization.time_series import TimeSeriesVisualizer

# Create time series visualizer with your time series data
visualizer = TimeSeriesVisualizer(time_series_data)
```

### Step 2: Static Time Series Plots

```python
# Static VRAM efficiency plot
visualizer.plot_vram_efficiency(
    users=["user1", "user2"],
    annotation_style="none",
    show_secondary_y=False
)

# Static VRAM hours plot
visualizer.plot_vram_hours(
    users=["user1", "user2"],
    show_secondary_y=False
)
```

### Step 3: Interactive Visualizations

```python
# Interactive VRAM efficiency plot
fig = visualizer.plot_vram_efficiency_interactive(
    users=["user1", "user2"],
    max_points=100,
    job_count_trace=True
)

# Interactive per-job dot plot
fig = visualizer.plot_vram_efficiency_per_job_dot_interactive(
    users=["user1", "user2"],
    efficiency_metric="alloc_vram_efficiency",
    vram_metric="job_hours",
    max_points=500,
    exclude_fields=["Exit Code"]
)
```

### Step 4: Per-Job Analysis

```python
# Initialize with job-level data for individual job analysis
job_visualizer = TimeSeriesVisualizer(job_metrics)

# Static per-job dot plot
job_visualizer.plot_vram_efficiency_per_job_dot(
    users=["user1"],
    efficiency_metric="alloc_vram_efficiency",
    vram_metric="job_hours"
)
```

### Column Statistics

```python
from src.visualization.columns import ColumnStatsVisualizer

# Visualize column statistics
col_visualizer = ColumnStatsVisualizer(processed_df)
col_visualizer.visualize_all_columns()
```

**📚 Complete Examples**:

- [Basic Visualization](../notebooks/Basic%20Visualization/) - Column statistics and basic plots
- [Efficiency Analysis](../notebooks/Efficiency%20Analysis/) - Advanced efficiency analysis workflows

For more visualization options, see [Visualization](visualization/visualization.md).

## Typical Analysis Workflow Order

Based on our notebooks, here's the recommended order for conducting analysis:

1. **Data Setup** → Load database → Preprocess data
2. **Initialize Analyzers** → EfficiencyAnalysis → FrequencyAnalysis  
3. **Filter & Calculate** → Filter jobs → Calculate metrics
4. **Identify Users** → Find inefficient/efficient users
5. **Prepare Visualizations** → Time series data → Initialize visualizers
6. **Generate Plots** → Static plots → Interactive plots → Per-job analysis

## Optional Scripts (MVP Scripts)

The project includes several standalone scripts for quick analysis:

- **CPU Metrics**: Analyze CPU usage patterns
- **GPU Metrics**: Analyze GPU utilization and efficiency
- **Zero GPU Usage**: Identify jobs with zero GPU usage

See [MVP Scripts](mvp_scripts/cpu_metrics.md) for detailed usage instructions.

---

**Next Steps:**

- Follow the complete workflows in our [Demo Notebooks](demo.md)
- Explore the [Data and Efficiency Metrics](data-and-metrics.md) page for detailed metric definitions
- Visit [FAQ](faq.md) if you encounter any issues