# Demo

This page showcases the DS4CG Unity Job Analytics project in action with interactive examples and demonstrations.

## Complete Workflow Notebooks

Explore our comprehensive Jupyter notebooks that demonstrate the full capabilities:

### 📊 [Frequency Analysis Demo](notebooks/Frequency Analysis/)
**Complete end-to-end workflow** showing:

- Database connection and preprocessing
- Efficiency analysis setup and filtering
- Time series data preparation
- Interactive visualizations
- Best/worst user identification

### 📈 [Basic Visualization](notebooks/Basic%20Visualization/)
**Column statistics and exploratory analysis** including:

- Data loading and preprocessing
- Column-level statistical visualizations
- Distribution analysis
- Data quality assessment

### 🔍 [Efficiency Analysis](notebooks/Efficiency%20Analysis/)
**Advanced efficiency analysis techniques** covering:

- Job filtering and metrics calculation
- User and PI group analysis
- Inefficiency identification
- Performance comparison workflows

### 🎯 [Clustering Analysis](notebooks/clustering_analysis/)
**User behavior clustering and pattern analysis**

### 📊 [Frequency Analysis](notebooks/Frequency%20Analysis/)
**Time series frequency analysis and patterns**

---

## Quick Start Examples

For quick reference, here are the key workflow patterns:

### Database → Preprocessing → Analysis
```python
# See complete implementation in: VRAM Efficiency Analysis Demo notebook
db = DatabaseConnection("../slurm_data_new.db")
gpu_df = db.fetch_query("SELECT * FROM Jobs WHERE GPUs > 0")
processed_df = preprocess_data(gpu_df, min_elapsed_seconds=0)
```

### Efficiency Analysis Workflow
```python
# See complete implementation in: Efficiency Analysis notebook
efficiency_analyzer = EfficiencyAnalysis(jobs_df=processed_df)
filtered_jobs = efficiency_analyzer.filter_jobs_for_analysis(...)
job_metrics = efficiency_analyzer.calculate_job_efficiency_metrics(filtered_jobs)
```

### Interactive Visualizations
```python
# See complete implementation in: VRAM Efficiency Analysis Demo notebook
time_series_visualizer = TimeSeriesVisualizer(time_series_data)
fig = time_series_visualizer.plot_vram_efficiency_interactive(users=users_to_analyze)
```

---

## Notebook Features

### VRAM Efficiency Analysis Demo Features

- Complete efficiency analysis setup
- Time series data preparation and visualization
- Interactive plot generation
- Best/worst user identification
- Custom date range analysis

### Basic Visualization Features

- Database connection and data loading
- Column statistics generation
- Individual column visualizations

### Efficiency Analysis Features

- Job filtering and metrics calculation
- User efficiency analysis
- PI group analysis

---

## Performance Tips from Notebooks

Based on our notebook implementations:

### For Large Datasets
```python
# From VRAM Efficiency Analysis Demo notebook
filtered_jobs = efficiency_analyzer.filter_jobs_for_analysis(
    gpu_count_filter=1,
    allocated_vram_filter={"min": 0, "max": np.inf, "inclusive": False},
    gpu_mem_usage_filter={"min": 0, "max": np.inf, "inclusive": False}
)
```

### For Interactive Plots
```python
# From VRAM Efficiency Analysis Demo notebook
fig = time_series_visualizer.plot_vram_efficiency_per_job_dot_interactive(
    users=["user1", "user2"],
    efficiency_metric="alloc_vram_efficiency",
    max_points=500,  # Limit points for performance
    exclude_fields=["Exit Code"]
)
```

---

## Running the Notebooks

The notebooks are now integrated directly into this documentation! You can:

1. **View in Documentation**: Click on any notebook link above to view it rendered in the documentation
2. **Download and Run Locally**: 
   ```bash
   cd notebooks/
   jupyter lab
   ```
3. **Interactive Execution**: The notebooks contain complete, tested implementations with real data and interactive outputs

The integrated notebooks provide full access to working examples while keeping everything in one place!
