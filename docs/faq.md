# Frequently Asked Questions (FAQ)

This page addresses common questions and technical issues encountered when using the DS4CG Unity Job Analytics project.

## Installation and Setup

### Q: When I try to install the requirements, I get dependency conflicts. How do I resolve this?

**A:** This usually happens due to conflicting package versions. Try these steps:

1. Create a fresh virtual environment:
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# or
fresh_env\Scripts\activate     # Windows
```

2. Update pip and install requirements:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

3. If conflicts persist, try installing packages individually:
```bash
pip install pandas plotly matplotlib duckdb pydantic
```

### Q: I'm getting a "Python version not supported" error. What Python version should I use?

**A:** The project requires Python 3.10 or higher. Check your Python version with:
```bash
python --version
```

If you have an older version, install Python 3.10+ from [python.org](https://python.org) or use a version manager like pyenv.

## Performance and Memory Issues

### Q: When I run my code on my computer, it crashes or runs very slowly.

**A:** This happens because the job data can be quite large and memory-intensive. Consider these solutions:

1. **Run on Unity cluster**: The data is designed to be processed on Unity where more computational resources are available.

2. **Use data sampling**: Limit the dataset size for testing:
```python
# Sample 10% of the data for testing
sample_df = full_df.sample(frac=0.1, random_state=42)
```

3. **Limit visualization points**:
```python
# Limit interactive plots to avoid memory issues
fig = visualizer.plot_vram_efficiency_per_job_dot_interactive(
    users=users,
    efficiency_metric="alloc_vram_efficiency",
    max_points=500  # Reduce from default 1000
)
```

4. **Process in chunks**:
```python
chunk_size = 5000
for chunk in pd.read_sql(query, connection, chunksize=chunk_size):
    process_chunk(chunk)
```

### Q: The interactive plots are not loading or are very slow. What can I do?

**A:** Interactive Plotly visualizations can be resource-intensive. Try these optimizations:

1. **Reduce data points**: Use the `max_points` parameter
2. **Exclude unnecessary fields**: Use `exclude_fields` to reduce hover text complexity
3. **Use static plots for large datasets**: Switch to matplotlib versions for better performance
4. **Filter users**: Analyze fewer users at once

## Database and Data Issues

### Q: I'm getting a "database not found" error. Where should the database file be located?

**A:** The Slurm database files are typically located on the Unity cluster. Common locations:
- `slurm_data.db`
- `slurm_data_new.db` 
- Check the `data/` directory in your project folder

If working locally, ensure you've copied the database file from Unity.

### Q: My analysis shows no data or empty results. What's wrong?

**A:** This usually happens due to filtering issues. Check these common causes:

1. **User filtering**: Ensure the users you're analyzing actually exist in the dataset:
```python
print("Available users:", df["User"].unique())
```

2. **Date range**: Check if your data covers the expected time period:
```python
print("Date range:", df["StartTime"].min(), "to", df["StartTime"].max())
```

3. **Preprocessing filters**: The preprocessing might be removing your data:
```python
# Check data before and after preprocessing
print("Before preprocessing:", len(raw_df))
processed_df = preprocess_jobs(raw_df, min_elapsed_seconds=60)
print("After preprocessing:", len(processed_df))
```

### Q: I'm getting "KeyError" when trying to access certain columns. What's happening?

**A:** This usually means the column doesn't exist in your dataset. Common issues:

1. **Case sensitivity**: Column names are case-sensitive (`"User"` vs `"user"`)
2. **Column not in dataset**: Check available columns:
```python
print("Available columns:", df.columns.tolist())
```
3. **Preprocessing changes**: Some columns might be renamed or removed during preprocessing

## Visualization Issues

### Q: The plots are not displaying or showing empty charts.

**A:** Several possible causes:

1. **Empty filtered data**: Check if your user/time filters are too restrictive
2. **Zero values**: Try setting `remove_zero_values=False`
3. **Jupyter notebook issues**: Ensure you have the right backend:
```python
%matplotlib inline
import plotly.io as pio
pio.renderers.default = "notebook"
```

### Q: The legend in my plots is cut off or overlapping.

**A:** Adjust the figure layout:
```python
# For matplotlib plots
plt.tight_layout()
plt.subplots_adjust(right=0.8)  # Make room for legend

# For Plotly plots
fig.update_layout(
    width=1200,  # Increase width
    margin=dict(r=200)  # Add right margin for legend
)
```

## Analysis and Metrics

### Q: What's the difference between "alloc_vram_efficiency" and "avail_vram_efficiency"?

**A:** 
- **alloc_vram_efficiency**: Measures efficiency against allocated memory per GPU
- **avail_vram_efficiency**: Measures efficiency against total available memory per GPU

Use allocated efficiency for analyzing how well users utilize their requested resources, and available efficiency for understanding overall cluster utilization.

### Q: Why are some efficiency values over 100%?

**A:** This can happen when:
1. Memory usage (`MaxRSS`) exceeds the baseline calculation
2. Shared memory or system overhead affects measurements
3. Multiple processes share GPU memory

Values slightly over 100% are normal; significantly higher values may indicate measurement issues.

### Q: How do I interpret the efficiency categories (Excellent, Good, Fair, etc.)?

**A:** The categories are defined as:
- **Excellent**: >80% - Very efficient resource usage
- **Good**: 60-80% - Acceptable efficiency 
- **Fair**: 40-60% - Room for improvement
- **Poor**: 20-40% - Significant waste of resources
- **Very Poor**: <20% - Major inefficiency

## Development and Contributing

### Q: How do I run the tests?

**A:** Run the test suite using pytest:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_efficiency_analysis.py

# Run with coverage
pytest --cov=src tests/
```

### Q: I want to add a new visualization. How do I structure the code?

**A:** Follow these guidelines:

1. Add visualization classes to `src/visualization/`
2. Inherit from `DataVisualizer` base class
3. Use Pydantic models for parameter validation
4. Add both static (matplotlib) and interactive (Plotly) versions when possible
5. Include comprehensive docstrings and type hints

### Q: How do I contribute documentation changes?

**A:** 
1. Edit the markdown files in the `docs/` directory
2. Test locally with: `mkdocs serve`
3. Submit a pull request with your changes

## Getting Help

### Q: I found a bug or want to request a feature. What should I do?

**A:** Please create a GitHub issue with:
1. Clear description of the problem/feature request
2. Steps to reproduce (for bugs)
3. Expected vs actual behavior
4. Your environment details (Python version, OS, etc.)

### Q: The documentation doesn't cover my use case. Where can I get help?

**A:** 
1. Check the [Demo](demo.md) page for examples
2. Look at the Jupyter notebooks in the `notebooks/` directory
3. Create a GitHub issue for documentation improvements
4. Reach out via Unity Slack for urgent questions
