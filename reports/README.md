# GPU Efficiency User Reports

This directory contains templates and scripts for generating HTML reports that analyze GPU usage patterns of users. These reports identify inefficient usage patterns and provide recommendations for improvement.

## Features

The generated reports include:
- User-specific GPU usage statistics
- VRAM efficiency metrics and comparisons to average usage
- Time usage efficiency analysis
- Visualizations of usage patterns over time
- Personalized recommendations for improvement

## Requirements

- Python 3.9+ with pandas, numpy, matplotlib, seaborn, and plotly
- Quarto document converter (https://quarto.org/docs/get-started/)

## Usage

1. Make sure Quarto is installed and available in your PATH

2. Check your environment setup:

```bash
python scripts/check_report_setup.py
```

3. Generate reports in one of two modes:

   a. Generate reports for top inefficient users:
   
   ```bash
   python scripts/generate_user_reports.py top --n-top 5 --db-path ./slurm_data.db
   ```
   
   b. Generate reports for specific users:
   
   ```bash
   python scripts/generate_user_reports.py users --users user1,user2,user3 --db-path ./slurm_data.db
   ```

4. Command-line options:

   **Top Mode Options:**
   ```
   top                 Generate reports for top inefficient users
     --n-top INT       Number of top inefficient users (default: 5)
     --efficiency FLOAT Maximum efficiency threshold (0-1) (default: 0.3)
     --min-jobs INT    Minimum number of jobs a user must have (default: 10)
   ```

   **Users Mode Options:**
   ```
   users               Generate reports for specific users
     --users STRING    Comma-separated list of user IDs (required)
     --min-jobs INT    Minimum number of jobs a user must have (default: 1)
   ```

   **Common Options:**
   ```
   --db-path PATH      Path to the database file (default: ./slurm_data.db)
   --output-dir PATH   Directory to save the reports (default: ./reports/user_reports)
   --template PATH     Path to the Quarto template file
   ```

## Customization

- The template file `reports/user_report_template.qmd` can be modified to change the content and styling of reports
- The CSS file `reports/styles.css` controls the visual appearance of the reports
- The report generation parameters can be adjusted to focus on different user segments

## Examples

To generate reports for the top 20 users with efficiency below 20% and at least 20 jobs:

```bash
python scripts/generate_user_reports.py top --n-top 20 --efficiency 0.2 --min-jobs 20
```

To generate reports for specific users regardless of their efficiency:

```bash
python scripts/generate_user_reports.py users --users john_doe,jane_smith,bob_johnson
```

To generate reports for specific users with a custom database file:

```bash
python scripts/generate_user_reports.py users --users john_doe,jane_smith --db-path ./slurm_data_new.db
```

To check if your environment is properly configured:

```bash
python scripts/check_report_setup.py
```

## Troubleshooting

If you encounter issues generating reports, try the following:

1. Run the setup check script to verify your environment:
   ```bash
   python scripts/check_report_setup.py
   ```

2. Common issues:
   - **Module not found errors**: Make sure you're running from the project root directory
   - **Quarto not found**: Ensure Quarto is installed and in your PATH
   - **Parameter errors in Quarto**: The Quarto template accesses parameters using special syntax
   - **Empty reports**: Check that the database contains the necessary data

3. The reports use CSV files instead of Parquet to avoid extra dependencies

## Report Interpretation

Each report provides several key metrics:

- **VRAM Efficiency**: The ratio of used VRAM to allocated VRAM
- **Efficiency Category**: Classification of usage patterns (Low, Moderate, Good, Excellent)
- **Time Usage**: How effectively job time limits were estimated
- **CPU/GPU Memory Ratio**: Balance between CPU and GPU memory usage

## Report Structure

Each report includes:

1. **Summary section**: Key metrics about the user's GPU usage
2. **Visualizations**: Charts showing usage patterns
3. **Recommendations**: Personalized suggestions for improvement based on usage patterns
4. **Comparison**: How the user compares to others on the system
python scripts/generate_user_reports.py --n-top 20 --efficiency 0.2 --min-jobs 20
```

The reports will be saved as HTML files in the specified output directory.
