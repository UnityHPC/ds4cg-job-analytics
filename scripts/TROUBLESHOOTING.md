# Troubleshooting Guide for GPU Efficiency Reports

This guide addresses common issues when running the GPU efficiency report generation scripts.

## ModuleNotFoundError: No module named 'src'

This error occurs because Python can't find the `src` package. There are several ways to resolve this:

### Option 1: Run from project root

Make sure you're running the script from the project root directory:

```bash
cd /path/to/ds4cg-job-analytics
python scripts/generate_user_reports.py --db-path ./slurm_data.db
```

### Option 2: Set PYTHONPATH environment variable

Set the PYTHONPATH to include your project directory:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "path\to\ds4cg-job-analytics"
python scripts\generate_user_reports.py --db-path .\slurm_data.db

# Linux/macOS
export PYTHONPATH=/path/to/ds4cg-job-analytics
python scripts/generate_user_reports.py --db-path ./slurm_data.db
```

### Option 3: Install the package in development mode

Install the package in development mode (this makes Python see your package regardless of where you run it from):

```bash
cd /path/to/ds4cg-job-analytics
pip install -e .
```

## Checking Your Environment

Use the included setup check script to verify all dependencies are correctly installed:

```bash
python scripts/check_report_setup.py
```

This script will check:
- Python modules
- Quarto installation
- Template files
- Database files
- Project directory structure

## Quarto Not Found

If you get an error about Quarto not being found:

1. Install Quarto from: https://quarto.org/docs/get-started/
2. Make sure Quarto is added to your PATH
3. Verify installation by running `quarto --version` in your terminal

## Parameters Not Found in Quarto Template

If you see errors related to parameters not found in the template:

```
NameError: name 'params' is not defined
```

This is fixed in the current version of the template. The template now properly accesses Quarto parameters using the special `_quarto` object. If you're still seeing this error:

1. Ensure you're using the latest version of the template
2. Try updating Quarto to the latest version
3. Check that parameters are correctly passed to the template

## No Data Shows in Reports

If reports generate successfully but show no data:

1. Check that the database file exists and contains the necessary data
2. Ensure that the analysis parameters (efficiency threshold, min jobs) aren't filtering out all users
3. Verify that the CSV files are being generated correctly in the temporary directory

## Quarto Errors

### Error: "output option cannot specify a relative or absolute path"

If you encounter this error:

```
ERROR: --output option cannot specify a relative or absolute path
```

This is a Quarto restriction that has been fixed in the latest version of the script. The script now uses just the filename for the output option and makes all paths absolute.

### Error: "No valid input files passed to render"

If you encounter this error:

```
ERROR: No valid input files passed to render
```

This means Quarto can't find the template file. The latest version of the script addresses this by:

1. Converting template paths to absolute paths
2. Checking that template files exist before trying to render
3. Printing detailed error information

Try these fixes:

1. Make sure the template file exists at the expected path
2. Use an absolute path for the template: `--template "C:\full\path\to\template.qmd"`
3. Run the script from the project root directory

## Command Line Structure Changed

Note that the command line structure has been updated to use subcommands:

- Old style (still works with defaults): `python scripts/generate_user_reports.py`
- New style for top users: `python scripts/generate_user_reports.py top --n-top 5`
- New style for specific users: `python scripts/generate_user_reports.py users --users user1,user2`

## File Format Issues

The system uses CSV files instead of Parquet to reduce dependencies. If you prefer using Parquet:

1. Install additional dependencies: `pip install pyarrow fastparquet`
2. Modify the file format in `generate_user_reports.py` from CSV to Parquet

## Performance Issues

If report generation is slow:

1. Use the smaller database for testing
2. Reduce the number of users with the `--n-top` parameter
3. Run reports in batches rather than all at once
- Quarto installation
- Template files

## Other Common Issues

### Quarto Not Found

Make sure Quarto is installed and available in your PATH. Install from: https://quarto.org/docs/get-started/

### Missing Templates

Make sure the template files exist:
- `reports/user_report_template.qmd`
- `reports/styles.css`

### Database Connection Issues

Ensure the database file path is correct and accessible.

## Quick Start

After resolving any issues, run:

```bash
# Run setup check
python scripts/check_report_setup.py

# Generate reports
python scripts/generate_user_reports.py --db-path ./slurm_data.db --output-dir ./reports/user_reports
```
