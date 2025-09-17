# Unity GPU Efficiency Analytics Suite


This repository is a data analytics and reporting platform developed as part of the [Summer 2025 Data Science for the Common Good (DS4CG) program](https://ds.cs.umass.edu/programs/ds4cg/ds4cg-team-2025) in partnership with the Unity Research Computing Platform. It provides tools for analyzing HPC job data, generating interactive reports, and visualizing resource usage and efficiency.

## Motivation
High-performance GPUs are a critical resource on shared clusters, but they are often underutilized due to inefficient job scheduling, over-allocation, or lack of user awareness. Many jobs request more GPU memory or compute than they actually use, leading to wasted resources and longer queue times for others. This project aims to address these issues by providing analytics and reporting tools that help users and administrators understand GPU usage patterns, identify inefficiencies, and make data-driven decisions to improve overall cluster utilization.

## Project Overview
This project includes:
- Python scripts and modules for data preprocessing, analysis, and report generation
- Jupyter notebooks for interactive analysis and visualization
- Automated report generation scripts (see the `feature/reports` branch for the latest versions)
- Documentation built with MkDocs and Quarto

## Example Notebooks
The following notebooks generate comprehensive analysis for two subsets of the data:

- [`notebooks/analysis/No VRAM Use Analysis.ipynb`](notebooks/analysis/No%20VRAM%20Use%20Analysis.ipynb): Analysis of GPU jobs that end up using no VRAM.
- [`notebooks/analysis/Requested and Used VRAM.ipynb`](notebooks/analysis/Requested%20and%20Used%20VRAM.ipynb): Analysis of GPU jobs that request a specific amount of VRAM.

The following notebooks generate demonstrate key analyses, visualizations:

- [`notebooks/module_demos/Basic Visualization.ipynb`](notebooks/module_demos/Basic%20Visualization.ipynb): Basic plots and metrics
- [`notebooks/module_demos/Efficiency Analysis.ipynb`](notebooks/module_demos/Efficiency%20Analysis.ipynb): Calculation of efficiency metrics and user comparisons
- [`notebooks/module_demos/Resource Hoarding.ipynb`](notebooks/module_demos/Resource%20Hoarding.ipynb`): Analysis of CPU core and RAM overallocation 

The [`notebooks`](notebooks) directory contains all Jupyter notebooks.
  

## Documentation

This repository uses [MkDocs](https://www.mkdocs.org/) for project documentation. The documentation source files are located in the `docs/` directory and the configuration is in `mkdocs.yml`.

To build and serve the documentation locally:

    pip install -r dev-requirements.txt
    mkdocs serve

To build the static site:

    mkdocs build

To deploy the documentation (e.g., to GitHub Pages):

    mkdocs gh-deploy

See the [MkDocs documentation](https://www.mkdocs.org/user-guide/) for more details and advanced usage.

### Documenting New Features

For any new features, modules, or major changes, please add a corresponding `.md` file under the `docs/` directory. This helps keep the project documentation up to date and useful for all users and contributors.

## Dataset

The primary dataset for this project is a DuckDB database that contains information about jobs on
Unity. It is located under ```unity.rc.umass.edu:/modules/admin-resources/reporting/slurm_data.db``` and is updated daily.
The schema is provided below. In addition to the columns in the DuckDB file, this repository contains tools to add a number of useful derived columns for visualization and analysis.

| Column | Type | Description |
| :---    | :--- | :------------ |
| UUID   | VARCHAR | Unique identifier | 
| JobID  | INTEGER | Slurm job ID |
| ArrayID | INTEGER | Position in job array |
|ArrayJobID| INTEGER | Slurm job ID within array|
| JobName |  VARCHAR | Name of job |
| IsArray |  BOOLEAN | Indicator if job is part of an array |
| Interactive |  VARCHAR | Indicator if job was interactive
| Preempted |  BOOLEAN |  Was job preempted |
| Account |  VARCHAR |  Slurm account (PI group) |
| User |  VARCHAR |  Unity user |
| Constraints |  VARCHAR[] | Job constraints |
| QOS |  VARCHAR | Job QOS |
| Status |  VARCHAR | Job status on termination |
| ExitCode |  VARCHAR | Job exit code |
| SubmitTime |  TIMESTAMP_NS |  Job submission time |
| StartTime |  TIMESTAMP_NS | Job start time
| EndTime |  TIMESTAMP_NS | Job end time |
| Elapsed |  INTEGER | Job runtime (seconds) |
| TimeLimit |  INTEGER | Job time limit (seconds) |
| Partition |  VARCHAR | Job partition |
| Nodes |  VARCHAR | Job nodes as compact string |
| NodeList |  VARCHAR[] | List of job nodes |
| CPUs |  SMALLINT | Number of CPU cores |
| Memory |  INTEGER | Job allocated memory (bytes) |
| GPUs |  SMALLINT | Number of GPUs requested |
| GPUType |  DICT | Dictionary with keys as type of GPU (str) and the values as number of GPUs corresponding to that type (int) |
| GPUMemUsage |  FLOAT | GPU memory usage (bytes) |
| GPUComputeUsage |  FLOAT | GPU compute usage (pct) |
| CPUMemUsage |  FLOAT | GPU memory usage (bytes) |
| CPUComputeUsage |  FLOAT | CPU compute usage (pct) |


## Development Environment

To set up your development environment, use the provided [`dev-requirements.txt`](dev-requirements.txt) for all development dependencies (including linting, testing, and documentation tools).

This project requires **Python 3.11**. Make sure you have Python 3.11 installed before creating the virtual environment.

### Windows (PowerShell)

    python -m venv duckdb
    .\duckdb\Scripts\Activate.ps1
    pip install -r requirements.txt
    pip install -r dev-requirements.txt

### Linux / macOS (bash/zsh)

    python -m venv duckdb
    source duckdb/bin/activate
    pip install -r requirements.txt
    pip install -r dev-requirements.txt

If you need to reset your environment, you can delete the `duckdb` directory and recreate it as above.

### Version Control
To provide the path of the git configuration file of this project to git, run:

    git config --local include.path ../.gitconfig

To ensure consistent LF line endings across all platforms, run the following command when developing on Windows machines:

    git config --local core.autocrlf input

### Jupyter notebooks

You can run Jupyter notebooks on Unity through the OpenOnDemand portal. To make your environment 
visible in Jupyter, run 

    python -m ipykernel install --user --name "Duck DB"

from within the environment. This will add "Duck DB" as a kernel option in the dropdown.

By default, Jupyter Notebook outputs are removed via a git filter before the notebook is committed to git. To add an exception and keep the output of a notebook, add the file name of the notebook to [`scripts/strip_notebook_exclude.txt`](```scripts/.strip_notebook_exclude```).

## Code Style & Linting

### Ruff

This repository uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. All code must pass Ruff checks before being committed. To run Ruff:

    ruff check .
    ruff format .

If using VS Code, you can install the [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extension for an integrated experience. 

Ruff automatically checks your code when you open or edit Python or Jupyter Notebook files. You can use the extension to:

- View diagnostics: Linting issues are highlighted and can be viewed in the Problems panel (Ctrl+Shift+M).
- Fix issues: Use Quick Fix options for individual issues.
- Run the `Ruff: Fix all auto-fixable problems` command to address multiple violations. By default, unsafe fixes require explicit configuration in your Ruff file.
- Format code: Use the `Format Document` command or configure formatting on save.

### Mypy

This repository uses [Mypy](https://mypy-lang.org/) for static Typing in Python. All code should pass Mypy checks. To run it in terminal:

    mypy --config-file pyproject.toml .

If using VS Code, you can install the [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) extension for an integrated experience. To adjust the extension settings similar to:

    "mypy-type-checker.args": [
    "--config-file=pyproject.toml"
    ]
    "mypy-type-checker.cwd": "${workspaceFolder}/ds4cg-job-analytics"
    "mypy-type-checker.preferDaemon": true
    "mypy-type-checker.reportingScope": "workspace"

To manage Mypy with this extension, you can use the following commands:

- `Mypy: Recheck Workspace`: Re-run Mypy on the workspace.

- `Mypy: Restart Daemon and Recheck Workspace`: Restart the Mypy daemon and recheck.

### Docstrings

All Python code should use [**Google-style docstrings**](https://google.github.io/styleguide/pyguide.html#381-docstrings). Example template:

    def example_function(arg1: int, arg2: str) -> None:
        """
        Brief description of what the function does.

        Args:
            arg1 (int): Description of arg1.
            arg2 (str): Description of arg2.

        Returns:
            None
        """
        # ...function code...

## Testing

To run tests, use the provided test scripts or `pytest` (if available):

    pytest


## Support

The Unity documentation (https://docs.unity.rc.umass.edu/) has plenty of useful information about Unity and Slurm which would be helpful in understanding the data. For specific issues with the code in this repo or the DuckDB dataset, feel free to reach out to Benjamin Pachev on the Unity Slack.

