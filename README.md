## Introduction

This repository is a place to contain the tools developed over the course of the DS4CG 2025 summer
internship project with Unity.

## Contributing to this repository

The following guidelines may prove helpful in maximizing the utility of this repository:

- Please avoid committing code unless it is meant to be used by the rest of the team.
- New code should first be committed in a dedicated branch (```feature/newanalysis``` or ```bugfix/typo```), and later merged into ```main``` following a code review.
- Shared datasets should usually be managed with a shared folder on Unity, not committed to Git.
- Prefer committing Python modules with plotting routines like ```scripts/gpu_metrics.py``` instead of Jupyter notebooks, when possible. 
  
## Getting started on Unity

You'll need to first install a few dependencies, which include DuckDB, Pandas, and some plotting libraries. More details for running the project will need be added here later.

### Jupyter notebooks

You can run Jupyter notebooks on Unity through the OpenOnDemand portal. To make your environment 
visible in Jupyter, run 

    python -m ipykernel install --user --name "Duck DB"

from within the environment. This will add "Duck DB" as a kernel option in the dropdown.

## Development Environment

To set up your development environment, use the provided `dev-requirements.txt` for all development dependencies (including linting, testing, and documentation tools).

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

If you need to reset your environment, you can delete the `duckdb` folder and recreate it as above.

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

## Testing

To run tests, use the provided test scripts or `pytest` (if available):

    pytest


### Support

The Unity documentation (https://docs.unity.rc.umass.edu/) has a lot of useful
background information about Unity in particular and HPC in general. It will help explain a lot of
the terms used in the dataset schema below. For specific issues with the code in this repo or the
DuckDB dataset, feel free to reach out to Benjamin Pachev on the Unity Slack.

## The dataset

The primary dataset for this project is a DuckDB database that contains information about jobs on
Unity. It is contained under ```/modules/admin-resources/reporting/slurm_data.db``` and is updated daily.
A schema is provided below. In addition to the columns in the DuckDB file, ```scripts/gpu_metrics.py```
contains tools to add a number of useful derived columns for plotting and analysis.

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
| CPUs |  SMALLINT | Number of CPUs |
| Memory |  INTEGER | Job allocated memory (bytes) |
| GPUs |  SMALLINT | Number of GPUs requested |
| GPUType |  DICT | Dictionary of GPUTypes and number of GPUs as key-value pairs |
| GPUMemUsage |  FLOAT | GPU memory usage (bytes) |
| GPUComputeUsage |  FLOAT | GPU compute usage (pct) |
| CPUMemUsage |  FLOAT | GPU memory usage (bytes) |
| CPUComputeUsage |  FLOAT | CPU compute usage (pct) |

