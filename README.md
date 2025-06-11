## Introduction

This repository is a place to contain the tools developed over the course of the DS4CG 2025 summer
internship project with Unity.

## Contributing to this repository

The following guidelines may prove helpful in maximizing the utility of this repository:

- Please avoid committing code unless it is meant to be used by the rest of the team.
- New code should first be committed in a dedicated branch (```feature/newanalysis``` or ```bugfix/typo```), and later merged into ```main``` following a code
review.
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

To set up your development environment, use the provided `dev-requirements.txt` for all development dependencies (including linting, testing, and documentation tools):

    python -m venv duckdb && . duckdb/Scripts/activate  # On Windows PowerShell
    pip install -r requirements.txt
    pip install -r dev-requirements.txt

If you need to reset your environment, you can delete the `duckdb` folder and recreate it as above.

## Code Style & Linting

This repository uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. All code must pass Ruff checks before being committed. To run Ruff:

    ruff check .
    ruff format .

All Python code should use **Google-style docstrings**. Example template:

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

    python data/test.py
    # or
    pytest

## Visualization Utilities

The `src/analysis/visualize_columns.py` module provides a modular `DataVisualizer` class for column-wise data visualization from both DataFrame and DuckDB sources. See `data/test.py` for usage examples.

### User data and outreach

The ```zero_gpu_usage_list.py``` script generates a list of users who have repeatedly failed
to utilize requested GPUs in their jobs, and have never sucessfully used it. It generates personalized 
email bodies with user-specific resource usage. This script will only run on Unity, for users part
of the ```pi_bpachev_umass_edu``` group. It is included as an example of the sort of tool that 
might be useful to the Unity team as a final deliverable of this project.


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
| GPUType |  VARCHAR[] | List of GPU types |
| GPUMemUsage |  FLOAT | GPU memory usage (bytes) |
| GPUComputeUsage |  FLOAT | GPU compute usage (pct) |
| CPUMemUsage |  FLOAT | GPU memory usage (bytes) |
| CPUComputeUsage |  FLOAT | CPU compute usage (pct) |

