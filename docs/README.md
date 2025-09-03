# DS4CG Job Analytics Documentation

This directory contains the documentation for the DS4CG Job Analytics project.

## Overview
The documentation provides detailed information about the data pipeline, analysis scripts, reporting tools, and usage instructions for the DS4CG Job Analytics platform. It is intended for users, contributors, and administrators who want to understand or extend the analytics and reporting capabilities.

## How to Build and View the Documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) and [Quarto](https://quarto.org/) for interactive reports and notebooks.

### MkDocs
- To serve the documentation locally:
  ```sh
  mkdocs serve
  ```
  This will start a local server (usually at http://127.0.0.1:8000/) where you can browse the docs.

- To build the static site:
  ```sh
  mkdocs build
  ```
  The output will be in the `site/` directory.

### Quarto
- Quarto is used for rendering interactive reports and notebooks (e.g., `.qmd` files).
- To render a Quarto report:
  ```sh
  quarto render path/to/report.qmd
  ```

## Structure
- `index.md`: Main landing page for the documentation site.
- `about.md`: Project background and team information.
- `preprocess.md`: Data preprocessing details.
- `analysis/`, `visualization/`, `mvp_scripts/`: Subsections for specific topics and scripts.
- `notebooks/`: Example notebooks and interactive analysis.

## Requirements
- Python 3.10+
- MkDocs (`pip install mkdocs`)
- Quarto (see https://quarto.org/docs/get-started/ for installation)

## Contributing
Contributions to the documentation are welcome! Edit or add Markdown files in this directory and submit a pull request.

---
For more details, see the main project README or contact the maintainers.
