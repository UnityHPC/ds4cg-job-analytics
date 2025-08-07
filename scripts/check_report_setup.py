#!/usr/bin/env python
"""
Utility script to check if the GPU efficiency report environment is properly set up.

This script tests imports and dependencies to help troubleshoot any issues.
"""

import os
import sys
import importlib.util

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# List of required modules to check
REQUIRED_MODULES = [
    # Standard library
    "os", "sys", "subprocess", "importlib", "argparse",
    
    # External dependencies
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    
    # Project modules
    "src.analysis.efficiency_analysis", 
    "src.preprocess.preprocess", 
    "src.database.database_connection"
]


def check_module(module_name: str) -> tuple[bool, str]:
    """
    Check if a module can be imported.
    
    Args:
        module_name: The name of the module to check.
        
    Returns:
        A tuple containing (success_status, message).
    """
    try:
        if "." in module_name:
            # For packages with submodules
            parts = module_name.split(".")
            parent = importlib.import_module(".".join(parts[:-1]))
            getattr(parent, parts[-1])
        else:
            # For regular modules
            importlib.import_module(module_name)
        return True
    except (ImportError, AttributeError) as e:
        return False, str(e)


def check_quarto() -> tuple[bool, str]:
    """
    Check if Quarto is installed and available.
    
    Returns:
        A tuple containing (success_status, message).
    """
    try:
        import subprocess
        result = subprocess.run(["quarto", "--version"], 
                               capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except FileNotFoundError:
        return False, "Quarto command not found. Is it installed and in your PATH?"
    except Exception as e:
        return False, f"Error checking Quarto: {e}"


def test_quarto_rendering() -> tuple[bool, str]:
    """
    Test if Quarto can render a simple document.
    
    Returns:
        A tuple containing (success_status, message).
    """
    try:
        import subprocess
        import tempfile
        import os
        
        # Create a temporary Quarto document
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.qmd")
            with open(test_file, "w") as f:
                f.write("---\ntitle: Test\n---\n\nThis is a test.")
            
            # Try to render it
            result = subprocess.run(
                ["quarto", "render", test_file, "--to", "html"], 
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                return True, "Quarto rendering works correctly"
            return False, f"Quarto rendering failed: {result.stderr}"
    except Exception as e:
        return False, f"Error testing Quarto rendering: {e}"


def check_file_exists(file_path: str) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: The path to the file to check.
        
    Returns:
        True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


def main() -> None:
    """Run all checks and report results."""
    print("=== GPU Efficiency Report Environment Check ===")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check modules
    print("\nChecking required Python modules:")
    all_modules_ok = True
    for module in REQUIRED_MODULES:
        result = check_module(module)
        if result is True:
            print(f"  ✓ {module}")
        else:
            all_modules_ok = False
            print(f"  ✗ {module}: {result[1]}")
    
    # Check Quarto
    print("\nChecking Quarto installation:")
    quarto_result = check_quarto()
    if quarto_result[0]:
        print(f"  ✓ Quarto {quarto_result[1]}")
        
        # Test Quarto rendering if Quarto is installed
        print("\nTesting Quarto rendering:")
        render_result = test_quarto_rendering()
        if render_result[0]:
            print(f"  ✓ {render_result[1]}")
        else:
            print(f"  ✗ {render_result[1]}")
    else:
        print(f"  ✗ Quarto: {quarto_result[1]}")
    
    # Check template files
    print("\nChecking template files:")
    template_path = os.path.join(project_root, "reports", "user_report_template.qmd")
    css_path = os.path.join(project_root, "reports", "styles.css")
    
    if check_file_exists(template_path):
        print(f"  ✓ Template file: {template_path}")
    else:
        print(f"  ✗ Template file missing: {template_path}")
    
    if check_file_exists(css_path):
        print(f"  ✓ CSS file: {css_path}")
    else:
        print(f"  ✗ CSS file missing: {css_path}")
    
    # Check database files
    print("\nChecking database files:")
    db_files = [
        os.path.join(project_root, "slurm_data.db"),
        os.path.join(project_root, "slurm_data_small.db"),
        os.path.join(project_root, "slurm_data_updated.db"),
        os.path.join(project_root, "slurm_data_new.db")
    ]
    
    db_found = False
    for db_file in db_files:
        if check_file_exists(db_file):
            print(f"  ✓ Database file found: {db_file}")
            db_found = True
    
    if not db_found:
        print("  ✗ No database files found. Reports won't have data to work with.")
    
    # Check reports directory
    reports_dir = os.path.join(project_root, "reports", "user_reports")
    if not os.path.exists(reports_dir):
        try:
            os.makedirs(reports_dir, exist_ok=True)
            print(f"  ✓ Created reports directory: {reports_dir}")
        except Exception as e:
            print(f"  ✗ Could not create reports directory: {e}")
    else:
        print(f"  ✓ Reports directory exists: {reports_dir}")
    
    # Overall status
    print("\n=== Summary ===")
    if (all_modules_ok and quarto_result[0] and 
        check_file_exists(template_path) and 
        check_file_exists(css_path) and db_found):
        print("All checks passed! You're ready to generate reports.")
        print("\nTo generate reports, run:")
        print("  python scripts/generate_user_reports.py")
    else:
        print("Some checks failed. Please resolve the issues above before generating reports.")
        
        # Show common troubleshooting tips
        print("\n=== Troubleshooting Tips ===")
        print("1. If module imports fail:")
        print("   - Make sure you're running this script from the project root directory")
        print("   - Try installing missing modules with: pip install <module_name>")
        print("   - Check that src directory exists in the project root")
        
        print("\n2. If Quarto is not found:")
        print("   - Install Quarto from: https://quarto.org/docs/get-started/")
        print("   - Make sure Quarto is added to your PATH")
        
        print("\n3. If template files are missing:")
        print("   - Make sure the reports directory exists in the project root")
        print("   - Check that user_report_template.qmd and styles.css are in the reports directory")


if __name__ == "__main__":
    main()
