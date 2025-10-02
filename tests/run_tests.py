"""
Test runner script for the transcribe pipeline.

This script provides an easy way to run tests with different configurations.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(test_pattern=None, verbose=False, coverage=False):
    """
    Run tests with pytest.
    
    Args:
        test_pattern: Specific test pattern to run (e.g., 'test_config.py')
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src.transcribe_pipeline", "--cov-report=html", "--cov-report=term"])
    
    if test_pattern:
        cmd.append(f"tests/{test_pattern}")
    else:
        cmd.append(str(tests_dir))
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run transcribe pipeline tests")
    parser.add_argument("--pattern", help="Test pattern to run (e.g., test_config.py)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    args = parser.parse_args()
    
    exit_code = run_tests(
        test_pattern=args.pattern,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
