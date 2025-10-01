"""
Path Utilities

This module provides cross-platform path handling utilities for the
transcribe pipeline, supporting both regular Python execution and
PyInstaller frozen applications.

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional


def resolve_path(path_input: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path, handling both relative and absolute paths.
    
    This function provides cross-platform path resolution that works
    in both regular Python execution and PyInstaller frozen applications.
    
    Args:
        path_input: Input path (string or Path object)
        base_dir: Base directory for relative paths (defaults to script directory)
        
    Returns:
        Resolved Path object
        
    Raises:
        ValueError: If the path cannot be resolved
    """
    if base_dir is None:
        base_dir = get_script_directory()
    
    # Convert to Path object if needed
    if isinstance(path_input, str):
        path_input = Path(path_input)
    
    # Handle absolute paths
    if path_input.is_absolute():
        return path_input.resolve()
    
    # Handle relative paths
    resolved_path = (base_dir / path_input).resolve()
    
    return resolved_path


def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object pointing to the ensured directory
        
    Raises:
        OSError: If the directory cannot be created
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
    
    return directory_path


def get_script_directory() -> Path:
    """
    Get the directory containing the main script.
    
    This function handles both regular Python execution and PyInstaller
    frozen applications correctly.
    
    Returns:
        Path object pointing to the script directory
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return Path(sys.executable).parent
    else:
        # Running as regular Python script
        return Path(__file__).resolve().parent.parent


def get_relative_path(target_path: Union[str, Path], 
                     reference_path: Union[str, Path] = None) -> Path:
    """
    Get a path relative to a reference path.
    
    Args:
        target_path: Target path to make relative
        reference_path: Reference path (defaults to script directory)
        
    Returns:
        Relative Path object
    """
    if reference_path is None:
        reference_path = get_script_directory()
    
    target_path = Path(target_path)
    reference_path = Path(reference_path)
    
    try:
        return target_path.relative_to(reference_path)
    except ValueError:
        # Path is not relative to reference, return as-is
        return target_path


def is_safe_path(base_path: Union[str, Path], 
                target_path: Union[str, Path]) -> bool:
    """
    Check if a target path is safe (within the base path).
    
    This prevents directory traversal attacks.
    
    Args:
        base_path: Base path to check against
        target_path: Target path to validate
        
    Returns:
        True if the path is safe, False otherwise
    """
    try:
        base_path = Path(base_path).resolve()
        target_path = Path(target_path).resolve()
        
        # Check if target is within base
        target_path.relative_to(base_path)
        return True
        
    except ValueError:
        return False


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        path: Path to the file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        OSError: If the file cannot be accessed
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise OSError(f"Path is not a file: {path}")
    
    return path.stat().st_size


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Get the total size of a directory in bytes.
    
    Args:
        directory: Path to the directory
        
    Returns:
        Total size in bytes
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        OSError: If the directory cannot be accessed
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise OSError(f"Path is not a directory: {directory}")
    
    total_size = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except OSError:
                # Skip files that can't be accessed
                continue
    
    return total_size


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing unsafe characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Define unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    
    # Replace unsafe characters with underscores
    sanitized = filename
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = 'unnamed_file'
    
    # Limit filename length (keep extension)
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        if ext:
            max_name_length = 255 - len(ext)
            sanitized = name[:max_name_length] + ext
        else:
            sanitized = sanitized[:255]
    
    return sanitized
