"""
File handling utilities for the diamond segmentation pipeline.
"""

import os
import glob
from typing import List, Optional, Tuple
from pathlib import Path


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Name of file
        
    Returns:
        File extension including dot (e.g., '.png')
    """
    return os.path.splitext(filename)[1]


def get_filename_without_extension(filename: str) -> str:
    """
    Get filename without extension.
    
    Args:
        filename: Name of file
        
    Returns:
        Filename without extension
    """
    return os.path.splitext(filename)[0]


def is_image_file(filename: str) -> bool:
    """
    Check if file is a supported image format.
    
    Args:
        filename: Name of file to check
        
    Returns:
        True if supported image format
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    return get_file_extension(filename).lower() in valid_extensions


def is_video_file(filename: str) -> bool:
    """
    Check if file is a supported video format.
    
    Args:
        filename: Name of file to check
        
    Returns:
        True if supported video format
    """
    valid_extensions = {'.avi', '.mp4', '.mov', '.mkv'}
    return get_file_extension(filename).lower() in valid_extensions


def build_output_filename(prefix: str, index: int, suffix: str = 'result', 
                          extension: str = '.png') -> str:
    """
    Build standardized output filename.
    
    Args:
        prefix: Shape code or category prefix
        index: Image index number
        suffix: Description suffix (default: 'result')
        extension: File extension (default: '.png')
        
    Returns:
        Formatted filename
    """
    return f"{prefix}_{suffix}_{index:04d}{extension}"


def list_files_by_extension(directory: str, extension: str) -> List[str]:
    """
    List all files with specific extension in directory.
    
    Args:
        directory: Directory to search
        extension: File extension to filter (e.g., '.png')
        
    Returns:
        List of matching filenames (sorted)
    """
    if not os.path.exists(directory):
        return []
    
    files = [f for f in os.listdir(directory) 
             if f.endswith(extension)]
    return sorted(files)


def list_all_images(directory: str) -> List[str]:
    """
    List all image files in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of image filenames (sorted)
    """
    if not os.path.exists(directory):
        return []
    
    files = [f for f in os.listdir(directory) if is_image_file(f)]
    return sorted(files)


def get_file_size(filepath: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0


def get_directory_size(directory: str) -> int:
    """
    Get total size of all files in directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += get_file_size(filepath)
    return total_size


def format_bytes(size: int) -> str:
    """
    Format byte size to human readable string.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def safe_filename(filename: str) -> str:
    """
    Convert string to safe filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    return safe_name


def find_files_recursive(directory: str, pattern: str = '*') -> List[str]:
    """
    Recursively find files matching pattern.
    
    Args:
        directory: Root directory to search
        pattern: Glob pattern (default: '*' for all files)
        
    Returns:
        List of full file paths
    """
    return glob.glob(os.path.join(directory, '**', pattern), recursive=True)


def split_path(filepath: str) -> Tuple[str, str, str]:
    """
    Split filepath into directory, filename, and extension.
    
    Args:
        filepath: Full file path
        
    Returns:
        Tuple of (directory, filename_without_ext, extension)
    """
    directory = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    filename, extension = os.path.splitext(basename)
    return directory, filename, extension


def create_output_directory(base_dir: str, shape_code: str) -> str:
    """
    Create output directory for specific shape.
    
    Args:
        base_dir: Base output directory
        shape_code: Shape code for subdirectory
        
    Returns:
        Full path to created directory
    """
    output_dir = os.path.join(base_dir, shape_code)
    ensure_dir(output_dir)
    return output_dir