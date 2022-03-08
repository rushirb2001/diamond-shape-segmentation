"""
Validation utilities for input checking and error prevention.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional


def validate_image_path(image_path: str) -> Tuple[bool, str]:
    """
    Validate image file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(image_path):
        return False, f"File does not exist: {image_path}"
    
    if not os.path.isfile(image_path):
        return False, f"Path is not a file: {image_path}"
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    _, ext = os.path.splitext(image_path)
    
    if ext.lower() not in valid_extensions:
        return False, f"Unsupported file format: {ext}"
    
    return True, "Valid"


def validate_image_array(image: np.ndarray,
                        min_size: int = 64,
                        max_size: int = 4096) -> Tuple[bool, str]:
    """
    Validate image numpy array.
    
    Args:
        image: Image array
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, np.ndarray):
        return False, f"Image must be numpy array, got {type(image)}"
    
    if len(image.shape) not in [2, 3]:
        return False, f"Image must be 2D or 3D array, got shape {image.shape}"
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False, f"Image must have 1, 3, or 4 channels, got {image.shape[2]}"
    
    height, width = image.shape[:2]
    
    if height < min_size or width < min_size:
        return False, f"Image too small: {width}x{height}, minimum is {min_size}x{min_size}"
    
    if height > max_size or width > max_size:
        return False, f"Image too large: {width}x{height}, maximum is {max_size}x{max_size}"
    
    return True, "Valid"


def validate_output_directory(output_dir: str, create: bool = True) -> Tuple[bool, str]:
    """
    Validate output directory.
    
    Args:
        output_dir: Output directory path
        create: Whether to create directory if it doesn't exist
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            return False, f"Path exists but is not a directory: {output_dir}"
        
        if not os.access(output_dir, os.W_OK):
            return False, f"Directory is not writable: {output_dir}"
        
        return True, "Valid"
    else:
        if create:
            try:
                os.makedirs(output_dir)
                return True, "Directory created"
            except Exception as e:
                return False, f"Failed to create directory: {str(e)}"
        else:
            return False, f"Directory does not exist: {output_dir}"


def validate_processing_parameters(iterations: int,
                                  clip_limit: float,
                                  tile_grid_size: Tuple[int, int]) -> Tuple[bool, str]:
    """
    Validate processing parameters.
    
    Args:
        iterations: GrabCut iterations
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if iterations < 1 or iterations > 20:
        return False, f"Iterations must be between 1 and 20, got {iterations}"
    
    if clip_limit < 0.1 or clip_limit > 10.0:
        return False, f"Clip limit must be between 0.1 and 10.0, got {clip_limit}"
    
    if len(tile_grid_size) != 2:
        return False, f"Tile grid size must be tuple of 2 values, got {len(tile_grid_size)}"
    
    if tile_grid_size[0] < 2 or tile_grid_size[1] < 2:
        return False, f"Tile grid dimensions must be at least 2, got {tile_grid_size}"
    
    return True, "Valid"