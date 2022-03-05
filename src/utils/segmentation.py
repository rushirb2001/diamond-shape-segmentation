"""
Image segmentation utilities using GrabCut algorithm.
Handles mask initialization, preprocessing, and background removal.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def init_grabcut_mask(height: int, width: int) -> np.ndarray:
    """
    Initialize GrabCut mask with predefined regions.
    
    The mask defines probable foreground and background regions:
    - Center region: Definite foreground (diamond)
    - Middle ring: Probable foreground
    - Outer region: Probable background
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        
    Returns:
        Initialized mask array with GrabCut labels
        
    Note:
        Mask values:
        - cv2.GC_BGD (0): Definite background
        - cv2.GC_FGD (1): Definite foreground
        - cv2.GC_PR_BGD (2): Probable background
        - cv2.GC_PR_FGD (3): Probable foreground
    """
    # Initialize with probable background
    mask = np.ones((height, width), np.uint8) * cv2.GC_PR_BGD
    
    # Define probable foreground region (middle 60% of image)
    h_start, h_end = height // 5, 4 * height // 5
    w_start, w_end = width // 5, 4 * width // 5
    mask[h_start:h_end, w_start:w_end] = cv2.GC_PR_FGD
    
    # Define definite foreground region (center 20% of image)
    h_center_start, h_center_end = 2 * height // 5, 3 * height // 5
    w_center_start, w_center_end = 2 * width // 5, 3 * width // 5
    mask[h_center_start:h_center_end, w_center_start:w_center_end] = cv2.GC_FGD
    
    return mask


def visualize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert GrabCut mask to RGB image for visualization.
    
    Args:
        mask: GrabCut mask array
        
    Returns:
        RGB image representation of mask
        
    Color coding:
        - Black: Definite background
        - Dark gray: Probable background
        - Light gray: Probable foreground
        - White: Definite foreground
    """
    height, width = mask.shape
    vis_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Color mapping
    vis_mask[mask == cv2.GC_BGD] = [0, 0, 0]        # Black - definite background
    vis_mask[mask == cv2.GC_PR_BGD] = [64, 64, 64]  # Dark gray - probable background
    vis_mask[mask == cv2.GC_PR_FGD] = [192, 192, 192]  # Light gray - probable foreground
    vis_mask[mask == cv2.GC_FGD] = [255, 255, 255]  # White - definite foreground
    
    return vis_mask


def get_mask_stats(mask: np.ndarray) -> dict:
    """
    Calculate statistics about mask composition.
    
    Args:
        mask: GrabCut mask array
        
    Returns:
        Dictionary with pixel counts for each mask category
    """
    total_pixels = mask.size
    
    stats = {
        'definite_background': np.sum(mask == cv2.GC_BGD),
        'probable_background': np.sum(mask == cv2.GC_PR_BGD),
        'probable_foreground': np.sum(mask == cv2.GC_PR_FGD),
        'definite_foreground': np.sum(mask == cv2.GC_FGD),
        'total_pixels': total_pixels
    }
    
    # Add percentages
    for key in ['definite_background', 'probable_background', 
                'probable_foreground', 'definite_foreground']:
        stats[f'{key}_pct'] = (stats[key] / total_pixels) * 100
    
    return stats


def create_centered_mask(height: int, width: int, 
                        center_ratio: float = 0.6) -> np.ndarray:
    """
    Create a mask with centered foreground region.
    
    Args:
        height: Image height
        width: Image width
        center_ratio: Ratio of image to consider as foreground (default: 0.6)
        
    Returns:
        Initialized mask array
    """
    mask = np.ones((height, width), np.uint8) * cv2.GC_PR_BGD
    
    # Calculate centered region
    border_h = int(height * (1 - center_ratio) / 2)
    border_w = int(width * (1 - center_ratio) / 2)
    
    mask[border_h:height-border_h, border_w:width-border_w] = cv2.GC_PR_FGD
    
    # Create smaller definite foreground in center
    inner_ratio = center_ratio * 0.5
    inner_border_h = int(height * (1 - inner_ratio) / 2)
    inner_border_w = int(width * (1 - inner_ratio) / 2)
    
    mask[inner_border_h:height-inner_border_h, 
         inner_border_w:width-inner_border_w] = cv2.GC_FGD
    
    return mask


def validate_mask(mask: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that mask has correct format and values.
    
    Args:
        mask: Mask array to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if mask is None:
        return False, "Mask is None"
    
    if len(mask.shape) != 2:
        return False, f"Mask must be 2D, got shape {mask.shape}"
    
    if mask.dtype != np.uint8:
        return False, f"Mask must be uint8, got {mask.dtype}"
    
    unique_values = np.unique(mask)
    valid_values = {cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD}
    
    if not set(unique_values).issubset(valid_values):
        return False, f"Mask contains invalid values: {unique_values}"
    
    return True, "Valid mask"