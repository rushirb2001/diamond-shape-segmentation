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


# ============================================================================
# Image Preprocessing Functions - CLAHE and Color Space Conversion
# ============================================================================


def apply_clahe(image: np.ndarray, 
                clip_limit: float = 3.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE improves local contrast and enhances the definition of edges
    in each region of an image. It's particularly useful for diamond images
    where lighting may be uneven.
    
    Args:
        image: Input BGR image
        clip_limit: Threshold for contrast limiting (default: 3.0)
        tile_grid_size: Size of grid for histogram equalization (default: 8x8)
        
    Returns:
        CLAHE-enhanced BGR image
    """
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split into L, A, B channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    
    # Merge channels back
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to LAB color space.
    
    LAB color space separates lightness from color information,
    making it useful for processing diamond images.
    
    Args:
        image: Input RGB image
        
    Returns:
        LAB color space image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to LAB color space.
    
    Args:
        image: Input BGR image
        
    Returns:
        LAB color space image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def lab_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert LAB image to BGR color space.
    
    Args:
        image: Input LAB image
        
    Returns:
        BGR color space image
    """
    return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)


def lab_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert LAB image to RGB color space.
    
    Args:
        image: Input LAB image
        
    Returns:
        RGB color space image
    """
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)


def enhance_diamond_image(image: np.ndarray,
                         clip_limit: float = 3.0,
                         tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance diamond image using CLAHE in LAB color space.
    
    This is the recommended preprocessing step before segmentation.
    
    Args:
        image: Input BGR image
        clip_limit: CLAHE clip limit (default: 3.0)
        tile_grid_size: CLAHE tile grid size (default: 8x8)
        
    Returns:
        Enhanced BGR image
    """
    return apply_clahe(image, clip_limit, tile_grid_size)


def preprocess_for_segmentation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image for segmentation.
    
    Returns both original and enhanced version for use in segmentation.
    
    Args:
        image: Input BGR image
        
    Returns:
        Tuple of (original_image, enhanced_image)
    """
    enhanced = enhance_diamond_image(image)
    return image.copy(), enhanced


def adjust_brightness_contrast(image: np.ndarray,
                               brightness: int = 0,
                               contrast: int = 0) -> np.ndarray:
    """
    Adjust brightness and contrast of image.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
        
    Returns:
        Adjusted image
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image