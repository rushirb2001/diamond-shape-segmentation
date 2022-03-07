"""
Visualization utilities for diamond segmentation.
Handles contour detection, bounding boxes, and image display.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional


def find_contours(mask: np.ndarray, 
                 mode: int = cv2.RETR_EXTERNAL,
                 method: int = cv2.CHAIN_APPROX_NONE) -> List[np.ndarray]:
    """
    Find contours in binary mask.
    
    Args:
        mask: Binary mask (0 or 1)
        mode: Contour retrieval mode (default: RETR_EXTERNAL for outer contours only)
        method: Contour approximation method (default: CHAIN_APPROX_NONE for all points)
        
    Returns:
        List of contours as numpy arrays
    """
    # Ensure mask is uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask_uint8, mode, method)
    
    return contours


def get_largest_contour(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Get the largest contour by area.
    
    Args:
        contours: List of contours
        
    Returns:
        Largest contour or None if empty list
    """
    if len(contours) == 0:
        return None
    
    return max(contours, key=cv2.contourArea)


def draw_contours(image: np.ndarray,
                 contours: List[np.ndarray],
                 color: Tuple[int, int, int] = (255, 0, 0),
                 thickness: int = 3,
                 contour_idx: int = -1) -> np.ndarray:
    """
    Draw contours on image.
    
    Args:
        image: Input image (will be copied, not modified)
        contours: List of contours to draw
        color: BGR color for contours (default: blue)
        thickness: Line thickness (default: 3)
        contour_idx: Index of contour to draw, -1 for all (default: -1)
        
    Returns:
        Image with contours drawn
    """
    result = image.copy()
    cv2.drawContours(result, contours, contour_idx, color, thickness)
    return result


def draw_bounding_box(image: np.ndarray,
                     bbox: Tuple[int, int, int, int],
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2,
                     label: Optional[str] = None) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        image: Input image (will be copied, not modified)
        bbox: Bounding box as (x, y, width, height)
        color: BGR color for box (default: green)
        thickness: Line thickness (default: 2)
        label: Optional text label to display
        
    Returns:
        Image with bounding box drawn
    """
    result = image.copy()
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(result, (x, y - text_h - 10), (x + text_w, y), color, -1)
        
        # Draw text
        cv2.putText(result, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return result


def get_contour_bounding_box(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding box for a contour.
    
    Args:
        contour: Contour array
        
    Returns:
        Bounding box as (x, y, width, height)
    """
    return cv2.boundingRect(contour)


def annotate_segmentation(image: np.ndarray,
                         mask: np.ndarray,
                         draw_contours_flag: bool = True,
                         draw_bbox_flag: bool = True,
                         contour_color: Tuple[int, int, int] = (255, 0, 0),
                         bbox_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Annotate image with contours and bounding box.
    
    Args:
        image: Input image
        mask: Binary mask
        draw_contours_flag: Whether to draw contours
        draw_bbox_flag: Whether to draw bounding box
        contour_color: Color for contours (default: blue)
        bbox_color: Color for bounding box (default: green)
        
    Returns:
        Annotated image
    """
    result = image.copy()
    
    # Find contours
    contours = find_contours(mask)
    
    if len(contours) == 0:
        return result
    
    # Draw contours if requested
    if draw_contours_flag:
        result = draw_contours(result, contours, contour_color, 3)
    
    # Draw bounding box if requested
    if draw_bbox_flag:
        largest_contour = get_largest_contour(contours)
        if largest_contour is not None:
            bbox = get_contour_bounding_box(largest_contour)
            result = draw_bounding_box(result, bbox, bbox_color, 2)
    
    return result


def create_comparison_grid(images: List[np.ndarray],
                          titles: Optional[List[str]] = None,
                          rows: int = 1,
                          cols: int = 2,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Create a grid of images for comparison.
    
    Args:
        images: List of images to display
        titles: Optional list of titles for each image
        rows: Number of rows in grid
        cols: Number of columns in grid
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        # Convert BGR to RGB for matplotlib
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_display = img
        
        ax.imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
        ax.axis('off')
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
    
    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def create_before_after_comparison(original: np.ndarray,
                                  segmented: np.ndarray,
                                  figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Create before/after comparison visualization.
    
    Args:
        original: Original image
        segmented: Segmented image
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    return create_comparison_grid(
        [original, segmented],
        titles=['Original', 'Segmented'],
        rows=1,
        cols=2,
        figsize=figsize
    )


def create_segmentation_pipeline_viz(original: np.ndarray,
                                    enhanced: np.ndarray,
                                    mask: np.ndarray,
                                    segmented: np.ndarray,
                                    figsize: Tuple[int, int] = (16, 4)) -> plt.Figure:
    """
    Visualize complete segmentation pipeline.
    
    Args:
        original: Original image
        enhanced: Enhanced/preprocessed image
        mask: Binary segmentation mask
        segmented: Final segmented result
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert mask to 3-channel for display
    mask_viz = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    
    return create_comparison_grid(
        [original, enhanced, mask_viz, segmented],
        titles=['Original', 'Enhanced', 'Mask', 'Segmented'],
        rows=1,
        cols=4,
        figsize=figsize
    )


def save_visualization(fig: plt.Figure, output_path: str, dpi: int = 150) -> None:
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save image
        dpi: Resolution in dots per inch (default: 150)
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def display_image(image: np.ndarray, 
                 title: str = 'Image',
                 figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Display single image using matplotlib.
    
    Args:
        image: Image to display
        title: Window title
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_display = image
    
    ax.imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
    ax.axis('off')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def get_contour_properties(contour: np.ndarray) -> dict:
    """
    Calculate properties of a contour.
    
    Args:
        contour: Contour array
        
    Returns:
        Dictionary with contour properties
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Moments
    M = cv2.moments(contour)
    
    # Centroid
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    
    # Aspect ratio
    aspect_ratio = float(w) / h if h != 0 else 0
    
    # Extent (ratio of contour area to bounding box area)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area != 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'bounding_box': (x, y, w, h),
        'centroid': (cx, cy),
        'aspect_ratio': aspect_ratio,
        'extent': extent
    }


def add_text_overlay(image: np.ndarray,
                    text: str,
                    position: Tuple[int, int] = (10, 30),
                    font_scale: float = 0.7,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2,
                    bg_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Add text overlay to image.
    
    Args:
        image: Input image
        text: Text to display
        position: Text position (x, y)
        font_scale: Font size scale
        color: Text color in BGR
        thickness: Text thickness
        bg_color: Optional background color for text
        
    Returns:
        Image with text overlay
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background if specified
    if bg_color:
        x, y = position
        cv2.rectangle(result, 
                     (x - 5, y - text_h - 5), 
                     (x + text_w + 5, y + 5), 
                     bg_color, -1)
    
    # Draw text
    cv2.putText(result, text, position, font, font_scale, color, thickness)
    
    return result