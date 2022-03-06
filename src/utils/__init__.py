"""
Utility functions for image processing, file handling, and visualization.
"""

from src.utils.file_utils import (
    ensure_dir,
    get_file_extension,
    get_filename_without_extension,
    is_image_file,
    is_video_file,
    build_output_filename,
    list_files_by_extension,
    list_all_images,
    format_bytes,
    safe_filename,
    create_output_directory
)

from src.utils.segmentation import (
    init_grabcut_mask,
    visualize_mask,
    get_mask_stats,
    create_centered_mask,
    validate_mask,
    apply_clahe,
    rgb_to_lab,
    bgr_to_lab,
    lab_to_bgr,
    lab_to_rgb,
    enhance_diamond_image,
    preprocess_for_segmentation,
    adjust_brightness_contrast
)

__all__ = [
    # File utilities
    'ensure_dir',
    'get_file_extension',
    'get_filename_without_extension',
    'is_image_file',
    'is_video_file',
    'build_output_filename',
    'list_files_by_extension',
    'list_all_images',
    'format_bytes',
    'safe_filename',
    'create_output_directory',
    # Segmentation utilities
    'init_grabcut_mask',
    'visualize_mask',
    'get_mask_stats',
    'create_centered_mask',
    'validate_mask',
    # Image preprocessing
    'apply_clahe',
    'rgb_to_lab',
    'bgr_to_lab',
    'lab_to_bgr',
    'lab_to_rgb',
    'enhance_diamond_image',
    'preprocess_for_segmentation',
    'adjust_brightness_contrast'
]