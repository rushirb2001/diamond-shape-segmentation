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

from src.utils.output import (
    OutputManager,
    ResultCollector,
    create_output_structure,
    save_results_batch,
    generate_output_report
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
    adjust_brightness_contrast,
    remove_background,
    remove_background_batch,
    apply_mask_to_image,
    refine_mask_morphology,
    create_soft_mask,
    segment_with_postprocessing,
    get_foreground_bbox,
    crop_to_foreground,
    compare_segmentation_results
)

from src.utils.validators import (
    validate_image_path,
    validate_image_array,
    validate_output_directory,
    validate_processing_parameters
)

from src.utils.visualization import (
    find_contours,
    get_largest_contour,
    draw_contours,
    draw_bounding_box,
    get_contour_bounding_box,
    annotate_segmentation,
    create_comparison_grid,
    create_before_after_comparison,
    create_segmentation_pipeline_viz,
    save_visualization,
    display_image,
    get_contour_properties,
    add_text_overlay
)

from src.utils.profiling import (
    PerformanceProfiler,
    PerformanceMetrics,
    BatchProfiler,
    profile_function,
    measure_throughput,
    compare_performance,
    get_global_profiler,
    reset_global_profiler
)

from src.utils.logging_config import (
    setup_logging,
    get_logger,
    set_log_level,
    enable_debug_mode,
    disable_debug_mode,
    LogContext,
    log_function_call,
    create_session_log,
    log_system_info,
    log_processing_start,
    log_processing_end,
    log_error,
    clear_logs,
    initialize_default_logger
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
    # Output management
    'OutputManager',
    'ResultCollector',
    'create_output_structure',
    'save_results_batch',
    'generate_output_report'
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
    'adjust_brightness_contrast',
    # Background removal
    'remove_background',
    'remove_background_batch',
    'apply_mask_to_image',
    'refine_mask_morphology',
    'create_soft_mask',
    'segment_with_postprocessing',
    'get_foreground_bbox',
    'crop_to_foreground',
    'compare_segmentation_results',
    # Validators
    'validate_image_path',
    'validate_image_array',
    'validate_output_directory',
    'validate_processing_parameters'
    # Visualization
    'find_contours',
    'get_largest_contour',
    'draw_contours',
    'draw_bounding_box',
    'get_contour_bounding_box',
    'annotate_segmentation',
    'create_comparison_grid',
    'create_before_after_comparison',
    'create_segmentation_pipeline_viz',
    'save_visualization',
    'display_image',
    'get_contour_properties',
    'add_text_overlay'
    # Profiling
    'PerformanceProfiler',
    'PerformanceMetrics',
    'BatchProfiler',
    'profile_function',
    'measure_throughput',
    'compare_performance',
    'get_global_profiler',
    'reset_global_profiler'
    # Logging
    'setup_logging',
    'get_logger',
    'set_log_level',
    'enable_debug_mode',
    'disable_debug_mode',
    'LogContext',
    'log_function_call',
    'create_session_log',
    'log_system_info',
    'log_processing_start',
    'log_processing_end',
    'log_error',
    'clear_logs',
    'initialize_default_logger'
]