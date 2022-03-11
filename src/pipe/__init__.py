"""
Processing pipeline modules for batch diamond segmentation.
"""

from src.pipe.processor import (
    DiamondProcessor,
    BatchProcessor,
    AdvancedDiamondProcessor,
    ProcessingStats,
    InteractiveProcessor,
    process_diamond_dataset,
    run_interactive_mode
)

from src.pipe.video_creator import (
    VideoCreator,
    create_video_from_images,
    create_video_from_directory,
    create_shape_video,
    create_comparison_video,
    create_all_shapes_video,
    add_text_to_video,
    get_video_info
)

from src.pipe.video_comparison import (
    VideoComparator,
    OverlayComparator,
    create_triple_split_video,
    create_comparison_grid,
    add_progress_bar,
    add_frame_counter,
    calculate_segmentation_metrics,
    add_metrics_overlay,
    create_metrics_comparison_video,
    create_quality_report,
    create_annotated_comparison
)

__all__ = [
    # Processors
    'DiamondProcessor',
    'BatchProcessor',
    'AdvancedDiamondProcessor',
    'ProcessingStats',
    'InteractiveProcessor',
    'process_diamond_dataset',
    'run_interactive_mode',
    # Video creation
    'VideoCreator',
    'create_video_from_images',
    'create_video_from_directory',
    'create_shape_video',
    'create_comparison_video',
    'create_all_shapes_video',
    'add_text_to_video',
    'get_video_info',
    # Video comparison
    'VideoComparator',
    'OverlayComparator',
    'create_triple_split_video',
    'create_comparison_grid',
    'add_progress_bar',
    'add_frame_counter',
    'calculate_segmentation_metrics',
    'add_metrics_overlay',
    'create_metrics_comparison_video',
    'create_quality_report',
    'create_annotated_comparison'
]