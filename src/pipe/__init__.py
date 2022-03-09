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
    'get_video_info'
]