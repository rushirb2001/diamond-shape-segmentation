"""
Centralized configuration management for diamond segmentation pipeline.
Handles paths, parameters, and runtime settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class Config:
    """
    Configuration class for diamond segmentation pipeline.
    """
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # Dataset variants
    DATASET_VARIANTS = ['Shape_1d_256i', 'Shape_5d_256i', 'Shape_10d_256i']
    DEFAULT_VARIANT = 'Shape_1d_256i'
    
    # Image specifications
    IMAGE_SIZE = 256
    IMAGE_CHANNELS = 3
    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp']
    
    # GrabCut segmentation parameters
    GRABCUT_ITERATIONS = 5
    GRABCUT_MODE_MASK = 1  # cv2.GC_INIT_WITH_MASK
    GRABCUT_MODE_RECT = 0  # cv2.GC_INIT_WITH_RECT
    
    # CLAHE preprocessing parameters
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # Morphological operations
    MORPH_KERNEL_SIZE = 3
    MORPH_ITERATIONS = 2
    
    # Visualization parameters
    CONTOUR_COLOR = (255, 0, 0)  # Blue in BGR
    BBOX_COLOR = (0, 255, 0)     # Green in BGR
    CONTOUR_THICKNESS = 3
    BBOX_THICKNESS = 2
    
    # Video generation parameters
    VIDEO_FPS = 15
    VIDEO_CODEC = 'DIVX'
    VIDEO_FORMAT = '.avi'
    
    # Processing constraints
    MAX_IMAGE_SIZE = 1024
    MIN_IMAGE_SIZE = 64
    MAX_BATCH_SIZE = 100
    
    # Shape categories
    SHAPE_COUNT = 14
    IMAGES_PER_SHAPE = 256
    
    # Output structure
    OUTPUT_SUBDIRS = ['segmented', 'masks', 'annotated', 'stats', 'logs']
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to YAML config file to override defaults
        """
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: Path):
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
        """
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update class attributes with loaded config
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_file: Path):
        """
        Save current configuration to YAML file.
        
        Args:
            config_file: Path to save configuration
        """
        config_data = self.to_dict()
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary of configuration values
        """
        return {
            'dataset_variants': self.DATASET_VARIANTS,
            'default_variant': self.DEFAULT_VARIANT,
            'image_size': self.IMAGE_SIZE,
            'image_channels': self.IMAGE_CHANNELS,
            'grabcut_iterations': self.GRABCUT_ITERATIONS,
            'clahe_clip_limit': self.CLAHE_CLIP_LIMIT,
            'clahe_tile_grid_size': self.CLAHE_TILE_GRID_SIZE,
            'morph_kernel_size': self.MORPH_KERNEL_SIZE,
            'morph_iterations': self.MORPH_ITERATIONS,
            'video_fps': self.VIDEO_FPS,
            'video_codec': self.VIDEO_CODEC,
            'max_image_size': self.MAX_IMAGE_SIZE,
            'min_image_size': self.MIN_IMAGE_SIZE,
        }
    
    @classmethod
    def get_data_path(cls, variant: str = None) -> Path:
        """
        Get path to dataset variant.
        
        Args:
            variant: Dataset variant name (default: DEFAULT_VARIANT)
            
        Returns:
            Path to dataset directory
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return cls.RAW_DATA_DIR / variant
    
    @classmethod
    def get_output_path(cls, subdir: str = None) -> Path:
        """
        Get path to output directory.
        
        Args:
            subdir: Optional subdirectory name
            
        Returns:
            Path to output directory
        """
        if subdir:
            return cls.PROCESSED_DATA_DIR / subdir
        return cls.PROCESSED_DATA_DIR
    
    @classmethod
    def create_output_structure(cls) -> Dict[str, Path]:
        """
        Create standard output directory structure.
        
        Returns:
            Dictionary mapping output types to paths
        """
        structure = {}
        
        # Create main output directory
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        structure['base'] = cls.PROCESSED_DATA_DIR
        
        # Create subdirectories
        for subdir in cls.OUTPUT_SUBDIRS:
            path = cls.PROCESSED_DATA_DIR / subdir
            path.mkdir(parents=True, exist_ok=True)
            structure[subdir] = path
        
        return structure
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(variant={self.DEFAULT_VARIANT}, iterations={self.GRABCUT_ITERATIONS})"


class ProcessingConfig:
    """
    Configuration specific to processing operations.
    """
    
    def __init__(self,
                 variant: str = 'Shape_1d_256i',
                 iterations: int = 5,
                 add_annotations: bool = False,
                 save_masks: bool = False,
                 save_stats: bool = True):
        """
        Initialize processing configuration.
        
        Args:
            variant: Dataset variant to process
            iterations: Number of GrabCut iterations
            add_annotations: Whether to add contour annotations
            save_masks: Whether to save binary masks
            save_stats: Whether to save processing statistics
        """
        self.variant = variant
        self.iterations = iterations
        self.add_annotations = add_annotations
        self.save_masks = save_masks
        self.save_stats = save_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'variant': self.variant,
            'iterations': self.iterations,
            'add_annotations': self.add_annotations,
            'save_masks': self.save_masks,
            'save_stats': self.save_stats
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ProcessingConfig instance
        """
        return cls(**config_dict)


class VideoConfig:
    """
    Configuration for video generation.
    """
    
    def __init__(self,
                 fps: int = 15,
                 codec: str = 'DIVX',
                 frame_size: tuple = None,
                 layout: str = 'horizontal'):
        """
        Initialize video configuration.
        
        Args:
            fps: Frames per second
            codec: Video codec fourcc code
            frame_size: Optional fixed frame size (width, height)
            layout: Layout for comparison videos ('horizontal' or 'vertical')
        """
        self.fps = fps
        self.codec = codec
        self.frame_size = frame_size
        self.layout = layout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fps': self.fps,
            'codec': self.codec,
            'frame_size': self.frame_size,
            'layout': self.layout
        }


# Global default configuration instance
default_config = Config()


def get_config(config_file: Optional[Path] = None) -> Config:
    """
    Get configuration instance.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        Config instance
    """
    if config_file:
        return Config(config_file)
    return default_config


def load_processing_config(config_file: Path) -> ProcessingConfig:
    """
    Load processing configuration from file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        ProcessingConfig instance
    """
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return ProcessingConfig.from_dict(config_data.get('processing', {}))


def save_processing_config(config: ProcessingConfig, output_file: Path):
    """
    Save processing configuration to file.
    
    Args:
        config: ProcessingConfig instance
        output_file: Path to save configuration
    """
    config_data = {'processing': config.to_dict()}
    
    with open(output_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)


# Convenience functions for common configurations
def get_quick_processing_config() -> ProcessingConfig:
    """Get configuration for quick processing (minimal features)."""
    return ProcessingConfig(
        variant='Shape_1d_256i',
        iterations=4,
        add_annotations=False,
        save_masks=False,
        save_stats=False
    )


def get_full_processing_config() -> ProcessingConfig:
    """Get configuration for full processing (all features)."""
    return ProcessingConfig(
        variant='Shape_1d_256i',
        iterations=5,
        add_annotations=True,
        save_masks=True,
        save_stats=True
    )


def get_debug_processing_config() -> ProcessingConfig:
    """Get configuration for debugging (maximum output)."""
    return ProcessingConfig(
        variant='Shape_1d_256i',
        iterations=5,
        add_annotations=True,
        save_masks=True,
        save_stats=True
    )