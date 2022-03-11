"""
Diamond Shape Segmentation - Computer Vision Pipeline
"""

__version__ = "1.0.0"
__author__ = "Rushir Bhavsar, Harshil Sanghvi, Ruju Shah, Vrunda Shah, Khushi Patel"
__email__ = "rushirbhavsar@gmail.com"

# Module exports
from src import data
from src import utils
from src import pipe
from src.config import Config, ProcessingConfig, VideoConfig, get_config

__all__ = ['data', 'utils', 'pipe', 'Config', 'ProcessingConfig', 'VideoConfig', 'get_config']