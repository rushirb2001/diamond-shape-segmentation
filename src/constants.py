"""
Project-wide constants and configuration values.
"""

# Image specifications
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp']

# GrabCut parameters
GRABCUT_ITERATIONS = 4
GRABCUT_MODE_WITH_MASK = 1  # cv2.GC_INIT_WITH_MASK

# CLAHE parameters
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Video generation
VIDEO_FPS = 15
VIDEO_CODEC = 'DIVX'

# Default paths
DEFAULT_DATA_DIR = 'data/raw'
DEFAULT_OUTPUT_DIR = 'data/processed'

# Shape categories
SHAPE_COUNT = 14
IMAGES_PER_SHAPE = 256