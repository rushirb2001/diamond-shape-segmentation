"""
Project-wide constants and configuration values.
"""

# Image specifications
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp']

# GrabCut parameters (adjusted after testing)
GRABCUT_ITERATIONS = 5  # Increased from 4 for better results
GRABCUT_MODE_WITH_MASK = 1  # cv2.GC_INIT_WITH_MASK

# CLAHE parameters (tuned for diamond images)
CLAHE_CLIP_LIMIT = 2.5  # Reduced from 3.0 to prevent over-enhancement
CLAHE_TILE_GRID_SIZE = (8, 8)

# Morphological refinement parameters
MORPH_KERNEL_SIZE = 3
MORPH_ITERATIONS = 2  # Increased from 1 for better mask refinement

# Video generation
VIDEO_FPS = 15
VIDEO_CODEC = 'DIVX'

# Default paths
DEFAULT_DATA_DIR = 'data/raw'
DEFAULT_OUTPUT_DIR = 'data/processed'

# Shape categories
SHAPE_COUNT = 14
IMAGES_PER_SHAPE = 256

# Processing
MAX_IMAGE_SIZE = 1024  # Maximum dimension for processing
MIN_IMAGE_SIZE = 64    # Minimum dimension for processing