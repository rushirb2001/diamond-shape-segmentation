"""
Video generation utilities for creating videos from processed diamond images.
Compiles sequences of segmented images into video format.
"""

import cv2
import os
import glob
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm

from src.utils.file_utils import ensure_dir, list_files_by_extension, is_image_file


class VideoCreator:
    """
    Creates videos from sequences of images.
    """
    
    def __init__(self,
                 output_path: str,
                 fps: int = 15,
                 codec: str = 'DIVX',
                 frame_size: Optional[Tuple[int, int]] = None):
        """
        Initialize video creator.
        
        Args:
            output_path: Path for output video file
            fps: Frames per second (default: 15)
            codec: Video codec fourcc code (default: 'DIVX')
            frame_size: Optional fixed frame size (width, height)
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.frame_size = frame_size
        self.writer = None
        
        # Ensure output directory exists
        ensure_dir(os.path.dirname(output_path))
    
    def add_frame(self, frame: np.ndarray):
        """
        Add a frame to the video.
        
        Args:
            frame: Image frame to add
        """
        if self.writer is None:
            # Initialize writer with first frame
            if self.frame_size is None:
                height, width = frame.shape[:2]
                self.frame_size = (width, height)
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.frame_size
            )
        
        # Resize frame if needed
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        
        # Write frame
        self.writer.write(frame)
    
    def finalize(self):
        """Finalize and close the video file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


def create_video_from_images(image_paths: List[str],
                            output_path: str,
                            fps: int = 15,
                            codec: str = 'DIVX',
                            frame_size: Optional[Tuple[int, int]] = None,
                            sort: bool = True) -> str:
    """
    Create video from list of image paths.
    
    Args:
        image_paths: List of paths to images
        output_path: Path for output video
        fps: Frames per second
        codec: Video codec
        frame_size: Optional fixed frame size
        sort: Whether to sort image paths
        
    Returns:
        Path to created video file
    """
    if not image_paths:
        raise ValueError("No images provided")
    
    if sort:
        image_paths = sorted(image_paths)
    
    print(f"Creating video from {len(image_paths)} images...")
    
    with VideoCreator(output_path, fps, codec, frame_size) as creator:
        for img_path in tqdm(image_paths, desc="Processing frames"):
            frame = cv2.imread(img_path)
            if frame is not None:
                creator.add_frame(frame)
            else:
                print(f"\nWarning: Failed to load image: {img_path}")
    
    print(f"Video saved to: {output_path}")
    return output_path


def create_video_from_directory(input_dir: str,
                               output_path: str,
                               pattern: str = '*.png',
                               fps: int = 15,
                               codec: str = 'DIVX',
                               frame_size: Optional[Tuple[int, int]] = None) -> str:
    """
    Create video from all images in a directory.
    
    Args:
        input_dir: Directory containing images
        output_path: Path for output video
        pattern: Glob pattern for image files (default: '*.png')
        fps: Frames per second
        codec: Video codec
        frame_size: Optional fixed frame size
        
    Returns:
        Path to created video file
    """
    # Find all matching images
    search_pattern = os.path.join(input_dir, pattern)
    image_paths = glob.glob(search_pattern)
    
    if not image_paths:
        raise ValueError(f"No images found matching pattern: {search_pattern}")
    
    return create_video_from_images(image_paths, output_path, fps, codec, frame_size)


def create_shape_video(shape_code: str,
                      input_dir: str,
                      output_dir: str,
                      fps: int = 15) -> str:
    """
    Create video for a specific shape category.
    
    Args:
        shape_code: Shape code (e.g., 'AS', 'BR')
        input_dir: Directory containing processed images
        output_dir: Directory for output video
        fps: Frames per second
        
    Returns:
        Path to created video file
    """
    # Find all images for this shape
    pattern = f"{shape_code}_result_*.png"
    image_paths = glob.glob(os.path.join(input_dir, pattern))
    
    if not image_paths:
        raise ValueError(f"No images found for shape: {shape_code}")
    
    # Create output path
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"{shape_code}_segmentation.avi")
    
    return create_video_from_images(image_paths, output_path, fps)


def create_comparison_video(original_dir: str,
                           segmented_dir: str,
                           output_path: str,
                           fps: int = 15,
                           layout: str = 'horizontal') -> str:
    """
    Create side-by-side comparison video of original and segmented images.
    
    Args:
        original_dir: Directory with original images
        segmented_dir: Directory with segmented images
        output_path: Path for output video
        fps: Frames per second
        layout: 'horizontal' or 'vertical' layout
        
    Returns:
        Path to created video file
    """
    # Get matching image pairs
    original_images = sorted(glob.glob(os.path.join(original_dir, '*.png')))
    segmented_images = sorted(glob.glob(os.path.join(segmented_dir, '*.png')))
    
    if len(original_images) != len(segmented_images):
        raise ValueError("Number of original and segmented images don't match")
    
    print(f"Creating comparison video from {len(original_images)} image pairs...")
    
    creator = None
    
    try:
        for orig_path, seg_path in tqdm(zip(original_images, segmented_images), 
                                       total=len(original_images),
                                       desc="Processing frames"):
            # Load images
            orig = cv2.imread(orig_path)
            seg = cv2.imread(seg_path)
            
            if orig is None or seg is None:
                continue
            
            # Ensure same size
            if orig.shape != seg.shape:
                seg = cv2.resize(seg, (orig.shape[1], orig.shape[0]))
            
            # Combine images
            if layout == 'horizontal':
                combined = np.hstack([orig, seg])
            else:  # vertical
                combined = np.vstack([orig, seg])
            
            # Initialize creator on first frame
            if creator is None:
                height, width = combined.shape[:2]
                creator = VideoCreator(output_path, fps, frame_size=(width, height))
            
            creator.add_frame(combined)
        
        if creator is not None:
            creator.finalize()
        
        print(f"Comparison video saved to: {output_path}")
        return output_path
        
    finally:
        if creator is not None:
            creator.finalize()


def create_all_shapes_video(input_dir: str,
                           output_path: str,
                           fps: int = 15,
                           images_per_shape: int = 10) -> str:
    """
    Create video showcasing all shape categories.
    
    Args:
        input_dir: Directory containing processed images
        output_path: Path for output video
        fps: Frames per second
        images_per_shape: Number of images to include per shape
        
    Returns:
        Path to created video file
    """
    from src.data.loader import DiamondShapeMapper
    
    mapper = DiamondShapeMapper()
    all_shapes = mapper.list_all_shapes()
    
    selected_images = []
    
    # Collect images from each shape
    for shape_code in all_shapes:
        pattern = f"{shape_code}_result_*.png"
        shape_images = sorted(glob.glob(os.path.join(input_dir, pattern)))
        
        # Take first N images from this shape
        selected_images.extend(shape_images[:images_per_shape])
    
    if not selected_images:
        raise ValueError("No processed images found")
    
    return create_video_from_images(selected_images, output_path, fps)


def add_text_to_video(input_video_path: str,
                     output_video_path: str,
                     text: str,
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 1.0,
                     color: Tuple[int, int, int] = (255, 255, 255)) -> str:
    """
    Add text overlay to an existing video.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output video
        text: Text to overlay
        position: Text position (x, y)
        font_scale: Font size scale
        color: Text color (BGR)
        
    Returns:
        Path to output video
    """
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process each frame
    for _ in tqdm(range(total_frames), desc="Adding text overlay"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, 2, cv2.LINE_AA)
        
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_video_path


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'path': video_path,
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    
    return info