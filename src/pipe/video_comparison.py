"""
Video comparison utilities for creating before/CLAHE/after demonstrations.
Generates triple-split videos showing the segmentation pipeline stages.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

from src.utils.segmentation import preprocess_for_segmentation, remove_background
from src.utils.visualization import add_text_overlay
from src.utils.file_utils import ensure_dir


class VideoComparator:
    """
    Creates comparison videos showing segmentation pipeline stages.
    """
    
    def __init__(self,
                 output_path: Union[str, Path],
                 fps: int = 15,
                 codec: str = 'DIVX'):
        """
        Initialize video comparator.
        
        Args:
            output_path: Path for output video
            fps: Frames per second
            codec: Video codec
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.writer = None
        
        # Ensure output directory exists
        ensure_dir(self.output_path.parent)
    
    def create_triple_split(self,
                           image_paths: List[str],
                           iterations: int = 5,
                           add_labels: bool = True,
                           label_color: Tuple[int, int, int] = (255, 255, 255),
                           label_bg_color: Tuple[int, int, int] = (0, 0, 0)) -> str:
        """
        Create triple-split video showing Before | CLAHE | After.
        
        Args:
            image_paths: List of paths to input images (256 frames)
            iterations: Number of GrabCut iterations
            add_labels: Whether to add text labels
            label_color: Text color for labels
            label_bg_color: Background color for label text
            
        Returns:
            Path to created video
        """
        print(f"Creating triple-split video with {len(image_paths)} frames...")
        
        # Get dimensions from first image
        first_img = cv2.imread(str(image_paths[0]))
        if first_img is None:
            raise ValueError(f"Failed to load first image: {image_paths[0]}")
        
        h, w = first_img.shape[:2]
        
        # Triple width for side-by-side
        frame_width = w * 3
        frame_height = h
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (frame_width, frame_height)
        )
        
        # Process each frame
        for img_path in tqdm(image_paths, desc="Processing frames"):
            # Load image
            original = cv2.imread(str(img_path))
            if original is None:
                print(f"Warning: Failed to load {img_path}, skipping...")
                continue
            
            # Preprocess
            _, enhanced = preprocess_for_segmentation(original)
            
            # Segment
            segmented, _ = remove_background(original, enhanced, iterations)
            
            # Add labels if requested
            if add_labels:
                original = add_text_overlay(
                    original, "Original", 
                    position=(10, 30),
                    color=label_color,
                    bg_color=label_bg_color
                )
                enhanced = add_text_overlay(
                    enhanced, "CLAHE Enhanced",
                    position=(10, 30),
                    color=label_color,
                    bg_color=label_bg_color
                )
                segmented = add_text_overlay(
                    segmented, "Segmented",
                    position=(10, 30),
                    color=label_color,
                    bg_color=label_bg_color
                )
            
            # Concatenate horizontally
            combined = np.hstack([original, enhanced, segmented])
            
            # Write frame
            self.writer.write(combined)
        
        # Release writer
        self.writer.release()
        
        print(f"Video saved: {self.output_path}")
        return str(self.output_path)
    
    def create_side_by_side(self,
                           image_paths: List[str],
                           iterations: int = 5,
                           mode: str = 'before-after',
                           add_labels: bool = True) -> str:
        """
        Create side-by-side comparison video.
        
        Args:
            image_paths: List of paths to input images
            iterations: Number of GrabCut iterations
            mode: Comparison mode ('before-after', 'clahe-after', 'before-clahe')
            add_labels: Whether to add text labels
            
        Returns:
            Path to created video
        """
        print(f"Creating {mode} comparison video...")
        
        # Get dimensions from first image
        first_img = cv2.imread(str(image_paths[0]))
        h, w = first_img.shape[:2]
        
        # Double width for side-by-side
        frame_width = w * 2
        frame_height = h
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (frame_width, frame_height)
        )
        
        # Process each frame
        for img_path in tqdm(image_paths, desc="Processing frames"):
            original = cv2.imread(str(img_path))
            if original is None:
                continue
            
            # Preprocess and segment
            _, enhanced = preprocess_for_segmentation(original)
            segmented, _ = remove_background(original, enhanced, iterations)
            
            # Select images based on mode
            if mode == 'before-after':
                left_img = original.copy()
                right_img = segmented
                left_label = "Original"
                right_label = "Segmented"
            elif mode == 'clahe-after':
                left_img = enhanced.copy()
                right_img = segmented
                left_label = "CLAHE"
                right_label = "Segmented"
            elif mode == 'before-clahe':
                left_img = original.copy()
                right_img = enhanced
                left_label = "Original"
                right_label = "CLAHE"
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Add labels
            if add_labels:
                left_img = add_text_overlay(left_img, left_label, position=(10, 30))
                right_img = add_text_overlay(right_img, right_label, position=(10, 30))
            
            # Concatenate
            combined = np.hstack([left_img, right_img])
            
            # Write frame
            self.writer.write(combined)
        
        # Release writer
        self.writer.release()
        
        print(f"Video saved: {self.output_path}")
        return str(self.output_path)


def create_triple_split_video(image_directory: Union[str, Path],
                              output_path: Union[str, Path],
                              shape_code: str,
                              fps: int = 15,
                              max_frames: Optional[int] = None) -> str:
    """
    Convenience function to create triple-split video from directory.
    
    Args:
        image_directory: Directory containing input images
        output_path: Path for output video
        shape_code: Shape code for filtering images
        fps: Frames per second
        max_frames: Maximum number of frames (None for all)
        
    Returns:
        Path to created video
    """
    import glob
    
    # Find images for this shape
    pattern = f"{shape_code}*.png"
    image_paths = sorted(glob.glob(str(Path(image_directory) / pattern)))
    
    if not image_paths:
        raise ValueError(f"No images found for shape {shape_code} in {image_directory}")
    
    # Limit frames if requested
    if max_frames:
        image_paths = image_paths[:max_frames]
    
    # Create comparator
    comparator = VideoComparator(output_path, fps)
    
    # Generate video
    return comparator.create_triple_split(image_paths)


def create_comparison_grid(frames_list: List[List[np.ndarray]],
                          labels: List[str],
                          output_path: Union[str, Path],
                          fps: int = 15,
                          grid_layout: Tuple[int, int] = None) -> str:
    """
    Create grid comparison video with multiple variations.
    
    Args:
        frames_list: List of frame sequences (each is a list of frames)
        labels: Labels for each sequence
        output_path: Path for output video
        fps: Frames per second
        grid_layout: Grid layout (rows, cols). Auto-calculated if None
        
    Returns:
        Path to created video
    """
    if not frames_list or not frames_list[0]:
        raise ValueError("frames_list cannot be empty")
    
    num_sequences = len(frames_list)
    num_frames = len(frames_list[0])
    
    # Auto-calculate grid layout if not provided
    if grid_layout is None:
        if num_sequences <= 3:
            grid_layout = (1, num_sequences)
        elif num_sequences <= 6:
            grid_layout = (2, 3)
        else:
            rows = int(np.ceil(np.sqrt(num_sequences)))
            cols = int(np.ceil(num_sequences / rows))
            grid_layout = (rows, cols)
    
    rows, cols = grid_layout
    
    # Get frame dimensions
    h, w = frames_list[0][0].shape[:2]
    
    # Calculate output dimensions
    frame_width = w * cols
    frame_height = h * rows
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    
    # Create each frame
    for frame_idx in tqdm(range(num_frames), desc="Creating grid video"):
        grid_rows = []
        
        for row in range(rows):
            row_frames = []
            for col in range(cols):
                seq_idx = row * cols + col
                
                if seq_idx < num_sequences:
                    frame = frames_list[seq_idx][frame_idx].copy()
                    
                    # Add label
                    frame = add_text_overlay(
                        frame,
                        labels[seq_idx],
                        position=(10, 30),
                        color=(255, 255, 255),
                        bg_color=(0, 0, 0)
                    )
                    row_frames.append(frame)
                else:
                    # Empty frame for unused grid cells
                    empty = np.zeros((h, w, 3), dtype=np.uint8)
                    row_frames.append(empty)
            
            # Concatenate row
            row_img = np.hstack(row_frames)
            grid_rows.append(row_img)
        
        # Concatenate all rows
        grid_frame = np.vstack(grid_rows)
        
        # Write frame
        writer.write(grid_frame)
    
    writer.release()
    
    print(f"Grid video saved: {output_path}")
    return str(output_path)


def add_progress_bar(frame: np.ndarray,
                    current: int,
                    total: int,
                    position: str = 'bottom',
                    bar_height: int = 20,
                    color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Add progress bar to frame.
    
    Args:
        frame: Input frame
        current: Current frame number
        total: Total number of frames
        position: 'top' or 'bottom'
        bar_height: Height of progress bar
        color: Color of progress bar
        
    Returns:
        Frame with progress bar
    """
    h, w = frame.shape[:2]
    result = frame.copy()
    
    # Calculate progress
    progress = int((current / total) * w)
    
    # Draw background bar
    if position == 'bottom':
        cv2.rectangle(result, (0, h - bar_height), (w, h), (50, 50, 50), -1)
        cv2.rectangle(result, (0, h - bar_height), (progress, h), color, -1)
    else:  # top
        cv2.rectangle(result, (0, 0), (w, bar_height), (50, 50, 50), -1)
        cv2.rectangle(result, (0, 0), (progress, bar_height), color, -1)
    
    return result


def add_frame_counter(frame: np.ndarray,
                     current: int,
                     total: int,
                     position: Tuple[int, int] = None) -> np.ndarray:
    """
    Add frame counter to frame.
    
    Args:
        frame: Input frame
        current: Current frame number
        total: Total number of frames
        position: Position for counter (default: top-right)
        
    Returns:
        Frame with counter
    """
    h, w = frame.shape[:2]
    
    if position is None:
        position = (w - 150, 30)
    
    text = f"{current}/{total}"
    
    return add_text_overlay(
        frame,
        text,
        position=position,
        font_scale=0.6,
        color=(255, 255, 255),
        bg_color=(0, 0, 0)
    )

# ============================================================================
# Advanced Video Comparison Features - Overlays and Metrics
# ============================================================================


class OverlayComparator:
    """
    Creates videos with overlay comparison modes.
    """
    
    def __init__(self,
                 output_path: Union[str, Path],
                 fps: int = 15,
                 codec: str = 'DIVX'):
        """
        Initialize overlay comparator.
        
        Args:
            output_path: Path for output video
            fps: Frames per second
            codec: Video codec
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.writer = None
    
    def create_fade_transition(self,
                               image_paths: List[str],
                               iterations: int = 5,
                               transition_frames: int = 30) -> str:
        """
        Create video with fade transitions between stages.
        
        Args:
            image_paths: List of paths to input images
            iterations: Number of GrabCut iterations
            transition_frames: Number of frames for each transition
            
        Returns:
            Path to created video
        """
        print(f"Creating fade transition video...")
        
        # Get dimensions
        first_img = cv2.imread(str(image_paths[0]))
        h, w = first_img.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (w, h)
        )
        
        for img_path in tqdm(image_paths, desc="Processing frames"):
            original = cv2.imread(str(img_path))
            if original is None:
                continue
            
            # Process
            _, enhanced = preprocess_for_segmentation(original)
            segmented, _ = remove_background(original, enhanced, iterations)
            
            # Original frame
            self.writer.write(original)
            
            # Fade to CLAHE
            for i in range(transition_frames):
                alpha = i / transition_frames
                blended = cv2.addWeighted(original, 1-alpha, enhanced, alpha, 0)
                blended = add_text_overlay(blended, f"Fade: {int(alpha*100)}%", 
                                          position=(10, 30))
                self.writer.write(blended)
            
            # CLAHE frame
            self.writer.write(enhanced)
            
            # Fade to segmented
            for i in range(transition_frames):
                alpha = i / transition_frames
                blended = cv2.addWeighted(enhanced, 1-alpha, segmented, alpha, 0)
                blended = add_text_overlay(blended, f"Fade: {int(alpha*100)}%",
                                          position=(10, 30))
                self.writer.write(blended)
            
            # Segmented frame
            self.writer.write(segmented)
        
        self.writer.release()
        print(f"Fade transition video saved: {self.output_path}")
        return str(self.output_path)
    
    def create_slider_comparison(self,
                                image_paths: List[str],
                                iterations: int = 5,
                                slider_position: float = 0.5) -> str:
        """
        Create video with vertical slider comparison.
        
        Args:
            image_paths: List of paths to input images
            iterations: Number of GrabCut iterations
            slider_position: Position of slider (0-1)
            
        Returns:
            Path to created video
        """
        print(f"Creating slider comparison video...")
        
        # Get dimensions
        first_img = cv2.imread(str(image_paths[0]))
        h, w = first_img.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (w, h)
        )
        
        slider_x = int(w * slider_position)
        
        for img_path in tqdm(image_paths, desc="Processing frames"):
            original = cv2.imread(str(img_path))
            if original is None:
                continue
            
            # Process
            _, enhanced = preprocess_for_segmentation(original)
            segmented, _ = remove_background(original, enhanced, iterations)
            
            # Create split view
            result = np.zeros_like(original)
            result[:, :slider_x] = original[:, :slider_x]
            result[:, slider_x:] = segmented[:, slider_x:]
            
            # Draw slider line
            cv2.line(result, (slider_x, 0), (slider_x, h), (0, 255, 0), 3)
            
            # Add labels
            result = add_text_overlay(result, "Original", position=(10, 30))
            result = add_text_overlay(result, "Segmented", 
                                     position=(slider_x + 10, 30))
            
            self.writer.write(result)
        
        self.writer.release()
        print(f"Slider comparison video saved: {self.output_path}")
        return str(self.output_path)


def calculate_segmentation_metrics(original: np.ndarray,
                                   segmented: np.ndarray,
                                   mask: np.ndarray) -> dict:
    """
    Calculate quality metrics for segmentation.
    
    Args:
        original: Original image
        segmented: Segmented image
        mask: Binary segmentation mask
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Foreground percentage
    total_pixels = mask.size
    foreground_pixels = np.sum(mask)
    metrics['foreground_percentage'] = (foreground_pixels / total_pixels) * 100
    
    # Mean intensity comparison
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    
    metrics['mean_intensity_original'] = np.mean(gray_original)
    metrics['mean_intensity_segmented'] = np.mean(gray_segmented[mask == 1])
    
    # Contrast (standard deviation)
    metrics['contrast_original'] = np.std(gray_original)
    metrics['contrast_segmented'] = np.std(gray_segmented[mask == 1])
    
    # Edge detection
    edges_original = cv2.Canny(gray_original, 100, 200)
    edges_segmented = cv2.Canny(gray_segmented, 100, 200)
    
    metrics['edge_density_original'] = np.sum(edges_original > 0) / total_pixels
    metrics['edge_density_segmented'] = np.sum(edges_segmented > 0) / foreground_pixels if foreground_pixels > 0 else 0
    
    return metrics


def add_metrics_overlay(frame: np.ndarray,
                       metrics: dict,
                       position: Tuple[int, int] = (10, 50)) -> np.ndarray:
    """
    Add metrics as text overlay on frame.
    
    Args:
        frame: Input frame
        metrics: Dictionary of metrics
        position: Starting position for text
        
    Returns:
        Frame with metrics overlay
    """
    result = frame.copy()
    x, y = position
    line_height = 25
    
    # Create semi-transparent background
    overlay = result.copy()
    cv2.rectangle(overlay, (x-5, y-20), (x+300, y+line_height*len(metrics)+10),
                 (0, 0, 0), -1)
    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    
    # Add metrics text
    for i, (key, value) in enumerate(metrics.items()):
        text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
        cv2.putText(result, text, (x, y + i*line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result


def create_metrics_comparison_video(image_paths: List[str],
                                   output_path: Union[str, Path],
                                   iterations: int = 5,
                                   fps: int = 15) -> str:
    """
    Create video with metrics overlay showing segmentation quality.
    
    Args:
        image_paths: List of paths to input images
        output_path: Path for output video
        iterations: Number of GrabCut iterations
        fps: Frames per second
        
    Returns:
        Path to created video
    """
    print(f"Creating metrics comparison video...")
    
    # Get dimensions
    first_img = cv2.imread(str(image_paths[0]))
    h, w = first_img.shape[:2]
    
    # Triple width for comparison
    frame_width = w * 3
    frame_height = h
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    
    for img_path in tqdm(image_paths, desc="Processing frames"):
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        
        # Process
        _, enhanced = preprocess_for_segmentation(original)
        segmented, mask = remove_background(original, enhanced, iterations)
        
        # Calculate metrics
        metrics = calculate_segmentation_metrics(original, segmented, mask)
        
        # Add labels
        original_labeled = add_text_overlay(original, "Original", position=(10, 30))
        enhanced_labeled = add_text_overlay(enhanced, "CLAHE Enhanced", position=(10, 30))
        segmented_labeled = add_text_overlay(segmented, "Segmented", position=(10, 30))
        
        # Add metrics to segmented frame
        display_metrics = {
            'FG%': metrics['foreground_percentage'],
            'Contrast': metrics['contrast_segmented']
        }
        segmented_labeled = add_metrics_overlay(segmented_labeled, display_metrics, 
                                               position=(10, 60))
        
        # Concatenate
        combined = np.hstack([original_labeled, enhanced_labeled, segmented_labeled])
        
        writer.write(combined)
    
    writer.release()
    print(f"Metrics comparison video saved: {output_path}")
    return str(output_path)


def create_quality_report(image_paths: List[str],
                         output_path: Union[str, Path],
                         iterations: int = 5) -> dict:
    """
    Generate quality report for segmentation results.
    
    Args:
        image_paths: List of paths to input images
        output_path: Path to save report JSON
        iterations: Number of GrabCut iterations
        
    Returns:
        Dictionary with aggregated metrics
    """
    print(f"Generating quality report for {len(image_paths)} images...")
    
    all_metrics = []
    
    for img_path in tqdm(image_paths, desc="Analyzing"):
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        
        # Process
        _, enhanced = preprocess_for_segmentation(original)
        segmented, mask = remove_background(original, enhanced, iterations)
        
        # Calculate metrics
        metrics = calculate_segmentation_metrics(original, segmented, mask)
        metrics['image_path'] = str(img_path)
        all_metrics.append(metrics)
    
    # Aggregate statistics
    import json
    
    report = {
        'total_images': len(all_metrics),
        'iterations': iterations,
        'metrics': all_metrics,
        'summary': {
            'avg_foreground_percentage': np.mean([m['foreground_percentage'] for m in all_metrics]),
            'avg_contrast_improvement': np.mean([
                m['contrast_segmented'] - m['contrast_original'] 
                for m in all_metrics
            ])
        }
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Quality report saved: {output_path}")
    return report


def create_annotated_comparison(image_paths: List[str],
                               output_path: Union[str, Path],
                               iterations: int = 5,
                               fps: int = 15,
                               show_progress: bool = True,
                               show_counter: bool = True) -> str:
    """
    Create fully annotated comparison video with progress bar and counter.
    
    Args:
        image_paths: List of paths to input images
        output_path: Path for output video
        iterations: Number of GrabCut iterations
        fps: Frames per second
        show_progress: Whether to show progress bar
        show_counter: Whether to show frame counter
        
    Returns:
        Path to created video
    """
    print(f"Creating annotated comparison video...")
    
    # Get dimensions
    first_img = cv2.imread(str(image_paths[0]))
    h, w = first_img.shape[:2]
    
    # Triple width
    frame_width = w * 3
    frame_height = h
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    
    total_frames = len(image_paths)
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Processing frames"), 1):
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        
        # Process
        _, enhanced = preprocess_for_segmentation(original)
        segmented, _ = remove_background(original, enhanced, iterations)
        
        # Add stage labels
        original = add_text_overlay(original, "Original", position=(10, 30))
        enhanced = add_text_overlay(enhanced, "CLAHE Enhanced", position=(10, 30))
        segmented = add_text_overlay(segmented, "Segmented", position=(10, 30))
        
        # Concatenate
        combined = np.hstack([original, enhanced, segmented])
        
        # Add progress bar
        if show_progress:
            combined = add_progress_bar(combined, idx, total_frames, 
                                       position='bottom', color=(0, 255, 0))
        
        # Add frame counter
        if show_counter:
            combined = add_frame_counter(combined, idx, total_frames)
        
        writer.write(combined)
    
    writer.release()
    print(f"Annotated comparison video saved: {output_path}")
    return str(output_path)

