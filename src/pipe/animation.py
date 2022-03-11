"""
Animation generation utilities for visualizing segmentation algorithm.
Creates animations showing mask evolution and processing steps.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

from src.utils.segmentation import (
    init_grabcut_mask,
    preprocess_for_segmentation,
    visualize_mask
)
from src.utils.visualization import add_text_overlay
from src.utils.file_utils import ensure_dir


class MaskEvolutionAnimator:
    """
    Creates animations showing how GrabCut mask evolves over iterations.
    """
    
    def __init__(self,
                 output_path: Union[str, Path],
                 fps: int = 15,
                 codec: str = 'DIVX'):
        """
        Initialize mask evolution animator.
        
        Args:
            output_path: Path for output video
            fps: Frames per second
            codec: Video codec
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        
        ensure_dir(self.output_path.parent)
    
    def animate_single_image_evolution(self,
                                       image_path: str,
                                       max_iterations: int = 10,
                                       frames_per_iteration: int = 10) -> str:
        """
        Create animation showing mask evolution for a single image.
        
        Args:
            image_path: Path to input image
            max_iterations: Maximum GrabCut iterations to show
            frames_per_iteration: Number of video frames per iteration
            
        Returns:
            Path to created video
        """
        print(f"Creating mask evolution animation...")
        
        # Load and preprocess image
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        _, enhanced = preprocess_for_segmentation(original)
        
        h, w = original.shape[:2]
        
        # Initialize video writer (triple width for side-by-side)
        frame_width = w * 3
        frame_height = h
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (frame_width, frame_height)
        )
        
        # Initialize mask
        mask = init_grabcut_mask(h, w)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Show initial state
        for _ in range(frames_per_iteration):
            initial_mask_viz = visualize_mask(mask)
            initial_result = original.copy()
            
            # Add labels
            original_labeled = add_text_overlay(original.copy(), "Original", position=(10, 30))
            mask_labeled = add_text_overlay(initial_mask_viz, "Initial Mask", position=(10, 30))
            result_labeled = add_text_overlay(initial_result, "Iteration 0", position=(10, 30))
            
            combined = np.hstack([original_labeled, mask_labeled, result_labeled])
            writer.write(combined)
        
        # Iterate and capture each step
        for iteration in range(1, max_iterations + 1):
            # Apply one GrabCut iteration
            cv2.grabCut(enhanced, mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
            
            # Create binary mask
            binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
            
            # Apply to original
            result = original * binary_mask[:, :, np.newaxis]
            
            # Visualize mask
            mask_viz = visualize_mask(mask)
            
            # Create frames for this iteration
            for _ in range(frames_per_iteration):
                # Add labels
                original_labeled = add_text_overlay(original.copy(), "Original", position=(10, 30))
                mask_labeled = add_text_overlay(mask_viz, f"Mask (Iter {iteration})", position=(10, 30))
                result_labeled = add_text_overlay(result, f"Result (Iter {iteration})", position=(10, 30))
                
                combined = np.hstack([original_labeled, mask_labeled, result_labeled])
                writer.write(combined)
        
        writer.release()
        print(f"Mask evolution animation saved: {self.output_path}")
        return str(self.output_path)
    
    def animate_multi_image_evolution(self,
                                     image_paths: List[str],
                                     iterations: int = 5) -> str:
        """
        Create animation showing mask evolution across multiple images.
        
        Args:
            image_paths: List of paths to input images
            iterations: Number of GrabCut iterations per image
            
        Returns:
            Path to created video
        """
        print(f"Creating multi-image mask evolution animation...")
        
        # Get dimensions from first image
        first_img = cv2.imread(str(image_paths[0]))
        h, w = first_img.shape[:2]
        
        # Initialize video writer
        frame_width = w * 2  # Original | Mask evolution
        frame_height = h
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (frame_width, frame_height)
        )
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            original = cv2.imread(str(img_path))
            if original is None:
                continue
            
            _, enhanced = preprocess_for_segmentation(original)
            
            # Initialize mask
            mask = init_grabcut_mask(h, w)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Show evolution
            for iteration in range(iterations + 1):
                if iteration > 0:
                    cv2.grabCut(enhanced, mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
                
                # Visualize
                mask_viz = visualize_mask(mask)
                
                # Add labels
                original_labeled = add_text_overlay(original.copy(), "Original", position=(10, 30))
                mask_labeled = add_text_overlay(mask_viz, f"Iteration {iteration}", position=(10, 30))
                
                combined = np.hstack([original_labeled, mask_labeled])
                writer.write(combined)
        
        writer.release()
        print(f"Multi-image mask evolution saved: {self.output_path}")
        return str(self.output_path)


class AlgorithmCoverageAnimator:
    """
    Creates animations showing algorithm coverage and processing steps.
    """
    
    def __init__(self,
                 output_path: Union[str, Path],
                 fps: int = 15,
                 codec: str = 'DIVX'):
        """
        Initialize algorithm coverage animator.
        
        Args:
            output_path: Path for output video
            fps: Frames per second
            codec: Video codec
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        
        ensure_dir(self.output_path.parent)
    
    def animate_pipeline_steps(self,
                               image_path: str,
                               iterations: int = 5,
                               frames_per_step: int = 15) -> str:
        """
        Create step-by-step animation of the entire pipeline.
        
        Args:
            image_path: Path to input image
            iterations: Number of GrabCut iterations
            frames_per_step: Frames to show each step
            
        Returns:
            Path to created video
        """
        print(f"Creating pipeline steps animation...")
        
        # Load image
        original = cv2.imread(str(image_path))
        h, w = original.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (w, h)
        )
        
        # Step 1: Original image
        for _ in range(frames_per_step):
            frame = add_text_overlay(original.copy(), 
                                     "Step 1: Load Original Image",
                                     position=(10, 30),
                                     font_scale=0.7,
                                     bg_color=(0, 0, 0))
            writer.write(frame)
        
        # Step 2: CLAHE preprocessing
        _, enhanced = preprocess_for_segmentation(original)
        for _ in range(frames_per_step):
            frame = add_text_overlay(enhanced.copy(),
                                     "Step 2: CLAHE Enhancement",
                                     position=(10, 30),
                                     font_scale=0.7,
                                     bg_color=(0, 0, 0))
            writer.write(frame)
        
        # Step 3: Initialize mask
        mask = init_grabcut_mask(h, w)
        mask_viz = visualize_mask(mask)
        for _ in range(frames_per_step):
            frame = add_text_overlay(mask_viz.copy(),
                                     "Step 3: Initialize GrabCut Mask",
                                     position=(10, 30),
                                     font_scale=0.7,
                                     bg_color=(0, 0, 0))
            writer.write(frame)
        
        # Step 4: GrabCut iterations
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        for iteration in range(1, iterations + 1):
            cv2.grabCut(enhanced, mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
            
            binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
            result = original * binary_mask[:, :, np.newaxis]
            
            for _ in range(frames_per_step):
                frame = add_text_overlay(result.copy(),
                                        f"Step 4: GrabCut Iteration {iteration}/{iterations}",
                                        position=(10, 30),
                                        font_scale=0.7,
                                        bg_color=(0, 0, 0))
                writer.write(frame)
        
        # Step 5: Final result
        for _ in range(frames_per_step * 2):  # Show final result longer
            frame = add_text_overlay(result.copy(),
                                     "Step 5: Final Segmented Result",
                                     position=(10, 30),
                                     font_scale=0.7,
                                     bg_color=(0, 0, 0))
            writer.write(frame)
        
        writer.release()
        print(f"Pipeline steps animation saved: {self.output_path}")
        return str(self.output_path)
    
    def animate_coverage_map(self,
                            image_paths: List[str],
                            grid_size: Tuple[int, int] = (4, 4)) -> str:
        """
        Create animation showing algorithm coverage across dataset.
        
        Args:
            image_paths: List of paths to input images
            grid_size: Grid dimensions (rows, cols)
            
        Returns:
            Path to created video
        """
        print(f"Creating coverage map animation...")
        
        rows, cols = grid_size
        num_cells = rows * cols
        
        # Get dimensions from first image
        first_img = cv2.imread(str(image_paths[0]))
        h, w = first_img.shape[:2]
        
        # Calculate grid dimensions
        cell_h = h // 2
        cell_w = w // 2
        
        frame_width = cell_w * cols
        frame_height = cell_h * rows
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (frame_width, frame_height)
        )
        
        # Process images in batches
        for batch_start in tqdm(range(0, len(image_paths), num_cells), 
                               desc="Processing batches"):
            batch_paths = image_paths[batch_start:batch_start + num_cells]
            
            # Create grid
            grid_rows = []
            for row in range(rows):
                row_cells = []
                for col in range(cols):
                    idx = row * cols + col
                    
                    if idx < len(batch_paths):
                        img = cv2.imread(str(batch_paths[idx]))
                        if img is not None:
                            # Process
                            _, enhanced = preprocess_for_segmentation(img)
                            
                            # Resize to cell size
                            cell = cv2.resize(enhanced, (cell_w, cell_h))
                            
                            # Add border
                            cell = cv2.copyMakeBorder(cell, 2, 2, 2, 2,
                                                     cv2.BORDER_CONSTANT,
                                                     value=(0, 255, 0))
                            row_cells.append(cell)
                        else:
                            # Empty cell
                            empty = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                            row_cells.append(empty)
                    else:
                        # Empty cell
                        empty = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                        row_cells.append(empty)
                
                row_img = np.hstack(row_cells)
                grid_rows.append(row_img)
            
            grid_frame = np.vstack(grid_rows)
            
            # Add title
            grid_frame = add_text_overlay(grid_frame,
                                         f"Processing: Images {batch_start}-{batch_start + num_cells}",
                                         position=(10, 30),
                                         font_scale=0.8,
                                         bg_color=(0, 0, 0))
            
            # Write multiple frames to slow down
            for _ in range(5):
                writer.write(grid_frame)
        
        writer.release()
        print(f"Coverage map animation saved: {self.output_path}")
        return str(self.output_path)


def create_evolution_gif(image_path: str,
                        output_path: Union[str, Path],
                        max_iterations: int = 10,
                        fps: int = 2) -> str:
    """
    Create GIF showing mask evolution.
    
    Args:
        image_path: Path to input image
        output_path: Path for output GIF
        max_iterations: Maximum iterations to show
        fps: Frames per second
        
    Returns:
        Path to created GIF
    """
    animator = MaskEvolutionAnimator(output_path, fps=fps)
    return animator.animate_single_image_evolution(image_path, max_iterations, frames_per_iteration=1)


def create_pipeline_animation(image_path: str,
                             output_path: Union[str, Path],
                             iterations: int = 5,
                             fps: int = 10) -> str:
    """
    Create animation showing complete pipeline.
    
    Args:
        image_path: Path to input image
        output_path: Path for output video
        iterations: Number of GrabCut iterations
        fps: Frames per second
        
    Returns:
        Path to created video
    """
    animator = AlgorithmCoverageAnimator(output_path, fps=fps)
    return animator.animate_pipeline_steps(image_path, iterations, frames_per_step=10)


def create_dataset_coverage_animation(image_paths: List[str],
                                     output_path: Union[str, Path],
                                     grid_size: Tuple[int, int] = (4, 4),
                                     fps: int = 15) -> str:
    """
    Create animation showing dataset coverage.
    
    Args:
        image_paths: List of paths to input images
        output_path: Path for output video
        grid_size: Grid dimensions
        fps: Frames per second
        
    Returns:
        Path to created video
    """
    animator = AlgorithmCoverageAnimator(output_path, fps=fps)
    return animator.animate_coverage_map(image_paths, grid_size)