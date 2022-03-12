"""
Main processing pipeline for diamond segmentation.
Handles batch processing of diamond images with progress tracking.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

from src.data.loader import DiamondDataLoader, DiamondShapeMapper
from src.utils.segmentation import (
    preprocess_for_segmentation,
    remove_background,
)
from src.utils.visualization import annotate_segmentation
from src.utils.file_utils import ensure_dir, build_output_filename


class DiamondProcessor:
    """
    Main processor for diamond image segmentation.
    Handles loading, processing, and saving of segmented images.
    """
    
    def __init__(self,
                data_loader: DiamondDataLoader,
                iterations: int = 4,
                add_annotations: bool = False):
        """
        Initialize diamond processor.
        
        Args:
            data_loader: Data loader instance
            output_dir: Directory to save processed images
            iterations: Number of GrabCut iterations
            add_annotations: Whether to add contour/bbox annotations
        """
        self.data_loader = data_loader
        self.output_dir = Path(__file__).parent.parent / 'data/processed'
        self.iterations = iterations
        self.add_annotations = add_annotations
        self.mapper = DiamondShapeMapper()
        
        # Ensure output directory exists
        ensure_dir(str(self.output_dir))
    
    def process_single_image(self,
                        image_path: str,
                        shape_code: str,
                        index: int,
                        save_result: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single diamond image.
        
        Args:
            image_path: Path to input image
            shape_code: Shape code for the diamond
            index: Image index for naming
            save_result: Whether to save the result to disk
            
        Returns:
            Tuple of (segmented_image, binary_mask)
        """
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Validate image dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"Invalid image dimensions: {image.shape}")
        
        # Preprocess
        original, enhanced = preprocess_for_segmentation(image)
        
        # Segment
        segmented, mask = remove_background(original, enhanced, self.iterations)
        
        # Add annotations if requested
        if self.add_annotations:
            segmented = annotate_segmentation(segmented, mask)
        
        # Save result if requested
        if save_result:
            output_filename = build_output_filename(shape_code, index, 'result', '.png')
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Ensure directory exists
            ensure_dir(os.path.dirname(output_path))
            
            # Save with error checking
            success = cv2.imwrite(output_path, segmented)
            if not success:
                raise IOError(f"Failed to save image: {output_path}")
        
        return segmented, mask
    
    def process_shape_category(self,
                              shape_id: int,
                              max_images: Optional[int] = None,
                              save_results: bool = True) -> Dict:
        """
        Process all images in a shape category.
        
        Args:
            shape_id: Numeric shape identifier (1-14)
            max_images: Maximum number of images to process (None for all)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with processing statistics
        """
        shape_code = self.mapper.get_shape_code(shape_id)
        if shape_code is None:
            raise ValueError(f"Invalid shape ID: {shape_id}")
        
        shape_name = self.mapper.get_shape_name(shape_code)
        
        # Get list of images
        image_list = self.data_loader.list_images(shape_id)
        
        if max_images is not None:
            image_list = image_list[:max_images]
        
        print(f"\nProcessing {len(image_list)} images for {shape_name} ({shape_code})...")
        
        # Process images with progress bar
        results = {
            'shape_code': shape_code,
            'shape_name': shape_name,
            'total_images': len(image_list),
            'processed': 0,
            'failed': 0,
            'failed_images': []
        }
        
        for idx, filename in enumerate(tqdm(image_list, desc=f"{shape_code}")):
            try:
                image_path = self.data_loader.get_image_path(shape_id, filename)
                self.process_single_image(image_path, shape_code, idx, save_results)
                results['processed'] += 1
            except Exception as e:
                results['failed'] += 1
                results['failed_images'].append((filename, str(e)))
                print(f"\nFailed to process {filename}: {e}")
        
        return results
    
    def process_all_shapes(self,
                          max_images_per_shape: Optional[int] = None,
                          save_results: bool = True) -> List[Dict]:
        """
        Process all shape categories in the dataset.
        
        Args:
            max_images_per_shape: Maximum images per category (None for all)
            save_results: Whether to save results to disk
            
        Returns:
            List of processing statistics for each shape
        """
        all_results = []
        
        print(f"Starting batch processing of all shapes...")
        print(f"Output directory: {self.output_dir}")
        print(f"GrabCut iterations: {self.iterations}")
        print(f"Annotations: {'Enabled' if self.add_annotations else 'Disabled'}")
        
        # Process each shape category
        for shape_id in range(1, 15):  # 1 to 14
            try:
                results = self.process_shape_category(
                    shape_id,
                    max_images_per_shape,
                    save_results
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nError processing shape {shape_id}: {e}")
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[Dict]) -> None:
        """
        Print processing summary.
        
        Args:
            results: List of processing results
        """
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        total_processed = sum(r['processed'] for r in results)
        total_failed = sum(r['failed'] for r in results)
        total_images = sum(r['total_images'] for r in results)
        
        print(f"\nTotal images: {total_images}")
        print(f"Successfully processed: {total_processed}")
        print(f"Failed: {total_failed}")
        print(f"Success rate: {(total_processed/total_images*100):.2f}%")
        
        print("\nPer-shape breakdown:")
        print("-" * 60)
        print(f"{'Shape':<15} {'Code':<6} {'Processed':<12} {'Failed':<8}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['shape_name']:<15} {result['shape_code']:<6} "
                  f"{result['processed']:<12} {result['failed']:<8}")
        
        print("="*60)
    
    def process_custom_image(self,
                           image_path: str,
                           output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a custom image (not from dataset).
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save result
            
        Returns:
            Tuple of (segmented_image, binary_mask)
        """
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Preprocess
        original, enhanced = preprocess_for_segmentation(image)
        
        # Segment
        segmented, mask = remove_background(original, enhanced, self.iterations)
        
        # Add annotations if requested
        if self.add_annotations:
            segmented = annotate_segmentation(segmented, mask)
        
        # Save if output path provided
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            cv2.imwrite(output_path, segmented)
        
        return segmented, mask


class BatchProcessor:
    """
    Helper class for batch processing with different configurations.
    """
    
    @staticmethod
    def quick_process(data_path: str,
                     variant: str = 'Shape_1d_256i',
                     output_dir: str = 'data/processed',
                     shape_id: Optional[int] = None) -> None:
        """
        Quick processing with default settings.
        
        Args:
            data_path: Path to dataset
            variant: Dataset variant
            output_dir: Output directory
            shape_id: Optional specific shape to process (None for all)
        """
        # Create data loader
        loader = DiamondDataLoader(data_path, variant)
        
        # Create processor
        processor = DiamondProcessor(
            data_loader=loader,
            output_dir=output_dir,
            iterations=4,
            add_annotations=False
        )
        
        # Process
        if shape_id is not None:
            processor.process_shape_category(shape_id)
        else:
            processor.process_all_shapes()
    
    @staticmethod
    def process_with_annotations(data_path: str,
                                variant: str = 'Shape_1d_256i',
                                output_dir: str = 'data/processed_annotated',
                                max_images: int = 10) -> None:
        """
        Process images with annotations (for visualization/debugging).
        
        Args:
            data_path: Path to dataset
            variant: Dataset variant
            output_dir: Output directory
            max_images: Max images per shape
        """
        loader = DiamondDataLoader(data_path, variant)
        
        processor = DiamondProcessor(
            data_loader=loader,
            output_dir=output_dir,
            iterations=4,
            add_annotations=True
        )
        
        processor.process_all_shapes(max_images_per_shape=max_images)


def process_diamond_dataset(data_path: str,
                           output_dir: str,
                           variant: str = 'Shape_1d_256i',
                           iterations: int = 4,
                           shape_id: Optional[int] = None) -> None:
    """
    Convenience function for processing diamond dataset.
    
    Args:
        data_path: Path to raw data directory
        output_dir: Path to output directory
        variant: Dataset variant to process
        iterations: Number of GrabCut iterations
        shape_id: Optional specific shape ID to process
    """
    loader = DiamondDataLoader(data_path, variant)
    processor = DiamondProcessor(loader, output_dir, iterations)
    
    if shape_id is not None:
        processor.process_shape_category(shape_id)
    else:
        processor.process_all_shapes()

# ============================================================================
# Enhanced Processing with Progress Tracking and Statistics
# ============================================================================

import time
import json
from datetime import datetime


class ProcessingStats:
    """Track and report processing statistics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.images_processed = 0
        self.images_failed = 0
        self.processing_times = []
        self.shape_stats = {}
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def end(self):
        """End timing."""
        self.end_time = time.time()
    
    def add_image(self, processing_time: float, success: bool = True):
        """
        Record image processing result.
        
        Args:
            processing_time: Time taken to process image
            success: Whether processing was successful
        """
        if success:
            self.images_processed += 1
            self.processing_times.append(processing_time)
        else:
            self.images_failed += 1
    
    def add_shape_stats(self, shape_code: str, stats: Dict):
        """
        Add statistics for a shape category.
        
        Args:
            shape_code: Shape code
            stats: Statistics dictionary
        """
        self.shape_stats[shape_code] = stats
    
    def get_summary(self) -> Dict:
        """
        Get processing summary.
        
        Returns:
            Dictionary with all statistics
        """
        total_time = self.end_time - self.start_time if self.end_time else 0
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'total_images': self.images_processed + self.images_failed,
            'processed': self.images_processed,
            'failed': self.images_failed,
            'total_time_seconds': total_time,
            'average_time_per_image': avg_time,
            'images_per_second': self.images_processed / total_time if total_time > 0 else 0,
            'shape_stats': self.shape_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_to_file(self, output_path: str):
        """
        Save statistics to JSON file.
        
        Args:
            output_path: Path to save statistics
        """
        with open(output_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("DETAILED PROCESSING STATISTICS")
        print("="*70)
        print(f"\nTotal images: {summary['total_images']}")
        print(f"Successfully processed: {summary['processed']}")
        print(f"Failed: {summary['failed']}")
        
        if summary['total_images'] > 0:
            print(f"Success rate: {(summary['processed']/summary['total_images']*100):.2f}%")
        
        print(f"\nTotal processing time: {summary['total_time_seconds']:.2f} seconds")
        print(f"Average time per image: {summary['average_time_per_image']:.3f} seconds")
        print(f"Processing speed: {summary['images_per_second']:.2f} images/second")
        print("="*70)


class AdvancedDiamondProcessor(DiamondProcessor):
    """
    Enhanced processor with advanced features like progress tracking,
    statistics collection, and error handling.
    """
    
    def __init__(self,
                 data_loader: DiamondDataLoader,
                 output_dir: str = 'data/processed',
                 iterations: int = 4,
                 add_annotations: bool = False,
                 save_masks: bool = False,
                 save_stats: bool = True):
        """
        Initialize advanced processor.
        
        Args:
            data_loader: Data loader instance
            output_dir: Directory to save processed images
            iterations: Number of GrabCut iterations
            add_annotations: Whether to add contour/bbox annotations
            save_masks: Whether to save binary masks
            save_stats: Whether to save processing statistics
        """
        super().__init__(data_loader, output_dir, iterations, add_annotations)
        self.save_masks = save_masks
        self.save_stats = save_stats
        self.stats = ProcessingStats()
        
        # Create subdirectories
        if save_masks:
            self.mask_dir = os.path.join(output_dir, 'masks')
            ensure_dir(self.mask_dir)
    
    def process_single_image(self,
                           image_path: str,
                           shape_code: str,
                           index: int,
                           save_result: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single diamond image with timing.
        
        Args:
            image_path: Path to input image
            shape_code: Shape code for the diamond
            index: Image index for naming
            save_result: Whether to save the result to disk
            
        Returns:
            Tuple of (segmented_image, binary_mask)
        """
        start_time = time.time()
        
        try:
            # Read image
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Preprocess
            original, enhanced = preprocess_for_segmentation(image)
            
            # Segment
            segmented, mask = remove_background(original, enhanced, self.iterations)
            
            # Add annotations if requested
            if self.add_annotations:
                segmented = annotate_segmentation(segmented, mask)
            
            # Save results if requested
            if save_result:
                # Save segmented image
                output_filename = build_output_filename(shape_code, index, 'result', '.png')
                output_path = os.path.join(self.output_dir, output_filename)
                cv2.imwrite(output_path, segmented)
                
                # Save mask if requested
                if self.save_masks:
                    mask_filename = build_output_filename(shape_code, index, 'mask', '.png')
                    mask_path = os.path.join(self.mask_dir, mask_filename)
                    cv2.imwrite(mask_path, mask * 255)
            
            # Record success
            processing_time = time.time() - start_time
            self.stats.add_image(processing_time, success=True)
            
            return segmented, mask
            
        except Exception as e:
            # Record failure
            processing_time = time.time() - start_time
            self.stats.add_image(processing_time, success=False)
            raise e
    
    def process_all_shapes(self,
                          max_images_per_shape: Optional[int] = None,
                          save_results: bool = True) -> List[Dict]:
        """
        Process all shape categories with statistics tracking.
        
        Args:
            max_images_per_shape: Maximum images per category (None for all)
            save_results: Whether to save results to disk
            
        Returns:
            List of processing statistics for each shape
        """
        self.stats.start()
        
        results = super().process_all_shapes(max_images_per_shape, save_results)
        
        self.stats.end()
        
        # Save statistics if requested
        if self.save_stats:
            stats_path = os.path.join(self.output_dir, 'processing_stats.json')
            self.stats.save_to_file(stats_path)
            print(f"\nStatistics saved to: {stats_path}")
        
        # Print detailed summary
        self.stats.print_summary()
        
        return results


class InteractiveProcessor:
    """
    Interactive processor for command-line usage.
    """
    
    def __init__(self, data_path: str, output_dir: str = 'data/processed'):
        """
        Initialize interactive processor.
        
        Args:
            data_path: Path to raw data
            output_dir: Output directory
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.mapper = DiamondShapeMapper()
    
    def select_shape(self) -> int:
        """
        Prompt user to select a shape.
        
        Returns:
            Shape ID
        """
        print("\n" + "="*60)
        print("DIAMOND SHAPE SELECTION")
        print("="*60)
        
        shapes = self.mapper.list_all_shapes()
        for i, code in enumerate(shapes, 1):
            name = self.mapper.get_shape_name(code)
            print(f"{i:2d}. {code:<6} - {name}")
        
        print("="*60)
        
        while True:
            try:
                choice = int(input("\nEnter shape number (1-14): "))
                if 1 <= choice <= 14:
                    return choice
                else:
                    print("Invalid choice. Please enter a number between 1 and 14.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def select_variant(self) -> str:
        """
        Prompt user to select dataset variant.
        
        Returns:
            Variant name
        """
        variants = ['Shape_1d_256i', 'Shape_5d_256i', 'Shape_10d_256i']
        
        print("\n" + "="*60)
        print("DATASET VARIANT SELECTION")
        print("="*60)
        print("1. Shape_1d_256i  - Single diamond per image")
        print("2. Shape_5d_256i  - 5 diamonds per image")
        print("3. Shape_10d_256i - 10 diamonds per image")
        print("="*60)
        
        while True:
            try:
                choice = int(input("\nEnter variant number (1-3): "))
                if 1 <= choice <= 3:
                    return variants[choice - 1]
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def run(self):
        """Run interactive processing session."""
        print("\n" + "="*60)
        print("DIAMOND SEGMENTATION - INTERACTIVE MODE")
        print("="*60)
        
        # Select variant
        variant = self.select_variant()
        
        # Select shape
        shape_id = self.select_shape()
        shape_code = self.mapper.get_shape_code(shape_id)
        shape_name = self.mapper.get_shape_name(shape_code)
        
        # Get number of images to process
        while True:
            try:
                max_images = input("\nNumber of images to process (press Enter for all): ")
                if max_images.strip() == "":
                    max_images = None
                    break
                else:
                    max_images = int(max_images)
                    if max_images > 0:
                        break
                    else:
                        print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter a number or press Enter.")
        
        # Confirm
        print("\n" + "-"*60)
        print("PROCESSING CONFIGURATION")
        print("-"*60)
        print(f"Dataset variant: {variant}")
        print(f"Shape: {shape_name} ({shape_code})")
        print(f"Images to process: {'All' if max_images is None else max_images}")
        print(f"Output directory: {self.output_dir}")
        print("-"*60)
        
        confirm = input("\nProceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Processing cancelled.")
            return
        
        # Process
        loader = DiamondDataLoader(self.data_path, variant)
        processor = DiamondProcessor(loader, self.output_dir)
        
        processor.process_shape_category(shape_id, max_images)
        
        print("\nProcessing complete!")


def run_interactive_mode(data_path: str, output_dir: str = 'data/processed'):
    """
    Run processor in interactive mode.
    
    Args:
        data_path: Path to raw data
        output_dir: Output directory
    """
    processor = InteractiveProcessor(data_path, output_dir)
    processor.run()