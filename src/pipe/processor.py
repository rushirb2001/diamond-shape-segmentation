"""
Main processing pipeline for diamond segmentation.
Handles batch processing of diamond images with progress tracking.
"""

import cv2
import os
import numpy as np
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

from src.data.loader import DiamondDataLoader, DiamondShapeMapper
from src.utils.segmentation import (
    preprocess_for_segmentation,
    remove_background,
    segment_with_postprocessing
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
                 output_dir: str = 'data/processed',
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
        self.output_dir = output_dir
        self.iterations = iterations
        self.add_annotations = add_annotations
        self.mapper = DiamondShapeMapper()
        
        # Ensure output directory exists
        ensure_dir(output_dir)
    
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
            cv2.imwrite(output_path, segmented)
        
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