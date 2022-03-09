"""
Output management utilities for saving processed images and results.
Handles directory creation, file naming, and result organization.
"""

import cv2
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.file_utils import ensure_dir, build_output_filename


class OutputManager:
    """
    Manages output directory structure and file saving.
    """
    
    def __init__(self, base_output_dir: str, create_subdirs: bool = True):
        """
        Initialize output manager.
        
        Args:
            base_output_dir: Base directory for all outputs
            create_subdirs: Whether to create subdirectories for different output types
        """
        self.base_output_dir = base_output_dir
        self.create_subdirs = create_subdirs
        
        # Create base directory
        ensure_dir(base_output_dir)
        
        # Create subdirectories if requested
        if create_subdirs:
            self.segmented_dir = os.path.join(base_output_dir, 'segmented')
            self.masks_dir = os.path.join(base_output_dir, 'masks')
            self.annotated_dir = os.path.join(base_output_dir, 'annotated')
            self.stats_dir = os.path.join(base_output_dir, 'stats')
            
            ensure_dir(self.segmented_dir)
            ensure_dir(self.masks_dir)
            ensure_dir(self.annotated_dir)
            ensure_dir(self.stats_dir)
        else:
            self.segmented_dir = base_output_dir
            self.masks_dir = base_output_dir
            self.annotated_dir = base_output_dir
            self.stats_dir = base_output_dir
    
    def save_segmented_image(self,
                            image: np.ndarray,
                            shape_code: str,
                            index: int,
                            suffix: str = 'result') -> str:
        """
        Save segmented image.
        
        Args:
            image: Segmented image array
            shape_code: Shape code
            index: Image index
            suffix: Filename suffix
            
        Returns:
            Path to saved file
        """
        filename = build_output_filename(shape_code, index, suffix, '.png')
        output_path = os.path.join(self.segmented_dir, filename)
        cv2.imwrite(output_path, image)
        return output_path
    
    def save_mask(self,
                 mask: np.ndarray,
                 shape_code: str,
                 index: int) -> str:
        """
        Save binary mask.
        
        Args:
            mask: Binary mask array (0 or 1)
            shape_code: Shape code
            index: Image index
            
        Returns:
            Path to saved file
        """
        filename = build_output_filename(shape_code, index, 'mask', '.png')
        output_path = os.path.join(self.masks_dir, filename)
        # Convert to 0-255 range for saving
        cv2.imwrite(output_path, mask * 255)
        return output_path
    
    def save_annotated_image(self,
                            image: np.ndarray,
                            shape_code: str,
                            index: int) -> str:
        """
        Save annotated image with contours and bounding boxes.
        
        Args:
            image: Annotated image array
            shape_code: Shape code
            index: Image index
            
        Returns:
            Path to saved file
        """
        filename = build_output_filename(shape_code, index, 'annotated', '.png')
        output_path = os.path.join(self.annotated_dir, filename)
        cv2.imwrite(output_path, image)
        return output_path
    
    def save_comparison(self,
                       original: np.ndarray,
                       segmented: np.ndarray,
                       shape_code: str,
                       index: int) -> str:
        """
        Save side-by-side comparison of original and segmented images.
        
        Args:
            original: Original image
            segmented: Segmented image
            shape_code: Shape code
            index: Image index
            
        Returns:
            Path to saved file
        """
        # Concatenate images horizontally
        comparison = np.hstack([original, segmented])
        
        filename = build_output_filename(shape_code, index, 'comparison', '.png')
        output_path = os.path.join(self.base_output_dir, filename)
        cv2.imwrite(output_path, comparison)
        return output_path
    
    def save_processing_log(self,
                           shape_code: str,
                           log_data: Dict) -> str:
        """
        Save processing log for a shape category.
        
        Args:
            shape_code: Shape code
            log_data: Log data dictionary
            
        Returns:
            Path to saved file
        """
        filename = f"{shape_code}_processing_log.json"
        output_path = os.path.join(self.stats_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return output_path
    
    def save_batch_statistics(self, stats: Dict) -> str:
        """
        Save batch processing statistics.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"batch_stats_{timestamp}.json"
        output_path = os.path.join(self.stats_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return output_path
    
    def get_output_summary(self) -> Dict:
        """
        Get summary of output directory contents.
        
        Returns:
            Dictionary with file counts and sizes
        """
        summary = {
            'base_dir': self.base_output_dir,
            'segmented_count': len([f for f in os.listdir(self.segmented_dir) 
                                   if f.endswith('.png')]) if os.path.exists(self.segmented_dir) else 0,
            'masks_count': len([f for f in os.listdir(self.masks_dir) 
                               if f.endswith('.png')]) if os.path.exists(self.masks_dir) else 0,
            'annotated_count': len([f for f in os.listdir(self.annotated_dir) 
                                   if f.endswith('.png')]) if os.path.exists(self.annotated_dir) else 0,
        }
        
        return summary


class ResultCollector:
    """
    Collects and organizes processing results.
    """
    
    def __init__(self):
        """Initialize result collector."""
        self.results = []
        self.errors = []
    
    def add_result(self,
                  shape_code: str,
                  image_index: int,
                  image_path: str,
                  output_path: str,
                  processing_time: float,
                  success: bool = True,
                  error_message: Optional[str] = None):
        """
        Add a processing result.
        
        Args:
            shape_code: Shape code
            image_index: Image index
            image_path: Input image path
            output_path: Output image path
            processing_time: Time taken to process
            success: Whether processing succeeded
            error_message: Error message if failed
        """
        result = {
            'shape_code': shape_code,
            'image_index': image_index,
            'image_path': image_path,
            'output_path': output_path,
            'processing_time': processing_time,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if not success:
            result['error_message'] = error_message
            self.errors.append(result)
        
        self.results.append(result)
    
    def get_summary(self) -> Dict:
        """
        Get summary of all results.
        
        Returns:
            Dictionary with summary statistics
        """
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        processing_times = [r['processing_time'] for r in successful]
        
        summary = {
            'total_processed': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'average_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'min_time': min(processing_times) if processing_times else 0,
            'max_time': max(processing_times) if processing_times else 0,
            'errors': self.errors
        }
        
        return summary
    
    def save_to_file(self, output_path: str):
        """
        Save results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        summary = self.get_summary()
        summary['all_results'] = self.results
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def clear(self):
        """Clear all collected results."""
        self.results = []
        self.errors = []


def create_output_structure(base_dir: str) -> Dict[str, str]:
    """
    Create standard output directory structure.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary mapping output types to paths
    """
    structure = {
        'base': base_dir,
        'segmented': os.path.join(base_dir, 'segmented'),
        'masks': os.path.join(base_dir, 'masks'),
        'annotated': os.path.join(base_dir, 'annotated'),
        'comparisons': os.path.join(base_dir, 'comparisons'),
        'stats': os.path.join(base_dir, 'stats'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    # Create all directories
    for dir_path in structure.values():
        ensure_dir(dir_path)
    
    return structure


def save_results_batch(images: List[np.ndarray],
                      masks: List[np.ndarray],
                      shape_code: str,
                      output_dir: str,
                      start_index: int = 0) -> List[str]:
    """
    Save multiple results in batch.
    
    Args:
        images: List of segmented images
        masks: List of binary masks
        shape_code: Shape code
        output_dir: Output directory
        start_index: Starting index for numbering
        
    Returns:
        List of output file paths
    """
    output_paths = []
    
    for i, (image, mask) in enumerate(zip(images, masks)):
        index = start_index + i
        filename = build_output_filename(shape_code, index, 'result', '.png')
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)
        output_paths.append(output_path)
    
    return output_paths


def generate_output_report(output_dir: str,
                          results: List[Dict],
                          report_path: Optional[str] = None) -> str:
    """
    Generate a text report of processing results.
    
    Args:
        output_dir: Output directory
        results: List of result dictionaries
        report_path: Optional path for report file
        
    Returns:
        Path to generated report
    """
    if report_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'processing_report_{timestamp}.txt')
    
    successful = [r for r in results if r.get('success', True)]
    failed = [r for r in results if not r.get('success', True)]
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DIAMOND SEGMENTATION PROCESSING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Images Processed: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        
        if len(results) > 0:
            f.write(f"Success Rate: {len(successful)/len(results)*100:.2f}%\n")
        
        f.write("\n")
        
        if failed:
            f.write("-"*70 + "\n")
            f.write("FAILED IMAGES\n")
            f.write("-"*70 + "\n")
            for r in failed:
                f.write(f"Shape: {r.get('shape_code', 'N/A')}\n")
                f.write(f"Index: {r.get('image_index', 'N/A')}\n")
                f.write(f"Error: {r.get('error_message', 'Unknown error')}\n")
                f.write("\n")
        
        f.write("="*70 + "\n")
    
    return report_path