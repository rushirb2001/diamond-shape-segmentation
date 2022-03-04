"""
Data loader module for diamond image datasets.
Handles shape category mappings and directory structure.
"""

import os
from typing import Dict, List, Optional, Tuple
import yaml


class DiamondShapeMapper:
    """Maps diamond shape codes to full names and handles directory paths."""
    
    SHAPE_MAPPING: Dict[int, str] = {
        1: 'AS',    # Asscher
        2: 'BR',    # Brilliant
        3: 'CMB',   # Combination
        4: 'EM',    # Emerald
        5: 'HS',    # Heart Shape
        6: 'MQ',    # Marquise
        7: 'OV',    # Oval
        8: 'PE',    # Pear
        9: 'PR',    # Princess
        10: 'PS',   # Pearshape
        11: 'RA',   # Radiant
        12: 'RD',   # Round Diamond
        13: 'SEM',  # Semi
        14: 'TRI'   # Triangle
    }
    
    SHAPE_NAMES: Dict[str, str] = {
        'AS': 'Asscher',
        'BR': 'Brilliant',
        'CMB': 'Combination',
        'EM': 'Emerald',
        'HS': 'Heart Shape',
        'MQ': 'Marquise',
        'OV': 'Oval',
        'PE': 'Pear',
        'PR': 'Princess',
        'PS': 'Pearshape',
        'RA': 'Radiant',
        'RD': 'Round',
        'SEM': 'Semi',
        'TRI': 'Triangle'
    }
    
    @classmethod
    def get_shape_code(cls, shape_id: int) -> Optional[str]:
        """
        Get shape code from numeric ID.
        
        Args:
            shape_id: Numeric identifier (1-14)
            
        Returns:
            Two-letter shape code or None if invalid
        """
        return cls.SHAPE_MAPPING.get(shape_id)
    
    @classmethod
    def get_shape_name(cls, shape_code: str) -> Optional[str]:
        """
        Get full shape name from code.
        
        Args:
            shape_code: Two-letter shape code
            
        Returns:
            Full shape name or None if invalid
        """
        return cls.SHAPE_NAMES.get(shape_code)
    
    @classmethod
    def get_shape_id(cls, shape_code: str) -> Optional[int]:
        """
        Get numeric ID from shape code.
        
        Args:
            shape_code: Two-letter shape code
            
        Returns:
            Numeric identifier or None if invalid
        """
        for id, code in cls.SHAPE_MAPPING.items():
            if code == shape_code:
                return id
        return None
    
    @classmethod
    def list_all_shapes(cls) -> List[str]:
        """
        Get list of all shape codes.
        
        Returns:
            List of all two-letter shape codes
        """
        return list(cls.SHAPE_NAMES.keys())
    
    @classmethod
    def get_shape_count(cls) -> int:
        """Get total number of shape categories."""
        return len(cls.SHAPE_NAMES)


class DiamondDataLoader:
    """
    Handles loading and organizing diamond image datasets.
    Supports multiple dataset variants (1d, 5d, 10d).
    """
    
    DATASET_VARIANTS = ['Shape_1d_256i', 'Shape_5d_256i', 'Shape_10d_256i']
    
    def __init__(self, base_path: str, variant: str = 'Shape_1d_256i'):
        """
        Initialize data loader.
        
        Args:
            base_path: Root directory containing dataset folders
            variant: Dataset variant to use (default: Shape_1d_256i)
        
        Raises:
            ValueError: If variant is not supported
            FileNotFoundError: If dataset path doesn't exist
        """
        if variant not in self.DATASET_VARIANTS:
            raise ValueError(f"Variant must be one of {self.DATASET_VARIANTS}")
        
        self.base_path = base_path
        self.variant = variant
        self.dataset_path = os.path.join(base_path, variant)
        self.mapper = DiamondShapeMapper()
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
    
    def get_shape_directory(self, shape_id: int) -> Optional[str]:
        """
        Get full path to shape directory.
        
        Args:
            shape_id: Numeric shape identifier (1-14)
            
        Returns:
            Full directory path or None if invalid
        """
        shape_code = self.mapper.get_shape_code(shape_id)
        if shape_code:
            return os.path.join(self.dataset_path, shape_code)
        return None
    
    def get_shape_directory_by_code(self, shape_code: str) -> str:
        """
        Get full path to shape directory by code.
        
        Args:
            shape_code: Two-letter shape code
            
        Returns:
            Full directory path
        """
        return os.path.join(self.dataset_path, shape_code)
    
    def list_images(self, shape_id: int) -> List[str]:
        """
        List all images in a shape directory.
        
        Args:
            shape_id: Numeric shape identifier
            
        Returns:
            Sorted list of image filenames
        """
        shape_dir = self.get_shape_directory(shape_id)
        if shape_dir and os.path.exists(shape_dir):
            files = [f for f in os.listdir(shape_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files.sort()
            return files
        return []
    
    def list_images_by_code(self, shape_code: str) -> List[str]:
        """
        List all images in a shape directory by code.
        
        Args:
            shape_code: Two-letter shape code
            
        Returns:
            Sorted list of image filenames
        """
        shape_dir = self.get_shape_directory_by_code(shape_code)
        if os.path.exists(shape_dir):
            files = [f for f in os.listdir(shape_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files.sort()
            return files
        return []
    
    def get_image_path(self, shape_id: int, filename: str) -> str:
        """
        Get full path to specific image.
        
        Args:
            shape_id: Numeric shape identifier
            filename: Image filename
            
        Returns:
            Full path to image file
        """
        shape_dir = self.get_shape_directory(shape_id)
        return os.path.join(shape_dir, filename)
    
    def get_image_path_by_code(self, shape_code: str, filename: str) -> str:
        """
        Get full path to specific image by code.
        
        Args:
            shape_code: Two-letter shape code
            filename: Image filename
            
        Returns:
            Full path to image file
        """
        shape_dir = self.get_shape_directory_by_code(shape_code)
        return os.path.join(shape_dir, filename)
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        info = {
            'variant': self.variant,
            'base_path': self.base_path,
            'dataset_path': self.dataset_path,
            'shapes': {},
            'total_images': 0
        }
        
        for shape_code in self.mapper.list_all_shapes():
            images = self.list_images_by_code(shape_code)
            image_count = len(images)
            info['shapes'][shape_code] = {
                'name': self.mapper.get_shape_name(shape_code),
                'image_count': image_count
            }
            info['total_images'] += image_count
        
        return info
    
    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset structure and completeness.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if all shape directories exist
        for shape_code in self.mapper.list_all_shapes():
            shape_dir = self.get_shape_directory_by_code(shape_code)
            if not os.path.exists(shape_dir):
                issues.append(f"Missing directory: {shape_code}")
                continue
            
            # Check image count
            images = self.list_images_by_code(shape_code)
            if len(images) == 0:
                issues.append(f"No images found in {shape_code}")
            elif len(images) != 256:
                issues.append(f"Expected 256 images in {shape_code}, found {len(images)}")
        
        return len(issues) == 0, issues


def load_dataset_config(config_path: str) -> Dict:
    """
    Load dataset configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config