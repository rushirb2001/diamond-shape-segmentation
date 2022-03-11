"""
Data loader module for diamond image datasets.
Handles shape category mappings and directory structure.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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
    
    def __init__(self, 
                variant: str = 'Shape_1d_256i | Shape_5d_256i | Shape_10d_256i',
        ) -> None:
        """
        Initialize data loader.
        
        Args:
            base_path: Root directory containing dataset folders (can be relative or absolute)
            variant: Dataset variant to use (default: Shape_1d_256i)
        
        Raises:
            ValueError: If variant is not supported
            FileNotFoundError: If dataset path doesn't exist
        """
        if variant not in self.DATASET_VARIANTS:
            raise ValueError(f"Variant must be one of {self.DATASET_VARIANTS}")
        
        # Get the path to the raw data directory (src/data/raw)
        self.base_path = Path(__file__).parent / 'raw'
        self.variant = variant
        self.dataset_path = self.base_path / variant
        self.mapper = DiamondShapeMapper()
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            # Try without variant suffix (in case images are directly in base_path)
            if self._check_shape_folders_in_base():
                self.dataset_path = self.base_path
                print(f"Note: Using base path directly as shape folders found: {self.base_path}")
            else:
                raise FileNotFoundError(
                    f"Dataset path not found: {self.dataset_path}\n"
                    f"Base path: {self.base_path}\n"
                    f"Please check that the path exists and contains the dataset variant folders."
                )
    
    def _check_shape_folders_in_base(self) -> bool:
        """
        Check if shape folders exist directly in base path.
        
        Returns:
            True if at least one shape folder is found
        """
        if not self.base_path.exists():
            return False
        
        shape_codes = self.mapper.list_all_shapes()
        for code in shape_codes:
            if (self.base_path / code).exists():
                return True
        return False
    
    def get_shape_directory(self, shape_id: int) -> Optional[Path]:
        """
        Get full path to shape directory.
        
        Args:
            shape_id: Numeric shape identifier (1-14)
            
        Returns:
            Full directory path or None if invalid
        """
        shape_code = self.mapper.get_shape_code(shape_id)
        if shape_code:
            return self.dataset_path / shape_code
        return None
    
    def get_shape_directory_by_code(self, shape_code: str) -> Path:
        """
        Get full path to shape directory by code.
        
        Args:
            shape_code: Two-letter shape code
            
        Returns:
            Full directory path
        """
        return self.dataset_path / shape_code
    
    def list_images(self, shape_id: int) -> List[str]:
        """
        List all images in a shape directory.
        
        Args:
            shape_id: Numeric shape identifier
            
        Returns:
            Sorted list of image filenames
        """
        shape_dir = self.get_shape_directory(shape_id)
        if shape_dir and shape_dir.exists():
            files = [f.name for f in shape_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
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
        if shape_dir.exists():
            files = [f.name for f in shape_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
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
            Full path to image file as string
        """
        shape_dir = self.get_shape_directory(shape_id)
        return str(shape_dir / filename)
    
    def get_image_path_by_code(self, shape_code: str, filename: str) -> str:
        """
        Get full path to specific image by code.
        
        Args:
            shape_code: Two-letter shape code
            filename: Image filename
            
        Returns:
            Full path to image file as string
        """
        shape_dir = self.get_shape_directory_by_code(shape_code)
        return str(shape_dir / filename)
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        info = {
            'variant': self.variant,
            'base_path': str(self.base_path),
            'dataset_path': str(self.dataset_path),
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
            if not shape_dir.exists():
                issues.append(f"Missing directory: {shape_code}")
                continue
            
            # Check image count
            images = self.list_images_by_code(shape_code)
            if len(images) == 0:
                issues.append(f"No images found in {shape_code}")
            elif len(images) != 256:
                issues.append(f"Expected 256 images in {shape_code}, found {len(images)}")
        
        return len(issues) == 0, issues


def load_dataset_config(config_path: Union[str, Path]) -> Dict:
    """
    Load dataset configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path).resolve()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config