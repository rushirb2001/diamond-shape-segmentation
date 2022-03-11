"""
Automated demo video generation script.
Generates triple-split and five-split videos for multiple diamond shapes.

Usage:
    python scripts/generate_demo_videos.py --data-path data/raw --output-dir demo_videos
"""

import argparse
import sys
from pathlib import Path
import glob
import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DiamondDataLoader, DiamondShapeMapper
from src.pipe.video_comparison import VideoComparator, create_comparison_grid
from src.utils.segmentation import preprocess_for_segmentation, remove_background
from src.utils.visualization import add_text_overlay
from src.utils.file_utils import ensure_dir


# Selected shapes for demo
DEMO_SHAPES = ['AS', 'BR', 'EM', 'MQ', 'OV']
SHAPE_NAMES = {
    'AS': 'Asscher',
    'BR': 'Brilliant',
    'EM': 'Emerald',
    'MQ': 'Marquise',
    'OV': 'Oval'
}


class DemoVideoGenerator:
    """
    Generates demo videos showing algorithm performance.
    """
    
    def __init__(self,
                 data_path: str,
                 output_dir: str,
                 fps: int = 15,
                 iterations: int = 5):
        """
        Initialize demo generator.
        
        Args:
            data_path: Path to raw data directory
            output_dir: Output directory for videos
            fps: Frames per second
            iterations: Number of GrabCut iterations
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.iterations = iterations
        
        # Create output directory
        ensure_dir(self.output_dir)
        
        # Initialize loaders
        self.loader_1d = DiamondDataLoader(data_path, 'Shape_1d_256i')
        self.loader_5d = DiamondDataLoader(data_path, 'Shape_5d_256i')
        self.mapper = DiamondShapeMapper()
    
    def generate_triple_split_video(self, shape_code: str) -> str:
        """
        Generate triple-split video (Before | CLAHE | After) for a shape.
        
        Args:
            shape_code: Shape code (e.g., 'AS', 'BR')
            
        Returns:
            Path to created video
        """
        print(f"\n{'='*60}")
        print(f"Generating Triple-Split Video: {SHAPE_NAMES[shape_code]} ({shape_code})")
        print(f"{'='*60}")
        
        # Get images from 1d dataset
        shape_id = self.mapper.get_shape_id(shape_code)
        image_list = self.loader_1d.list_images(shape_id)
        
        if not image_list:
            print(f"No images found for shape {shape_code}")
            return None
        
        print(f"Found {len(image_list)} images")
        
        # Output path
        output_path = self.output_dir / f"triple_split_{shape_code}_{SHAPE_NAMES[shape_code]}.avi"
        
        # Create video comparator
        comparator = VideoComparator(output_path, fps=self.fps)
        
        # Get image paths
        image_paths = [
            self.loader_1d.get_image_path(shape_id, img_name)
            for img_name in image_list
        ]
        
        # Generate video
        comparator.create_triple_split(
            image_paths,
            iterations=self.iterations,
            add_labels=True,
            label_color=(255, 255, 255),
            label_bg_color=(0, 0, 0)
        )
        
        print(f"✓ Triple-split video saved: {output_path}")
        return str(output_path)
    
    def generate_five_split_video(self, shape_code: str) -> str:
        """
        Generate five-split video showing 5 variations of same shape.
        
        Args:
            shape_code: Shape code (e.g., 'AS', 'BR')
            
        Returns:
            Path to created video
        """
        print(f"\n{'='*60}")
        print(f"Generating Five-Split Video: {SHAPE_NAMES[shape_code]} ({shape_code})")
        print(f"{'='*60}")
        
        # Get images from 5d dataset
        shape_id = self.mapper.get_shape_id(shape_code)
        all_images = self.loader_5d.list_images(shape_id)
        
        if not all_images:
            print(f"No images found for shape {shape_code} in 5d dataset")
            return None
        
        print(f"Found {len(all_images)} total images in 5d dataset")
        
        # Group images by variation (each variation has 256 images)
        # Assuming naming pattern allows grouping
        # For simplicity, take first 256 images from each of 5 variations
        variations = []
        images_per_variation = 256
        
        for var_idx in range(5):
            start_idx = var_idx * images_per_variation
            end_idx = start_idx + images_per_variation
            
            if end_idx <= len(all_images):
                var_images = all_images[start_idx:end_idx]
                variations.append(var_images)
        
        if len(variations) < 5:
            print(f"Warning: Only found {len(variations)} complete variations")
        
        print(f"Processing {len(variations)} variations with {images_per_variation} images each")
        
        # Output path
        output_path = self.output_dir / f"five_split_{shape_code}_{SHAPE_NAMES[shape_code]}.avi"
        
        # Process all variations
        all_frames = [[] for _ in range(5)]
        
        for var_idx in range(len(variations)):
            print(f"  Processing variation {var_idx + 1}/5...")
            
            for img_name in tqdm(variations[var_idx], desc=f"  Var {var_idx+1}"):
                img_path = self.loader_5d.get_image_path(shape_id, img_name)
                original = cv2.imread(img_path)
                
                if original is None:
                    continue
                
                # Process
                _, enhanced = preprocess_for_segmentation(original)
                segmented, _ = remove_background(original, enhanced, self.iterations)
                
                all_frames[var_idx].append(segmented)
        
        # Create labels
        labels = [f"{SHAPE_NAMES[shape_code]} Var {i+1}" for i in range(len(variations))]
        
        # Generate grid video
        create_comparison_grid(
            all_frames,
            labels,
            output_path,
            fps=self.fps,
            grid_layout=(1, 5)  # 1 row, 5 columns
        )
        
        print(f"✓ Five-split video saved: {output_path}")
        return str(output_path)
    
    def generate_all_demos(self, shapes: list = None) -> dict:
        """
        Generate all demo videos.
        
        Args:
            shapes: List of shape codes to process (default: DEMO_SHAPES)
            
        Returns:
            Dictionary with generated video paths
        """
        if shapes is None:
            shapes = DEMO_SHAPES
        
        results = {
            'triple_split': {},
            'five_split': {}
        }
        
        print("\n" + "="*60)
        print("DIAMOND SEGMENTATION DEMO VIDEO GENERATION")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Shapes to process: {', '.join(shapes)}")
        print(f"FPS: {self.fps}")
        print(f"GrabCut iterations: {self.iterations}")
        print("="*60)
        
        # Generate triple-split videos (from 1d dataset)
        print("\n" + "="*60)
        print("PHASE 1: Triple-Split Videos (Before | CLAHE | After)")
        print("="*60)
        
        for shape in shapes:
            try:
                video_path = self.generate_triple_split_video(shape)
                results['triple_split'][shape] = video_path
            except Exception as e:
                print(f"✗ Error generating triple-split for {shape}: {e}")
                results['triple_split'][shape] = None
        
        # Generate five-split videos (from 5d dataset)
        print("\n" + "="*60)
        print("PHASE 2: Five-Split Videos (5 Variations)")
        print("="*60)
        
        for shape in shapes:
            try:
                video_path = self.generate_five_split_video(shape)
                results['five_split'][shape] = video_path
            except Exception as e:
                print(f"✗ Error generating five-split for {shape}: {e}")
                results['five_split'][shape] = None
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: dict):
        """Print generation summary."""
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        
        print("\nTriple-Split Videos:")
        for shape, path in results['triple_split'].items():
            status = "✓" if path else "✗"
            print(f"  {status} {SHAPE_NAMES[shape]} ({shape}): {path if path else 'Failed'}")
        
        print("\nFive-Split Videos:")
        for shape, path in results['five_split'].items():
            status = "✓" if path else "✗"
            print(f"  {status} {SHAPE_NAMES[shape]} ({shape}): {path if path else 'Failed'}")
        
        total_success = sum(1 for p in results['triple_split'].values() if p) + \
                       sum(1 for p in results['five_split'].values() if p)
        total_videos = len(results['triple_split']) + len(results['five_split'])
        
        print(f"\nTotal: {total_success}/{total_videos} videos generated successfully")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate demo videos for diamond segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw',
        help='Path to raw data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='demo_videos',
        help='Output directory for generated videos'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='Frames per second for videos'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of GrabCut iterations'
    )
    parser.add_argument(
        '--shapes',
        type=str,
        nargs='+',
        default=None,
        help='Specific shapes to process (default: AS BR EM MQ OV)'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = DemoVideoGenerator(
        data_path=args.data_path,
        output_dir=args.output_dir,
        fps=args.fps,
        iterations=args.iterations
    )
    
    # Generate demos
    shapes = args.shapes if args.shapes else DEMO_SHAPES
    generator.generate_all_demos(shapes)


if __name__ == '__main__':
    main()