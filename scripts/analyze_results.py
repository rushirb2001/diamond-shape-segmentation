"""
Results analysis script for diamond segmentation.
Analyzes segmentation quality, generates reports, and creates visualizations.

Usage:
    python scripts/analyze_results.py --input data/processed --output analysis/
"""

import argparse
import sys
from pathlib import Path
import json
import csv
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DiamondShapeMapper
from src.utils.file_utils import ensure_dir


class ResultsAnalyzer:
    """
    Analyzes segmentation results and generates reports.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize analyzer.
        
        Args:
            input_dir: Directory with processed results
            output_dir: Directory for analysis outputs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.mapper = DiamondShapeMapper()
        
        # Create output directory
        ensure_dir(self.output_dir)
        
        # Storage for metrics
        self.metrics = defaultdict(list)
        self.shape_metrics = defaultdict(lambda: defaultdict(list))
    
    def analyze_image_quality(self, image_path: str) -> Dict:
        """
        Analyze quality metrics for a single image.
        
        Args:
            image_path: Path to segmented image
            
        Returns:
            Dictionary of quality metrics
        """
        img = cv2.imread(str(image_path))
        
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        metrics = {}
        
        # Foreground percentage (non-black pixels)
        non_black = np.sum(gray > 10)
        total_pixels = gray.size
        metrics['foreground_percentage'] = (non_black / total_pixels) * 100
        
        # Mean intensity
        metrics['mean_intensity'] = np.mean(gray[gray > 10]) if non_black > 0 else 0
        
        # Standard deviation (contrast)
        metrics['contrast'] = np.std(gray[gray > 10]) if non_black > 0 else 0
        
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.sum(edges > 0)
        metrics['edge_density'] = (edge_pixels / non_black) * 100 if non_black > 0 else 0
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['sharpness'] = laplacian.var()
        
        # Image dimensions
        metrics['height'] = img.shape[0]
        metrics['width'] = img.shape[1]
        
        return metrics
    
    def analyze_shape_category(self, shape_code: str) -> Dict:
        """
        Analyze all images for a shape category.
        
        Args:
            shape_code: Shape code (e.g., 'AS', 'BR')
            
        Returns:
            Aggregated metrics for the shape
        """
        print(f"\nAnalyzing shape: {shape_code}")
        
        # Find images for this shape
        shape_dir = self.input_dir / 'segmented' / shape_code
        
        if not shape_dir.exists():
            print(f"  Warning: Directory not found: {shape_dir}")
            return {}
        
        # Get all images
        image_files = list(shape_dir.glob('*.png'))
        
        if not image_files:
            print(f"  Warning: No images found in {shape_dir}")
            return {}
        
        print(f"  Found {len(image_files)} images")
        
        # Analyze each image
        all_metrics = []
        
        for img_path in tqdm(image_files, desc=f"  Processing {shape_code}"):
            metrics = self.analyze_image_quality(str(img_path))
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate metrics
        aggregated = {
            'shape_code': shape_code,
            'shape_name': self.mapper.get_shape_name(shape_code),
            'image_count': len(all_metrics),
            'foreground_percentage': {
                'mean': np.mean([m['foreground_percentage'] for m in all_metrics]),
                'std': np.std([m['foreground_percentage'] for m in all_metrics]),
                'min': np.min([m['foreground_percentage'] for m in all_metrics]),
                'max': np.max([m['foreground_percentage'] for m in all_metrics])
            },
            'mean_intensity': {
                'mean': np.mean([m['mean_intensity'] for m in all_metrics]),
                'std': np.std([m['mean_intensity'] for m in all_metrics])
            },
            'contrast': {
                'mean': np.mean([m['contrast'] for m in all_metrics]),
                'std': np.std([m['contrast'] for m in all_metrics])
            },
            'edge_density': {
                'mean': np.mean([m['edge_density'] for m in all_metrics]),
                'std': np.std([m['edge_density'] for m in all_metrics])
            },
            'sharpness': {
                'mean': np.mean([m['sharpness'] for m in all_metrics]),
                'std': np.std([m['sharpness'] for m in all_metrics])
            }
        }
        
        # Store individual metrics
        self.shape_metrics[shape_code] = all_metrics
        
        return aggregated
    
    def analyze_all_shapes(self) -> Dict:
        """
        Analyze all shape categories.
        
        Returns:
            Complete analysis results
        """
        print("="*60)
        print("ANALYZING SEGMENTATION RESULTS")
        print("="*60)
        
        results = {
            'analysis_timestamp': str(Path(self.input_dir)),
            'shapes': {}
        }
        
        # Process each shape
        for shape_code in self.mapper.list_all_shapes():
            shape_results = self.analyze_shape_category(shape_code)
            if shape_results:
                results['shapes'][shape_code] = shape_results
        
        # Overall statistics
        if results['shapes']:
            results['overall'] = self._calculate_overall_stats(results['shapes'])
        
        return results
    
    def _calculate_overall_stats(self, shapes_data: Dict) -> Dict:
        """Calculate overall statistics across all shapes."""
        total_images = sum(s['image_count'] for s in shapes_data.values())
        
        all_fg_percentages = []
        all_contrasts = []
        all_sharpness = []
        
        for shape_data in shapes_data.values():
            all_fg_percentages.append(shape_data['foreground_percentage']['mean'])
            all_contrasts.append(shape_data['contrast']['mean'])
            all_sharpness.append(shape_data['sharpness']['mean'])
        
        return {
            'total_images': total_images,
            'shapes_processed': len(shapes_data),
            'avg_foreground_percentage': np.mean(all_fg_percentages),
            'avg_contrast': np.mean(all_contrasts),
            'avg_sharpness': np.mean(all_sharpness)
        }
    
    def generate_comparison_charts(self, results: Dict):
        """
        Generate comparison charts.
        
        Args:
            results: Analysis results
        """
        print("\nGenerating comparison charts...")
        
        if not results.get('shapes'):
            print("No data to visualize")
            return
        
        shapes_data = results['shapes']
        
        # Prepare data
        shape_codes = list(shapes_data.keys())
        shape_names = [shapes_data[c]['shape_name'] for c in shape_codes]
        
        fg_percentages = [shapes_data[c]['foreground_percentage']['mean'] for c in shape_codes]
        contrasts = [shapes_data[c]['contrast']['mean'] for c in shape_codes]
        sharpness = [shapes_data[c]['sharpness']['mean'] for c in shape_codes]
        edge_densities = [shapes_data[c]['edge_density']['mean'] for c in shape_codes]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Segmentation Quality Comparison by Shape', fontsize=16, fontweight='bold')
        
        # Chart 1: Foreground Percentage
        axes[0, 0].bar(shape_names, fg_percentages, color='steelblue')
        axes[0, 0].set_title('Average Foreground Percentage')
        axes[0, 0].set_ylabel('Percentage (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Chart 2: Contrast
        axes[0, 1].bar(shape_names, contrasts, color='coral')
        axes[0, 1].set_title('Average Contrast (Std Dev)')
        axes[0, 1].set_ylabel('Contrast')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Chart 3: Sharpness
        axes[1, 0].bar(shape_names, sharpness, color='seagreen')
        axes[1, 0].set_title('Average Sharpness (Laplacian Variance)')
        axes[1, 0].set_ylabel('Sharpness')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Chart 4: Edge Density
        axes[1, 1].bar(shape_names, edge_densities, color='mediumpurple')
        axes[1, 1].set_title('Average Edge Density')
        axes[1, 1].set_ylabel('Edge Density (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_dir / 'quality_comparison.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"  Chart saved: {chart_path}")
        
        plt.close()
    
    def generate_distribution_plots(self, results: Dict):
        """
        Generate distribution plots for key metrics.
        
        Args:
            results: Analysis results
        """
        print("\nGenerating distribution plots...")
        
        if not self.shape_metrics:
            print("No detailed metrics available")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Metric Distributions Across All Images', fontsize=16, fontweight='bold')
        
        # Collect all metrics
        all_fg = []
        all_contrast = []
        all_sharpness = []
        all_edge = []
        
        for metrics_list in self.shape_metrics.values():
            all_fg.extend([m['foreground_percentage'] for m in metrics_list])
            all_contrast.extend([m['contrast'] for m in metrics_list])
            all_sharpness.extend([m['sharpness'] for m in metrics_list])
            all_edge.extend([m['edge_density'] for m in metrics_list])
        
        # Plot distributions
        axes[0, 0].hist(all_fg, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Foreground Percentage Distribution')
        axes[0, 0].set_xlabel('Percentage (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        axes[0, 1].hist(all_contrast, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Contrast Distribution')
        axes[0, 1].set_xlabel('Contrast')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        axes[1, 0].hist(all_sharpness, bins=50, color='seagreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Sharpness Distribution')
        axes[1, 0].set_xlabel('Sharpness')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        axes[1, 1].hist(all_edge, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Edge Density Distribution')
        axes[1, 1].set_xlabel('Edge Density (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'metric_distributions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved: {plot_path}")
        
        plt.close()
    
    def save_report_json(self, results: Dict):
        """
        Save analysis report as JSON.
        
        Args:
            results: Analysis results
        """
        report_path = self.output_dir / 'analysis_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nJSON report saved: {report_path}")
    
    def save_report_csv(self, results: Dict):
        """
        Save analysis report as CSV.
        
        Args:
            results: Analysis results
        """
        if not results.get('shapes'):
            return
        
        csv_path = self.output_dir / 'analysis_summary.csv'
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'shape_code', 'shape_name', 'image_count',
                'avg_foreground_pct', 'std_foreground_pct',
                'avg_contrast', 'std_contrast',
                'avg_sharpness', 'std_sharpness',
                'avg_edge_density', 'std_edge_density'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for shape_code, data in results['shapes'].items():
                writer.writerow({
                    'shape_code': shape_code,
                    'shape_name': data['shape_name'],
                    'image_count': data['image_count'],
                    'avg_foreground_pct': f"{data['foreground_percentage']['mean']:.2f}",
                    'std_foreground_pct': f"{data['foreground_percentage']['std']:.2f}",
                    'avg_contrast': f"{data['contrast']['mean']:.2f}",
                    'std_contrast': f"{data['contrast']['std']:.2f}",
                    'avg_sharpness': f"{data['sharpness']['mean']:.2f}",
                    'std_sharpness': f"{data['sharpness']['std']:.2f}",
                    'avg_edge_density': f"{data['edge_density']['mean']:.2f}",
                    'std_edge_density': f"{data['edge_density']['std']:.2f}"
                })
        
        print(f"CSV report saved: {csv_path}")
    
    def print_summary(self, results: Dict):
        """
        Print analysis summary to console.
        
        Args:
            results: Analysis results
        """
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        if 'overall' in results:
            overall = results['overall']
            print(f"\nOverall Statistics:")
            print(f"  Total images processed: {overall['total_images']}")
            print(f"  Shapes analyzed: {overall['shapes_processed']}")
            print(f"  Avg foreground: {overall['avg_foreground_percentage']:.2f}%")
            print(f"  Avg contrast: {overall['avg_contrast']:.2f}")
            print(f"  Avg sharpness: {overall['avg_sharpness']:.2f}")
        
        if 'shapes' in results:
            print(f"\nPer-Shape Statistics:")
            print("-"*60)
            
            for shape_code, data in sorted(results['shapes'].items()):
                print(f"\n{data['shape_name']} ({shape_code}):")
                print(f"  Images: {data['image_count']}")
                print(f"  Foreground: {data['foreground_percentage']['mean']:.2f}% (±{data['foreground_percentage']['std']:.2f})")
                print(f"  Contrast: {data['contrast']['mean']:.2f} (±{data['contrast']['std']:.2f})")
                print(f"  Sharpness: {data['sharpness']['mean']:.2f}")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline."""
        # Analyze all shapes
        results = self.analyze_all_shapes()
        
        # Generate visualizations
        self.generate_comparison_charts(results)
        self.generate_distribution_plots(results)
        
        # Save reports
        self.save_report_json(results)
        self.save_report_csv(results)
        
        # Print summary
        self.print_summary(results)
        
        print("\n✓ Analysis complete!")
        print(f"  Results saved to: {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze diamond segmentation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed',
        help='Input directory with processed results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ResultsAnalyzer(
        input_dir=Path(args.input),
        output_dir=Path(args.output)
    )
    
    # Run analysis
    try:
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()