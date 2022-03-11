"""
Main entry point for diamond segmentation pipeline.
Provides command-line interface for processing operations.
"""

import argparse
import sys
from pathlib import Path

from src.config import Config, ProcessingConfig, get_config
from src.data.loader import DiamondDataLoader, DiamondShapeMapper
from src.pipe.processor import DiamondProcessor, AdvancedDiamondProcessor
from src.pipe.video_creator import create_shape_video, create_all_shapes_video


def setup_argparse() -> argparse.ArgumentParser:
    """
    Setup command-line argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Diamond Shape Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single shape
  python -m src.main process --shape AS --variant Shape_1d_256i
  
  # Process all shapes with custom iterations
  python -m src.main process --all --iterations 5
  
  # Generate video for specific shape
  python -m src.main video --shape BR --output videos/
  
  # Interactive mode
  python -m src.main interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process diamond images')
    process_parser.add_argument(
        '--shape',
        type=str,
        choices=['AS', 'BR', 'CMB', 'EM', 'HS', 'MQ', 'OV', 'PE', 'PR', 'PS', 'RA', 'RD', 'SEM', 'TRI'],
        help='Shape code to process'
    )
    process_parser.add_argument(
        '--all',
        action='store_true',
        help='Process all shapes'
    )
    process_parser.add_argument(
        '--variant',
        type=str,
        default='Shape_1d_256i',
        choices=['Shape_1d_256i', 'Shape_5d_256i', 'Shape_10d_256i'],
        help='Dataset variant to use'
    )
    process_parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw',
        help='Path to raw data directory'
    )
    process_parser.add_argument(
        '--output-path',
        type=str,
        default='data/processed',
        help='Path to output directory'
    )
    process_parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of GrabCut iterations'
    )
    process_parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum images to process (None for all)'
    )
    process_parser.add_argument(
        '--annotations',
        action='store_true',
        help='Add contour annotations to output'
    )
    process_parser.add_argument(
        '--save-masks',
        action='store_true',
        help='Save binary masks'
    )
    
    # Video command
    video_parser = subparsers.add_parser('video', help='Generate videos')
    video_parser.add_argument(
        '--shape',
        type=str,
        help='Shape code for video generation'
    )
    video_parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed',
        help='Directory with processed images'
    )
    video_parser.add_argument(
        '--output',
        type=str,
        default='videos/',
        help='Output directory for videos'
    )
    video_parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='Frames per second'
    )
    video_parser.add_argument(
        '--all-shapes',
        action='store_true',
        help='Create video with all shapes'
    )
    
    # Interactive command
    subparsers.add_parser('interactive', help='Run in interactive mode')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show dataset information')
    info_parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw',
        help='Path to raw data directory'
    )
    info_parser.add_argument(
        '--variant',
        type=str,
        default='Shape_1d_256i',
        help='Dataset variant'
    )
    
    return parser


def cmd_process(args):
    """
    Handle process command.
    
    Args:
        args: Parsed command-line arguments
    """
    print("="*60)
    print("DIAMOND SEGMENTATION PROCESSING")
    print("="*60)
    
    # Initialize data loader
    loader = DiamondDataLoader(args.data_path, args.variant)
    
    # Initialize processor
    processor = AdvancedDiamondProcessor(
        data_loader=loader,
        output_dir=args.output_path,
        iterations=args.iterations,
        add_annotations=args.annotations,
        save_masks=args.save_masks,
        save_stats=True
    )
    
    # Process based on options
    if args.all:
        print(f"\nProcessing all shapes from {args.variant}")
        processor.process_all_shapes(max_images_per_shape=args.max_images)
    elif args.shape:
        mapper = DiamondShapeMapper()
        shape_id = mapper.get_shape_id(args.shape)
        if shape_id:
            print(f"\nProcessing shape: {args.shape}")
            processor.process_shape_category(shape_id, max_images=args.max_images)
        else:
            print(f"Error: Invalid shape code: {args.shape}")
            sys.exit(1)
    else:
        print("Error: Must specify --shape or --all")
        sys.exit(1)


def cmd_video(args):
    """
    Handle video command.
    
    Args:
        args: Parsed command-line arguments
    """
    print("="*60)
    print("VIDEO GENERATION")
    print("="*60)
    
    if args.all_shapes:
        print("\nGenerating video with all shapes...")
        output_path = Path(args.output) / 'all_shapes.avi'
        create_all_shapes_video(args.input_dir, str(output_path), fps=args.fps)
        print(f"Video saved: {output_path}")
    elif args.shape:
        print(f"\nGenerating video for shape: {args.shape}")
        output_path = create_shape_video(
            args.shape,
            args.input_dir,
            args.output,
            fps=args.fps
        )
        print(f"Video saved: {output_path}")
    else:
        print("Error: Must specify --shape or --all-shapes")
        sys.exit(1)


def cmd_interactive(args):
    """
    Handle interactive command.
    
    Args:
        args: Parsed command-line arguments
    """
    from src.pipe.processor import run_interactive_mode
    
    # Use default paths
    data_path = 'data/raw'
    output_path = 'data/processed'
    
    run_interactive_mode(data_path, output_path)


def cmd_info(args):
    """
    Handle info command.
    
    Args:
        args: Parsed command-line arguments
    """
    print("="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    try:
        loader = DiamondDataLoader(args.data_path, args.variant)
        dataset_info = loader.get_dataset_info()
        
        print(f"\nVariant: {dataset_info['variant']}")
        print(f"Dataset path: {dataset_info['dataset_path']}")
        print(f"Total images: {dataset_info['total_images']}")
        print(f"\nShape categories: {len(dataset_info['shapes'])}")
        
        print("\n" + "-"*60)
        print(f"{'Code':<6} {'Name':<20} {'Images':<10}")
        print("-"*60)
        
        for code, info in sorted(dataset_info['shapes'].items()):
            print(f"{code:<6} {info['name']:<20} {info['image_count']:<10}")
        
        print("="*60)
        
        # Validate dataset
        is_valid, issues = loader.validate_dataset()
        
        if is_valid:
            print("\n✓ Dataset validation: PASSED")
        else:
            print("\n✗ Dataset validation: FAILED")
            print("\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command handler
    if args.command == 'process':
        cmd_process(args)
    elif args.command == 'video':
        cmd_video(args)
    elif args.command == 'interactive':
        cmd_interactive(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()