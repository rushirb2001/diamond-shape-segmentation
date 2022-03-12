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
from src.pipe.video_comparison import create_triple_split_video, create_annotated_comparison
from src.utils.logging_config import (
    setup_logging, 
    get_logger, 
    log_system_info,
    log_processing_start,
    log_processing_end,
    enable_debug_mode
)
from src.utils.profiling import PerformanceProfiler, BatchProfiler


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
  python -m src.main process --all --iterations 5 --profile
  
  # Generate triple-split video for specific shape
  python -m src.main video triple-split --shape BR --output videos/
  
  # Generate five-split video showing variations
  python -m src.main video five-split --shape AS --output videos/
  
  # Generate all demo videos
  python -m src.main video generate-demos --shapes AS BR EM MQ OV
  
  # Interactive mode
  python -m src.main interactive
  
  # Show dataset info with validation
  python -m src.main info --validate
        """
    )
    
    # Global options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for log files'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
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
    process_parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for progress reporting'
    )
    
    # Video command
    video_parser = subparsers.add_parser('video', help='Generate videos')
    video_subparsers = video_parser.add_subparsers(dest='video_command', help='Video generation command')
    
    # Triple-split video
    triple_parser = video_subparsers.add_parser('triple-split', help='Generate triple-split video')
    triple_parser.add_argument('--shape', type=str, required=True, help='Shape code')
    triple_parser.add_argument('--data-path', type=str, default='data/raw', help='Data directory')
    triple_parser.add_argument('--output', type=str, default='videos/', help='Output directory')
    triple_parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    triple_parser.add_argument('--iterations', type=int, default=5, help='GrabCut iterations')
    triple_parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames')
    
    # Five-split video
    five_parser = video_subparsers.add_parser('five-split', help='Generate five-split video')
    five_parser.add_argument('--shape', type=str, required=True, help='Shape code')
    five_parser.add_argument('--data-path', type=str, default='data/raw', help='Data directory')
    five_parser.add_argument('--output', type=str, default='videos/', help='Output directory')
    five_parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    five_parser.add_argument('--iterations', type=int, default=5, help='GrabCut iterations')
    
    # Generate all demos
    demos_parser = video_subparsers.add_parser('generate-demos', help='Generate all demo videos')
    demos_parser.add_argument('--shapes', type=str, nargs='+', default=['AS', 'BR', 'EM', 'MQ', 'OV'],
                             help='Shapes to process')
    demos_parser.add_argument('--data-path', type=str, default='data/raw', help='Data directory')
    demos_parser.add_argument('--output', type=str, default='demo_videos/', help='Output directory')
    demos_parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    demos_parser.add_argument('--iterations', type=int, default=5, help='GrabCut iterations')
    
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
    info_parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate dataset structure'
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--data-path', type=str, default='data/raw', help='Data directory')
    benchmark_parser.add_argument('--num-images', type=int, default=10, help='Number of images to benchmark')
    benchmark_parser.add_argument('--iterations', type=int, default=5, help='GrabCut iterations')
    
    return parser


def cmd_process(args):
    """
    Handle process command.
    
    Args:
        args: Parsed command-line arguments
    """
    logger = get_logger()
    
    log_processing_start(
        "Diamond Segmentation Processing",
        variant=args.variant,
        iterations=args.iterations,
        annotations=args.annotations,
        save_masks=args.save_masks
    )
    
    # Initialize profiler if requested
    profiler = BatchProfiler() if args.profile else None
    if profiler:
        profiler.start_batch()
    
    try:
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
            logger.info(f"Processing all shapes from {args.variant}")
            results = processor.process_all_shapes(max_images_per_shape=args.max_images)
        elif args.shape:
            mapper = DiamondShapeMapper()
            shape_id = mapper.get_shape_id(args.shape)
            if shape_id:
                logger.info(f"Processing shape: {args.shape}")
                results = processor.process_shape_category(shape_id, max_images=args.max_images)
            else:
                logger.error(f"Invalid shape code: {args.shape}")
                sys.exit(1)
        else:
            logger.error("Must specify --shape or --all")
            sys.exit(1)
        
        # Print profiling results
        if profiler:
            profiler.print_batch_summary()
        
        log_processing_end(
            "Diamond Segmentation Processing",
            success=True,
            processed=results.get('processed', 0),
            failed=results.get('failed', 0)
        )
        
    except Exception as e:
        from src.utils.logging_config import log_error
        log_error(e, operation="Process Command", variant=args.variant)
        sys.exit(1)


def cmd_video(args):
    """
    Handle video command.
    
    Args:
        args: Parsed command-line arguments
    """
    logger = get_logger()
    
    if not args.video_command:
        logger.error("Must specify video subcommand (triple-split, five-split, generate-demos)")
        sys.exit(1)
    
    try:
        if args.video_command == 'triple-split':
            cmd_video_triple_split(args)
        elif args.video_command == 'five-split':
            cmd_video_five_split(args)
        elif args.video_command == 'generate-demos':
            cmd_video_generate_demos(args)
        
    except Exception as e:
        from src.utils.logging_config import log_error
        log_error(e, operation="Video Generation")
        sys.exit(1)


def cmd_video_triple_split(args):
    """Generate triple-split video."""
    logger = get_logger()
    
    log_processing_start(
        "Triple-Split Video Generation",
        shape=args.shape,
        fps=args.fps,
        iterations=args.iterations
    )
    
    # Load data
    loader = DiamondDataLoader(args.data_path, 'Shape_1d_256i')
    mapper = DiamondShapeMapper()
    
    shape_id = mapper.get_shape_id(args.shape)
    shape_name = mapper.get_shape_name(args.shape)
    
    # Get images
    images = loader.list_images(shape_id)
    if args.max_frames:
        images = images[:args.max_frames]
    
    image_paths = [loader.get_image_path(shape_id, img) for img in images]
    
    # Create output directory
    from src.utils.file_utils import ensure_dir
    ensure_dir(args.output)
    
    output_path = Path(args.output) / f'triple_split_{args.shape}_{shape_name}.avi'
    
    # Generate video
    from src.pipe.video_comparison import VideoComparator
    comparator = VideoComparator(output_path, fps=args.fps)
    video_path = comparator.create_triple_split(image_paths, iterations=args.iterations)
    
    log_processing_end(
        "Triple-Split Video Generation",
        success=True,
        output=video_path,
        frames=len(images)
    )


def cmd_video_five_split(args):
    """Generate five-split video."""
    logger = get_logger()
    
    log_processing_start(
        "Five-Split Video Generation",
        shape=args.shape,
        fps=args.fps
    )
    
    from scripts.generate_demo_videos import DemoVideoGenerator
    
    generator = DemoVideoGenerator(
        data_path=args.data_path,
        output_dir=args.output,
        fps=args.fps,
        iterations=args.iterations
    )
    
    video_path = generator.generate_five_split_video(args.shape)
    
    log_processing_end(
        "Five-Split Video Generation",
        success=True,
        output=video_path
    )


def cmd_video_generate_demos(args):
    """Generate all demo videos."""
    logger = get_logger()
    
    log_processing_start(
        "Demo Video Generation",
        shapes=', '.join(args.shapes),
        fps=args.fps
    )
    
    from scripts.generate_demo_videos import DemoVideoGenerator
    
    generator = DemoVideoGenerator(
        data_path=args.data_path,
        output_dir=args.output,
        fps=args.fps,
        iterations=args.iterations
    )
    
    results = generator.generate_all_demos(args.shapes)
    
    total_success = sum(1 for p in results['triple_split'].values() if p) + \
                   sum(1 for p in results['five_split'].values() if p)
    
    log_processing_end(
        "Demo Video Generation",
        success=True,
        videos_generated=total_success
    )


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
    logger = get_logger()
    
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
        
        # Validate dataset if requested
        if args.validate:
            logger.info("Validating dataset structure...")
            is_valid, issues = loader.validate_dataset()
            
            if is_valid:
                print("\n✓ Dataset validation: PASSED")
                logger.info("Dataset validation passed")
            else:
                print("\n✗ Dataset validation: FAILED")
                print("\nIssues found:")
                for issue in issues:
                    print(f"  - {issue}")
                    logger.warning(f"Validation issue: {issue}")
        
    except Exception as e:
        logger.error(f"Error loading dataset info: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_benchmark(args):
    """Run performance benchmarks."""
    logger = get_logger()
    
    log_processing_start(
        "Performance Benchmark",
        num_images=args.num_images,
        iterations=args.iterations
    )
    
    from src.utils.segmentation import preprocess_for_segmentation, remove_background
    import cv2
    import time
    
    # Load test images
    loader = DiamondDataLoader(args.data_path, 'Shape_1d_256i')
    images = loader.list_images(1)[:args.num_images]
    
    profiler = PerformanceProfiler()
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Images: {len(images)}")
    print(f"Iterations: {args.iterations}")
    print()
    
    # Benchmark
    for img_name in images:
        img_path = loader.get_image_path(1, img_name)
        
        with profiler.measure('load_image'):
            img = cv2.imread(img_path)
        
        with profiler.measure('preprocess'):
            _, enhanced = preprocess_for_segmentation(img)
        
        with profiler.measure('segmentation'):
            _, _ = remove_background(img, enhanced, args.iterations)
    
    # Print results
    profiler.print_summary()
    
    log_processing_end(
        "Performance Benchmark",
        success=True,
        images_processed=len(images)
    )


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    setup_logging(log_dir=args.log_dir, log_level=log_level)
    
    logger = get_logger()
    
    if args.debug:
        enable_debug_mode()
        logger.debug("Debug mode enabled")
    
    # Log system info
    log_system_info()
    
    # Route to appropriate command handler
    try:
        if args.command == 'process':
            cmd_process(args)
        elif args.command == 'video':
            cmd_video(args)
        elif args.command == 'interactive':
            cmd_interactive(args)
        elif args.command == 'info':
            cmd_info(args)
        elif args.command == 'benchmark':
            cmd_benchmark(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        print("\n\nOperation cancelled.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unhandled exception in main")
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()