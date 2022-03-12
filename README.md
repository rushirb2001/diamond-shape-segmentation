# Diamond Shape Segmentation

A computer vision pipeline for automated diamond image segmentation using GrabCut algorithm with CLAHE preprocessing. This project segments diamonds from background images across 14 different shape categories.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## About This Project

This project was developed as part of the **MiNeD Hackathon** hosted by **Nirma University**. The system implements a comprehensive automated segmentation pipeline for diamond images, processing over 57,000 images across multiple dataset variants.

### Team Members

- **Rushir Bhavsar** - Lead Developer
- **Harshil Sanghvi** - Core Developer  
- **Ruju Shah** - Core Developer
- **Vrunda Shah** - Core Developer
- **Khushi Patel** - Core Developer

---

## Demo Results

### Triple-Split Video Demonstrations

Our segmentation pipeline demonstrates robust performance across different diamond shapes. Each video shows the complete processing pipeline: **Original → CLAHE Enhanced → Segmented**.

#### Asscher Cut
[![Asscher Demo](src/data/output/videos/triple_split_AS_Asscher.gif)](src/data/output/videos/triple_split_AS_Asscher.gif)

#### Brilliant Cut
[![Brilliant Demo](src/data/output/videos/triple_split_BR_Brilliant.gif)](src/data/output/videos/triple_split_BR_Brilliant.gif)


---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [Pipeline Architecture](#pipeline-architecture)
- [Video Generation](#video-generation)
- [Results Analysis](#results-analysis)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project implements an automated segmentation pipeline for diamond images using OpenCV's GrabCut algorithm enhanced with CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing. The system processes three dataset variants:

- **Shape_1d_256i**: Single diamond variation (256 images per shape)
- **Shape_5d_256i**: Five diamond variations (1,280 images per shape)
- **Shape_10d_256i**: Ten diamond variations (2,560 images per shape)

**Total Dataset**: 57,344 images across 14 diamond shape categories

---

## Features

### Core Capabilities

| Category | Features |
|----------|----------|
| **Segmentation** | GrabCut-based background removal<br>CLAHE preprocessing for enhanced contrast<br>Morphological operations for mask refinement<br>Contour detection and bounding box annotation<br>Multi-threaded batch processing<br>Comprehensive error handling and logging |
| **Video Generation** | Triple-split videos (Before \| CLAHE \| After)<br>Five-split comparison videos (5 variations side-by-side)<br>Mask evolution animations<br>Pipeline step-by-step demonstrations<br>Dataset coverage visualizations |
| **Analysis & Reporting** | Quality metrics calculation<br>Statistical analysis by shape category<br>Automated report generation (JSON/CSV)<br>Comparison charts and distribution plots<br>Performance profiling and benchmarking |
| **Developer Tools** | Interactive processing mode<br>Command-line interface (CLI)<br>Jupyter notebooks for exploration<br>Performance profiling utilities<br>Structured logging with rotation<br>Configurable parameters via YAML |

### Processing Pipeline Features
```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (256x256)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  CLAHE Enhancement  │  Clip Limit: 2.5  │  Grid: 8×8        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  GrabCut Init       │  Border-based mask (20px margin)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  GrabCut Iterations │  5× iterations with mask refinement    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Post-processing    │  Morphological operations & cleanup    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Output             │  Binary mask + Segmented image         │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

### Diamond Shape Categories

| Code | Shape Name | Description |
|------|------------|-------------|
| AS   | Asscher    | Square step cut with cropped corners |
| BR   | Brilliant  | Round brilliant cut |
| CMB  | Combination| Mixed faceting styles |
| EM   | Emerald    | Rectangular step cut |
| HS   | Heart      | Heart-shaped cut |
| MQ   | Marquise   | Elongated pointed ends |
| OV   | Oval       | Elliptical shape |
| PE   | Pear       | Teardrop shape |
| PR   | Princess   | Square brilliant cut |
| PS   | Pearshape  | Pear-shaped variation |
| RA   | Radiant    | Rectangular brilliant cut |
| RD   | Round      | Traditional round cut |
| SEM  | Semi       | Semi-faceted style |
| TRI  | Triangle   | Triangular cut |

### Dataset Structure
```
data/
├── raw/
│   ├── Shape_1d_256i/
│   │   ├── AS/  (256 images)
│   │   ├── BR/  (256 images)
│   │   └── ...
│   ├── Shape_5d_256i/
│   │   ├── AS/  (1,280 images)
│   │   ├── BR/  (1,280 images)
│   │   └── ...
│   └── Shape_10d_256i/
│       ├── AS/  (2,560 images)
│       ├── BR/  (2,560 images)
│       └── ...
└── processed/
    ├── segmented/
    ├── masks/
    ├── annotated/
    └── stats/
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/diamond-shape-segmentation.git
cd diamond-shape-segmentation
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -m src.main info --data-path data/raw --variant Shape_1d_256i
```

---

## Quick Start

### Process Single Shape
```bash
# Process Asscher shape (AS) with default settings
python -m src.main process --shape AS --variant Shape_1d_256i
```

### Process All Shapes
```bash
# Process all 14 shapes with custom iterations
python -m src.main process --all --variant Shape_1d_256i --iterations 5
```

### Generate Demo Videos
```bash
# Generate all demo videos for selected shapes
python scripts/generate_demo_videos.py --shapes AS BR EM MQ OV
```

### Run Interactive Mode
```bash
# Launch interactive processing interface
python -m src.main interactive
```

### Analyze Results
```bash
# Analyze segmentation quality
python scripts/analyze_results.py --input data/processed --output analysis/
```

---

## Usage

### Command Line Interface

#### Dataset Information
```bash
# Show dataset info and validate structure
python -m src.main info --data-path data/raw --variant Shape_1d_256i --validate
```

#### Process Images
```bash
# Basic processing
python -m src.main process --shape AS --variant Shape_1d_256i

# With annotations and masks
python -m src.main process --shape BR --annotations --save-masks

# Process all shapes with profiling
python -m src.main process --all --profile --iterations 5

# Debug mode with verbose logging
python -m src.main process --shape EM --debug --log-dir logs/
```

#### Generate Videos
```bash
# Triple-split video (Before | CLAHE | After)
python -m src.main video triple-split --shape AS --output videos/

# Five-split video (5 variations)
python -m src.main video five-split --shape BR --output videos/

# Generate all demo videos
python -m src.main video generate-demos --shapes AS BR EM MQ OV
```

#### Performance Benchmark
```bash
# Run benchmark on 50 images
python -m src.main benchmark --num-images 50 --iterations 5
```

### Python API

#### Basic Segmentation
```python
from src.data.loader import DiamondDataLoader
from src.pipe.processor import DiamondProcessor

# Initialize loader
loader = DiamondDataLoader('data/raw', 'Shape_1d_256i')

# Create processor
processor = DiamondProcessor(
    data_loader=loader,
    output_dir='data/processed',
    iterations=5
)

# Process a shape category
results = processor.process_shape_category(
    shape_id=1,  # Asscher
    max_images=100
)

print(f"Processed: {results['processed']}")
print(f"Failed: {results['failed']}")
```

#### Video Generation
```python
from src.pipe.video_comparison import VideoComparator

# Create triple-split video
comparator = VideoComparator('output/video.avi', fps=15)
video_path = comparator.create_triple_split(
    image_paths=['img1.png', 'img2.png', ...],
    iterations=5,
    add_labels=True
)
```

#### Performance Profiling
```python
from src.utils.profiling import PerformanceProfiler

profiler = PerformanceProfiler()

# Measure operation
with profiler.measure('segmentation', metadata={'iterations': 5}):
    # Your segmentation code here
    pass

# Print summary
profiler.print_summary()

# Save metrics
profiler.save_metrics('metrics.json', format='json')
```

### Jupyter Notebooks

The project includes three interactive notebooks:

1. **`01_data_exploration.ipynb`**: Dataset visualization and exploration
2. **`02_diamond_segmentation.ipynb`**: Complete segmentation pipeline demo
3. **`03_video_generation.ipynb`**: Video creation and comparisons
```bash
# Launch Jupyter
jupyter notebook notebooks/
```

---

## Pipeline Architecture

### Segmentation Pipeline
```
Input Image (256x256)
        ↓
1. CLAHE Enhancement
   - Clip Limit: 2.5
   - Grid Size: 8×8
        ↓
2. GrabCut Initialization
   - Border-based mask (20px margin)
   - Probable foreground: center region
   - Probable background: border region
        ↓
3. GrabCut Iterations (5×)
   - Gaussian Mixture Models (GMM)
   - Graph Cut optimization
   - Mask refinement
        ↓
4. Post-processing
   - Morphological operations
   - Noise reduction
        ↓
5. Final Segmentation
   - Binary mask
   - Segmented image
   - Optional annotations
```

### Processing Stages

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| Load | Raw image | BGR image | Read from disk |
| Preprocess | BGR image | Enhanced BGR | Improve contrast |
| Initialize | Enhanced image | Initial mask | Setup GrabCut |
| Segment | Enhanced + mask | Binary mask | Remove background |
| Refine | Binary mask | Clean mask | Morphological ops |
| Annotate | Segmented image | Annotated image | Add contours/boxes |
| Save | Final image | Disk file | Store results |

---

## Video Generation

### Triple-Split Videos

Shows the complete pipeline: Original → CLAHE Enhanced → Segmented
```bash
python -m src.main video triple-split --shape AS --fps 15 --output videos/
```

**Features:**
- Side-by-side comparison
- Stage labels
- Progress indicators
- 256 frames (one per image)

### Five-Split Videos

Demonstrates algorithm robustness across 5 variations of the same shape
```bash
python -m src.main video five-split --shape BR --fps 15 --output videos/
```

**Features:**
- 5 variations in parallel
- Synchronized playback
- Variation labels
- Grid layout (1×5)

### Animation Types

1. **Mask Evolution**: Shows GrabCut refinement over iterations
2. **Pipeline Steps**: Step-by-step algorithm demonstration
3. **Coverage Map**: Dataset processing visualization

---

## Results Analysis

### Quality Metrics

The analysis script calculates the following metrics:

- **Foreground Percentage**: Ratio of diamond to total pixels
- **Mean Intensity**: Average brightness of segmented region
- **Contrast**: Standard deviation of pixel intensities
- **Edge Density**: Percentage of edge pixels
- **Sharpness**: Laplacian variance

### Generate Analysis Report
```bash
python scripts/analyze_results.py --input data/processed --output analysis/
```

**Outputs:**
- `analysis_report.json`: Detailed JSON report
- `analysis_summary.csv`: Summary table
- `quality_comparison.png`: Bar charts by shape
- `metric_distributions.png`: Histograms

### Example Results
```
Overall Statistics:
  Total images processed: 3,584
  Shapes analyzed: 14
  Avg foreground: 68.45%
  Avg contrast: 52.31
  Avg sharpness: 145.67
```

---

## Performance

### Benchmarks

Hardware: Intel Core i7, 16GB RAM, No GPU

| Operation | Time (ms) | Throughput |
|-----------|-----------|------------|
| Load Image | 5.2 | 192 img/s |
| CLAHE Preprocess | 12.4 | 81 img/s |
| GrabCut (5 iter) | 245.8 | 4.1 img/s |
| Total Pipeline | 268.5 | 3.7 img/s |

### Optimization Tips

1. **Reduce Iterations**: Use 3-4 iterations for faster processing
2. **Batch Processing**: Process multiple images in parallel
3. **Skip Annotations**: Disable when not needed
4. **Lower Resolution**: Resize images if acceptable
```bash
# Fast processing (minimal quality loss)
python -m src.main process --all --iterations 3 --max-images 100
```

---

## Project Structure
```
diamond-shape-segmentation/
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── config.py               # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # Dataset loading
│   ├── pipe/
│   │   ├── __init__.py
│   │   ├── processor.py        # Batch processing
│   │   ├── video_creator.py   # Video generation
│   │   ├── video_comparison.py # Comparison videos
│   │   └── animation.py        # Animations
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py       # File operations
│       ├── segmentation.py     # Core algorithms
│       ├── visualization.py    # Drawing utilities
│       ├── profiling.py        # Performance tools
│       └── logging_config.py   # Logging setup
├── scripts/
│   ├── __init__.py
│   ├── generate_demo_videos.py # Demo generation
│   ├── analyze_results.py      # Results analysis
│   └── quick_demo.sh           # Quick demo script
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_diamond_segmentation.ipynb
│   └── 03_video_generation.ipynb
├── data/
│   ├── raw/                    # Input datasets
│   └── processed/              # Output results
├── tests/
│   └── (unit tests)
├── config.example.yaml         # Configuration template
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- OpenCV community for the GrabCut implementation
- **Nirma University** for hosting the MiNeD Hackathon
- Diamond dataset providers
- Academic advisors and mentors
- Open-source contributors

---

## References

1. Rother, C., Kolmogorov, V., & Blake, A. (2004). "GrabCut: Interactive foreground extraction using iterated graph cuts." *ACM Transactions on Graphics*, 23(3), 309-314.

2. Pizer, S. M., et al. (1987). "Adaptive histogram equalization and its variations." *Computer Vision, Graphics, and Image Processing*, 39(3), 355-368.

3. Bradski, G. (2000). "The OpenCV Library." *Dr. Dobb's Journal of Software Tools*.

---

## Support

For questions, issues, or suggestions:

- Email: rushirbhavsar@gmail.com
- Issues: [GitHub Issues](https://github.com/yourusername/diamond-shape-segmentation/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/diamond-shape-segmentation/discussions)

---

**Developed by Team MiNeD at Nirma University**

*Last Updated: March 11, 2022*