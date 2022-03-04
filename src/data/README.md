# Data Directory

This directory contains all dataset files for the diamond segmentation project.

## Structure

- `raw/` - Original diamond images organized by shape type
- `processed/` - Segmented and processed output images

## Raw Data Format

Images are organized in the following structure:
```
raw/
├── Shape_1d_256i/   # Single diamond per image (256 samples)
├── Shape_5d_256i/   # 5 diamonds per image (256 samples)
└── Shape_10d_256i/  # 10 diamonds per image (256 samples)
    ├── AS/          # Asscher cut
    ├── BR/          # Brilliant cut
    ├── CMB/         # Combination cut
    ├── EM/          # Emerald cut
    ├── HS/          # Heart shape
    ├── MQ/          # Marquise cut
    ├── OV/          # Oval cut
    ├── PE/          # Pear cut
    ├── PR/          # Princess cut
    ├── PS/          # Pearshape cut
    ├── RA/          # Radiant cut
    ├── RD/          # Round cut
    ├── SEM/         # Semi cut
    └── TRI/         # Triangle cut
```

Each shape folder contains 256 images in PNG format (256x256 resolution).