EnDecodingUNet: Deep Ghost Imaging with Learnable Row-Col Patterns

This repository contains a PyTorch implementation of EnDecodingUNet, a deep learning framework designed for computational ghost imaging. It features a custom RowColPatternGenerator that simulates DMD binary encoding using learnable, orthogonal patterns and a U-Net enhancer for high-quality image reconstruction.

Key Features:

Learnable Row-Col Patterns: Generates binary patterns using row and column vectors, optimized for hardware constraints like VCSEL arrays or DMDs.

DGI Integration: Incorporates a core Differential Ghost Imaging (DGI) layer within the neural network forward pass.

Flexible Initialization: Supports multiple pattern initialization modes, including Zigzag Hadamard (Gray-code ordered), random Hadamard, and pure random.

Deep Enhancement: Uses a 4-level U-Net architecture with skip connections to refine reconstruction details from noisy measurements.

Hardware & Environment:

Python: 3.9

PyTorch: 2.3.1

CUDA: 12.1

Recommended GPU: RTX 3060-12G or better

Project StructureUNET:
model_hjj1.py: Contains the model architecture (EnDecodingUNet, RowColPatternGenerator, and UNet).
UNETtraincode1.py: Main training script with integrated logging, early stopping, and metric visualization.

Usage:
To start training with default parameters (512 patterns):python UNETtraincode1.py --C 512 --select_mode 4
Arguments:
--C: Number of patterns (default: 512).
--select_mode: Pattern initialization mode (1: Zigzag, 2: Random Hadamard, 3: Even Spacing, 4: Pure Random).

Results:
The script automatically creates a Results folder containing:
Model: Saved .pth weights.
Pattern: Optimized patterns in .pth and .mat formats.
Log: Training logs and performance curves (SSIM/PSNR).
