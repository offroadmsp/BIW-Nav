# A Brain-Inspired World Model for Embodied Navigation via Multi-Scale Spatiotemporal Graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Ocean Capsule](https://img.shields.io/badge/Code%20Ocean-Reproducible-blue)](#) This repository contains the official, self-contained implementation of the Brain-Inspired World Model (BIW) described in our *Nature Communications* submission: **"A Brain-Inspired World Model for Embodied Navigation via Multi-Scale Spatiotemporal Graphs"**.

> **📝 Note to Reviewers:** > To facilitate a seamless peer-review process, we have fully integrated the vision transformer backbone (previously maintained as a separate module) directly into this repository under `biw_nav/perception/big_vistrans`. **This repository is completely self-contained; you do not need to clone any additional external repositories.**

---

## 1. System Requirements

We recommend running this code on a Linux machine with a discrete GPU for optimal performance, though the minimal working example (inference) can be executed on lower-tier hardware.

- **OS:** Ubuntu 20.04 LTS (Tested and Recommended)
- **CPU:** Intel Core i7 / AMD Ryzen 7 or equivalent
- **GPU:** NVIDIA GPU with at least 12GB VRAM (e.g., RTX 3060/3090) for inference. (24GB+ recommended for full training).
- **Python:** >= 3.8
- **CUDA:** >= 11.3

---

## 2. Quick Installation (approx. 20-30 minutes)

We have streamlined the dependencies to ensure out-of-the-box reproducibility. 

```bash
# 1. Clone this repository
git clone [https://github.com/offroadmsp/BIW-Nav.git](https://github.com/offroadmsp/BIW-Nav.git)
cd BIW-Nav

# 2. Create and activate a Conda virtual environment
conda create -n biw_nav python=3.8 -y
conda activate biw_nav

# 3. Install required dependencies
pip install -r requirements.txt
```

## 3. Minimal Working Example (Demo for Reviewers)
As per the journal's reproducible research guidelines, we provide a Minimal Working Example (MWE) and sample data to test the BIW model's multi-scale graph construction and inference capabilities instantly.

### Step 1: Download the Sample Weights and Data
Please download the checkpoints.zip and sample_data.zip from our anonymous repository:
🔗 [Link to Large files like checkpoints.zip and sample_data.zip](https://drive.google.com/drive/folders/1BSeoyu8o_eRbBXeP93l81EQt_PJUlkHz?usp=sharing)

Extract them into the root directory of this repository, ensuring the file structure looks like this:

```Plaintext
BIW-Nav/
├── checkpoints/
│   └── biw_pretrained.pt      # Pre-trained model weights
├── data/
│   └── sample_trajectory.npy  # Sample sequence for inference
...
```

### Step 2: Run the Demo Inference
Execute the provided testing script:

```bash
python scripts/demo_inference.py --weights checkpoints/biw_pretrained.pt --data data/sample_trajectory.npy
```

### Expected Output
The script takes approximately 1-2 minutes to execute. It will:

Print a success log to the console indicating the model has been successfully loaded and inference is complete.

Generate a visualization plot saved at results/demo_output.png. This plot demonstrates the constructed multi-scale cognitive graph and the predicted navigation waypoints based on the sample input.

## 4. Reproducing Main Paper Results
For researchers and reviewers wishing to reproduce the complete quantitative results presented in the manuscript (e.g., dynamic environment evaluations, cross-scale generalization), please refer to the detailed instructions in the respective sub-modules:

Core BIW Graph & Rodent Data Evaluation: Check the documentation in biw_nav/core/biw_graph/README.md.

Scale Ablation Studies (Fig 7 & Supp Figs): Navigate to biw_nav/core/scale_ablation/README.md.

Full Dataset Preparation: Instructions for downloading the KITTI Odometry dataset, Tokyo 24/7, and CityScale Simulator maps are detailed in docs/DATASETS.md (or relevant data folders).

## 5. Repository Structure Overview

```Plaintext
BIW-Nav/
├── biw_nav/                     # Main package directory
│   ├── core/                    # Core mechanisms (Grid Cells, Cognitive Graphs, Bayesian loops)
│   ├── perception/              # Vision backbone (Integrated ViNT / Transformer modules)
│   └── ...
├── configs/                     # YAML configuration files for different environments
├── scripts/                     # Executable scripts for training, evaluation, and demo
├── requirements.txt             # Unified Python dependencies
└── README.md                    # This file
```

