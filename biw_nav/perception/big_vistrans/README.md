# The Downstream Navigation Models: BIW



This directory contains the adapted implementation of the **BIW Navigation Models**, serving as the visual feature extraction and action generation backbone for the Brain-Inspired World Model (BIW). General Navigation Models are general-purpose goal-conditioned visual navigation policies trained on diverse, cross-embodiment training data, providing robust latent representations that BIW leverages for multi-scale cognitive mapping.



## Overview

This sub-module focuses purely on the training and processing aspects of the vision models. Code execution scripts related to ROS deployment on specific robots (like LoCoBot) have been omitted in this repository to focus on the core neural network architectures integrated into BIW.

- `train.py`: The main training script to train or fine-tune the ViNT/GNM models.
- `vint_train/models/`: Contains the PyTorch model definitions for BIW.
- `process_*.py`: Utility scripts for processing raw trajectories (e.g., from rosbags or HDF5s) into the required training data format.

## Training the Vision Models

### Setup

Ensure you have created the Conda environment as specified in the main BIW repository's `requirements.txt`. The necessary dependencies for training ViNT/GNM are included there.

If the `diffusion_policy` package is required for NoMaD, install it via:
```bash
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

### Data Processing

To train the models, datasets must be processed into a specific structure. You can use the provided scripts like `process_bags.py` (for ROS bags) or `process_recon.py` (for RECON HDF5s).

The required directory structure after processing is:

```text
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   ...
```

Each `*.jpg` is a forward-facing RGB observation. The `traj_data.pkl` is a dictionary containing odometry:
- `"position"`: `np.ndarray` `[T, 2]` of xy-coordinates.
- `"yaw"`: `np.ndarray` `[T,]` of yaw angles.

After generating trajectories, create the data splits by running `data_split.py` to organize the data into `train/` and `test/` splits.

### Training Configuration

To train the vision model, execute:
```bash
python train.py -c config/<config_file>.yaml
```

Configuration files (e.g., `vint.yaml`, `gnm.yaml`) are located in the `config/` directory. You must update these files to point to your processed dataset paths.

Example dataset configuration in the `.yaml` file:
```yaml
datasets:
  <dataset_name>:
    data_folder: <path_to_the_dataset>
    train: data/data_splits/<dataset_name>/train/ 
    test: data/data_splits/<dataset_name>/test/ 
    end_slack: 0 
    goals_per_obs: 1 
    negative_mining: True 
```

### Checkpoints
To resume training or fine-tune from existing checkpoints (such as the official ViNT weights), add the following to your config file:
```yaml
load_run: <project_name>/<log_run_name>
```
The code expects the checkpoint to be located at `logs/<project_name>/<log_run_name>/latest.pth`.

