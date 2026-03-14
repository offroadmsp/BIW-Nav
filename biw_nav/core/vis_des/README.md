# Visual Feature Extraction: BIW-Representations (BIW Perception Module)

This directory contains the PyTorch implementation of the BIW-Representations architecture, which serves as a visual feature extraction and place recognition baseline within the Brain-Inspired World Model (BIW) framework. This module is responsible for processing high-dimensional sensory inputs into robust spatial descriptors for embodied navigation.

## 1. Validation of the Visual Backbone

To ensure the integrity of the visual perception module, we have validated our implementation against the original NetVLAD performance on the Pittsburgh dataset (Table 1, right column of the original manuscript). The performance metrics (Recall@N) are summarized below:

| Model Configuration | R@1 | R@5 | R@10 |
| :--- | :--- | :--- | :--- |
| Original NetVLAD paper | 84.1 | 94.6 | 95.5 |
| BIW-NetVLAD (AlexNet) | 68.6 | 84.6 | 89.3 |
| BIW-NetVLAD (VGG-16) | 85.2 | 94.8 | 97.0 |

Executing `main.py` in `train` mode with default parameters yields comparable performance to the VGG-16 results shown above. For reproducibility, the pre-trained model checkpoint for this specific run is available for download: 
🔗 [Download Pre-trained VGG-16 NetVLAD Checkpoint](https://drive.google.com/open?id=17luTjZFCX639guSVy00OUtzfTQo4AMF2)

To verify these results using the provided checkpoint, execute the following command:
```bash
python main.py --mode=test --split=val --resume=vgg16_netvlad_checkpoint/
```

---

## 2. Setup and Dependencies

### Dependencies
The dependencies for this module are included in the global `requirements.txt` of the BIW repository. Key libraries utilized include:
* PyTorch (>= 0.4.0)
* Faiss (for efficient similarity search and clustering)
* SciPy ecosystem (`numpy`, `scipy`, `scikit-learn`)
* `h5py` (for dataset storage)
* `tensorboardX` (for training visualization)

### Dataset Preparation
Training and evaluating this module requires the Pittsburgh 250k dataset (images available [here](https://github.com/Relja/netvlad/issues/42)) and the corresponding dataset specifications (.mat files, available [here](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz)).

Please ensure the data is structured as expected by `pittsburgh.py`. The required directory hierarchy is:
* `[Data Root]/`
  * `000` to `010`: Subdirectories containing database images.
  * `queries_real/`: Directory containing subdirectories `000` to `010` with query images.
  * `datasets/`: Directory containing the dataset specification `.mat` files.

*(Note: The root path is currently hardcoded in `pittsburgh.py`. Please update it to reflect your local environment).*

---

## 3. Usage Instructions

The core logic is contained within `main.py`, which operates in three primary modes: `cluster`, `train`, and `test`. 

### Phase 1: Cluster (Initialization)
Prior to training, the NetVLAD layer requires initialization by sampling the dataset to compute `opt.num_clusters` centroids. This step establishes the visual vocabulary and is required for each new network configuration or dataset.
```bash
python main.py --mode=cluster --arch=vgg16 --pooling=netvlad --num_clusters=64
```

### Phase 2: Train
Once the centroids are initialized, the model can be trained. The command-line arguments, TensorBoard logs, and model checkpoints will be automatically saved to the directory specified by `opt.runsPath`.
```bash
python main.py --mode=train --arch=vgg16 --pooling=netvlad --num_clusters=64
```
*For a comprehensive list of configurable hyperparameters, run `python main.py --help`.*

### Phase 3: Test
To evaluate a trained model on the testing split (e.g., Pittsburgh 30k), provide the path to the saved checkpoint. The evaluation script will automatically load the training configurations.
```bash
python main.py --mode=test --resume=runsPath/Nov19_12-00-00_vgg16_netvlad --split=test
```
To evaluate the 'off-the-shelf' pre-trained network without a specific resume directory, simply execute:
```bash
python main.py --mode=test
```

---

## 4. System Configuration Notes

**Handling Large Memory Requirements (Note by Zhen Sun):**
When extracting features or clustering on large-scale datasets (e.g., Pittsburgh 250k), the process may generate substantial temporary cache files, potentially exhausting the default `/tmp` directory. If you encounter "No space left on device" errors, it is recommended to redirect the temporary directory to a storage drive with sufficient capacity prior to running the scripts:

```bash
# Redirect TMPDIR to a high-capacity storage location
export TMPDIR=/media/zhen/Share/temp
cd $TMPDIR

# Execute training/clustering commands here...

# Reset the environment variable after execution
unset TMPDIR
```