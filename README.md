# VITMMA19-HAZI

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Jókay Benedek
- **Aiming for +1 Mark**: No

### Solution Description

This project tackles an image classification task focused on ankle alignment assessment. The model distinguishes between three distinct ankle positions: Pronation (inward lean), Neutral (aligned), and Supination (outward lean). These categories represent different biomechanical states that can be visually identified by analyzing the ankle's angular position relative to the leg.

The implementation follows a four-stage pipeline executed through a containerized Docker environment. Data preprocessing is handled in `src/01-data-preprocessing.py`, where the system parses Label Studio JSON exports and performs image validation using PIL to filter out corrupted files before creating stratified dataset splits. The preprocessing stage works with 321 manually annotated images collected from 13 different sources, applying a 70/15/15 split for training, validation, and testing while maintaining class distribution balance.

Training occurs in the second stage via `src/02-training.py`, employing a custom-designed CNN architecture called AnkleNet. This network features four convolutional blocks that progressively expand feature channels from 48 to 256, with batch normalization and dropout layers integrated throughout for regularization. The training process leverages Adam optimization combined with a ReduceLROnPlateau scheduler that dynamically adjusts learning rates based on validation performance. Data augmentation strategies including geometric transformations (crops, flips, rotations) and color space manipulations enhance model generalization. An early stopping mechanism monitors validation accuracy to halt training when performance plateaus, preventing overfitting while using standard cross-entropy loss for multi-class classification.

Model assessment is conducted in `src/03-evaluation.py`, which processes the test set to generate comprehensive performance analytics. This stage produces confusion matrices, per-class precision/recall/F1-score calculations, visualization plots for metric analysis, and structured JSON reports containing all evaluation results.

The pipeline concludes with `src/04-inference.py`, demonstrating practical model deployment by running predictions on randomly selected test images and displaying classification results with associated confidence levels. All configurable parameters spanning learning rates, batch sizes, augmentation intensities, and network dimensions are consolidated in `src/config.py` for centralized hyperparameter management.

The complete solution operates within a PyTorch 2.5.1 environment with CUDA 12.4 support, processing standardized 224×224 pixel inputs and delivering GPU-accelerated performance that exceeds baseline majority-class prediction accuracy.

### Data Preparation

Follow these steps for the data preparation part of the process:

1. Download the `anklealign.zip` from the root directory of the SharePoint.
2. Extract the zip into the `.\data\` folder of this project (mount this data folder to Docker in the later part of the instructions).
3. Manually correct the following inconsistencies in the data:
    - In `H51B9J.json`, remove the `_H51B9J` suffix from all image filenames (use Ctrl + F → Replace All).
    - In the `NC1O2T` folder, extract images from subfolders to the main folder and delete the empty subfolders.
    - In `project-2-at-2025-10-16-02-08-8ee4fdfa.json` in the `ODZF0M` folder, ensure label names match the standard format: `1_Pronacio`, `2_Neutralis`, or `3_Szupinacio`.
    - Delete the `ECSGGY` and `GI9Y8B` folders since these don't have a JSON file.

**Note:** These manual steps are only necessary when the data does not follow the expected structure. Manual editing was chosen over implementing multiple special-case exceptions in the code for better maintainability.

### Docker Instructions

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/data` and output directory to `/app/output` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

**Windows (PowerShell):**
```powershell
docker run --rm --gpus all -v D:\path\to\data:/data -v D:\path\to\output:/app/output -v D:\path\to\log:/app/log dl-project > log/run.log 2>&1
```

**Linux/macOS:**
```bash
docker run --rm --gpus all -v /path/to/data:/data -v /path/to/output:/app/output -v /path/to/log:/app/log dl-project > log/run.log 2>&1
```

**Important Notes:**
*   Replace the paths with actual paths to your dataset, output, and log directories.
*   Your data should be structured as: `/your/data/path/anklealign/{student_folders}/`
*   The `--gpus all` flag enables GPU support (requires nvidia-docker or NVIDIA Container Toolkit).
*   The `--rm` flag automatically removes the container after it finishes.
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container automatically runs the entire pipeline: data preprocessing, training, evaluation, and inference.
*   Results will be saved to your local output directory:
    - `model.pth` - Trained model checkpoint with best validation accuracy
    - `evaluation/` - Confusion matrix, per-class metrics plots, and JSON summary
    - Processed data will be saved to `/data/processed/` with train/val/test splits


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Parses Label Studio JSON exports, validates images, and creates stratified train/val/test splits.
    - `02-training.py`: Main training script with data augmentation, model training loop, and checkpoint saving.
    - `03-evaluation.py`: Evaluates the trained model on test data and generates confusion matrix, metrics plots, and JSON reports.
    - `04-inference.py`: Runs inference on random test samples and displays predictions with confidence scores.
    - `models.py`: Defines the AnkleNet CNN architecture and model factory functions.
    - `config.py`: Central configuration file containing all hyperparameters (epochs, learning rate, batch size, augmentation settings) and paths.
    - `utils.py`: Shared utility functions including logger setup.
    - `run.sh`: Shell script that orchestrates the entire pipeline execution (preprocessing → training → evaluation → inference).

- **`log/`**: Contains execution log files.
    - `run.log`: Complete log file capturing all pipeline outputs (preprocessing stats, training progress, evaluation metrics, and inference results).

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
