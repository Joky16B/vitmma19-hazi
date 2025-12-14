# VITMMA19-HAZI

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Jókay Benedek
- **Aiming for +1 Mark**: No

### Solution Description

### Data Preparation

Follow these steps for the data preparation part of the process:
1. Download the anklealign.zip from the root directory of the sharepoint.
2. Extract the zip into `.\data\` folder of this project (mount this data folder to docker in the later part of the instructions)
3. Manually change 3 things in the data.
    - From `H51B9J.json` remove the _H51B9J part from the file ends (ctrl + F, replace all)
    - In the NC102T folder extract the images from the subfolders and delete the subfolders
    - In `project-2-at-2025-10-16-02-08-8ee4fdfa.json` in the ODZF0M replace the incorrect label names with the correct ones. ()

(Note: These manual steps were only necessary in cases where the data was not following the predefined structure. I found that manual editing was quicker than making exceptions in the code for each case.)

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
docker run --rm --gpus all -v D:\path\to\data:/data -v D:\path\to\output:/app/output dl-project > log/run.log 2>&1
```

**Linux/macOS:**
```bash
docker run --rm --gpus all -v /path/to/data:/data -v /path/to/output:/app/output dl-project > log/run.log 2>&1
```

*   Replace the paths with actual paths to your dataset and desired output location.
*   Your data should be structured as: `/your/data/path/anklealign/{student_folders}/`
*   The `--gpus all` flag enables GPU support (requires nvidia-docker).
*   The `--rm` flag automatically removes the container after it finishes.
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).
*   Results will be saved to your local output directory:
    - `model.pth` - Trained model checkpoint
    - `evaluation/` - Confusion matrix, metrics plots, and JSON summary


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.
    - `run.sh`: 

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
