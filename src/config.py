import os

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_DIR = os.environ.get("DATA_DIR", LOCAL_DATA_DIR if os.path.exists(LOCAL_DATA_DIR) else "/data")

# Training hyperparameters
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Image preprocessing
IMAGE_SIZE = 224
NUM_CLASSES = 3
CLASS_NAMES = ['Pronation', 'Neutral', 'Supination']

# Data augmentation parameters
AUG_RESIZE = 256
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.2
AUG_ROTATION = 10

# Image normalization (ImageNet stats)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Learning rate scheduler parameters
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")


LOCAL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output" if os.path.exists("/.dockerenv") else LOCAL_OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model.pth")
