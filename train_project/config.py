import torch

BATCH_SIZE = 8  # Increase / decrease according to GPU memeory.
RESIZE_TO = 1280  # Resize the image for training and transforms.
NUM_EPOCHS = 10  # Number of epochs to train for.
NUM_WORKERS = 4  # Number of parallel workers for data loading.

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Training images and labels files directory.
TRAIN_DIR = "data/train"
# Validation images and labels files directory.
VALID_DIR = "data/val"
# Test images and labels files directory.
TEST_DIR = "data/test"

# Classes: 0 index is reserved for background.
CLASSES = ["__background__", "pedestrian"]


NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = True

# Location to save model and plots.
OUT_DIR = "outputs"
