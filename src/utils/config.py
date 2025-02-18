import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = os.path.join(BASE_DIR, "dataset/images/")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "embeddings.pkl")
FILENAMES_FILE = os.path.join(BASE_DIR, "data", "filenames.pkl")
MODEL_WEIGHTS = "imagenet"
IMAGE_SIZE = (224, 224)
