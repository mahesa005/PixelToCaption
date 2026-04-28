from PIL import Image
from keras.applications import InceptionV3
import numpy as np

def load_image(path: str, target_dim: tuple = (224, 224)) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = np.array(image.resize(target_dim)) / 255.0
    return image

def load_batch(paths: list[str], target_dim: tuple = (224, 224)) -> np.ndarray:
    batch = []
    for path in paths:
        batch.append(load_image(path, target_dim))
    return np.array(batch)

def extract_features(paths: list[str], output_path: str, encoder=None):
    if encoder is None:
        encoder = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    target_dim = encoder.input_shape[1:3]
    features = encoder.predict(load_batch(paths, target_dim))
    np.save(output_path, features)