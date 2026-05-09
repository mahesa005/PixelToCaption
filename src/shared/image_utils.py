from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import os
from tqdm import tqdm

# fungsi load gambar dan resize ke dimensi 299x299 (input InceptionV3)
def load_image(path: str, target_dim: tuple = (299, 299)) -> np.ndarray:
    try:
        image = Image.open(path).convert("RGB")
        image_array = np.array(image.resize(target_dim), dtype=np.float32)
        return preprocess_input(image_array)
    
    except Exception as e:
        print(f"Gagal memuat gambar {path}: {e}")
        return None
    
# fungsi load batch gambar, mengembalikan array 4D (batch_size, height, width, channels)
def load_batch(paths: list[str], target_dim: tuple = (299, 299)) -> np.ndarray:
    batch = []
    valid_paths = [] # path yang berhasil dimuat

    for path in paths:
        img = load_image(path, target_dim)
        if img is not None:
            batch.append(img)
            valid_paths.append(path)

    return np.array(batch), valid_paths

# fungsi ekstrak fitur CNN dan simpan ke dictionary .npy
def extract_features(paths: list[str], output_path: str, batch_size: int = 64, encoder=None):
    # buat folder output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if encoder is None:
        encoder = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    target_dim = encoder.input_shape[1:3]
    if target_dim == (None, None): # fallback input_shape None
        target_dim = (299, 299)  # default InceptionV3

    # dictionary untuk menyimpan fitur, key: nama file, value: vektor fitur
    features_dict = {}

    # ekstraksi fitur per batch untuk menghindari Out of Memory
    print(f"Memulai ekstraksi fitur untuk {len(paths)} gambar (Batch size: {batch_size})...")
    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting"):
        batch_paths = paths[i : i + batch_size]
        batch_images, valid_paths = load_batch(batch_paths, target_dim)
        
        # prediksi hanya jika ada gambar yang valid di batch ini
        if len(batch_images) > 0:
            batch_features = encoder.predict(batch_images, verbose=0)

            # mapping filename ke vektor fitur
            for path, feature in zip(valid_paths, batch_features):
                filename = os.path.basename(path) # ambil nama file saja
                features_dict[filename] = feature
         
    # simpan dictionary fitur ke file .npy
    np.save(output_path, features_dict)
    print(f"\nEkstraksi fitur selesai. Berhasil menyimpan {len(features_dict)} fitur ke {output_path}")

    return features_dict