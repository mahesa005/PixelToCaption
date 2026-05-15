import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = (150, 150)
BATCH_SIZE  = 32
EPOCHS      = 20
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '../../weights/cnn')


def load_dataset(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    norm = layers.Rescaling(1./255)

    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'seg_train', 'seg_train'),
        image_size=img_size, batch_size=batch_size,
        label_mode='int', shuffle=True, seed=42
    ).map(lambda x, y: (norm(x), y)).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'seg_test', 'seg_test'),
        image_size=img_size, batch_size=batch_size,
        label_mode='int', shuffle=False
    ).map(lambda x, y: (norm(x), y)).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_cnn_model(num_conv_layers=2, filters=None, kernel_sizes=None,
                    pooling_type='max', img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    if filters is None:
        filters = [32] * num_conv_layers
    if kernel_sizes is None:
        kernel_sizes = [3] * num_conv_layers

    PoolLayer = layers.MaxPooling2D if pooling_type == 'max' else layers.AveragePooling2D

    inputs = keras.Input(shape=(*img_size, 3))
    x = inputs

    for i in range(num_conv_layers):
        x = layers.Conv2D(filters[i], kernel_sizes[i], activation='relu', padding='same')(x)
        x = PoolLayer(pool_size=(2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_locally_connected_model(filters=None, kernel_sizes=None,
                                   pooling_type='max', img_size=IMG_SIZE,
                                   num_classes=NUM_CLASSES):
    if filters is None:
        filters = [32, 64]
    if kernel_sizes is None:
        kernel_sizes = [3, 3]

    PoolLayer = layers.MaxPooling2D if pooling_type == 'max' else layers.AveragePooling2D

    inputs = keras.Input(shape=(*img_size, 3))
    x = inputs

    for i in range(len(filters)):
        x = layers.LocallyConnected2D(filters[i], kernel_sizes[i], activation='relu')(x)
        x = PoolLayer(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_experiments(data_dir):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    train_ds, val_ds = load_dataset(data_dir)
    results = {}

    filter_variants = {
        'small': {2: [32, 64],       3: [32, 64, 64]},
        'large': {2: [64, 128],      3: [64, 128, 128]}
    }

    kernel_variants = {
        'k3': {2: [3, 3],   3: [3, 3, 3]},
        'k5': {2: [5, 5],   3: [5, 5, 5]}
    }

    configs = []
    for n_layers in [2, 3]:
        for fkey, f_dict in filter_variants.items():
            for kkey, k_dict in kernel_variants.items():
                for pool in ['max', 'average']:
                    name = f"conv{n_layers}_{fkey}_{kkey}_{pool}"
                    configs.append({
                        'name':            name,
                        'num_conv_layers': n_layers,
                        'filters':         f_dict[n_layers],
                        'kernel_sizes':    k_dict[n_layers],
                        'pooling_type':    pool
                    })

    for cfg in configs:
        print(f"\n{'='*50}\nEksperimen: {cfg['name']}\n{'='*50}")
        model = build_cnn_model(
            cfg['num_conv_layers'], cfg['filters'],
            cfg['kernel_sizes'],   cfg['pooling_type']
        )
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

        w_path = os.path.join(WEIGHTS_DIR, f"{cfg['name']}.weights.h5")
        model.save_weights(w_path)
        print(f"Bobot disimpan: {w_path}")

        results[cfg['name']] = {'history': history.history, 'config': cfg}

    return results