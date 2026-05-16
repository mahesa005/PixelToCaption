import numpy as np
import json
from pathlib import Path
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU enabled: {[gpu.name for gpu in gpus]}")
else:
    print("No GPU found, running on CPU")

from tensorflow import keras
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import tokenizer_from_json


class PrependCNN(tf.keras.layers.Layer):
    """
    Concatenates a (batch, 1, embed_dim) CNN projection in front of the
    (batch, T, embed_dim) caption embedding and correctly propagates the
    Embedding mask so padding zeros are not fed to the LSTM.
    """
    def call(self, inputs):
        cnn_proj, cap_embed = inputs
        return tf.concat([cnn_proj, cap_embed], axis=1)

    def compute_mask(self, inputs, mask=None):
        _, cap_mask = mask
        if cap_mask is None:
            return None
        batch_size = tf.shape(inputs[0])[0]
        cnn_mask   = tf.ones((batch_size, 1), dtype=tf.bool)
        return tf.concat([cnn_mask, cap_mask], axis=1)

from shared.preprocessing import (
    load_captions, clean_and_wrap_captions, build_tokenizer,
    calculate_max_length, captions_to_sequences_and_pad
)

# Config
CAPTIONS_PATH   = 'data/captions.txt'
FEATURES_PATH   = 'data/flickr8k_features.npy'
TOKENIZER_PATH  = 'data/tokenizer.json'
WEIGHTS_PATH    = 'weights/lstm_keras.weights.h5'

EMBED_DIM   = 256
HIDDEN_DIM  = 512
EPOCHS      = 20
BATCH_SIZE  = 64

# Load Data
def load_data():
    """
    Load CNN features, captions, and build the tokenizer.

    Returns:
        features: dict mapping image_id to feature vector, shape (2048,).
        tokenizer: fitted Keras tokenizer.
        cleaned_mapping: dict mapping image_id to list of cleaned caption strings.
        max_length: maximum caption length in tokens.
    """
    # load CNN features
    features = np.load(FEATURES_PATH, allow_pickle=True).item()

    # load captions
    caption_mapping = load_captions(CAPTIONS_PATH)
    cleaned_mapping = clean_and_wrap_captions(caption_mapping)

    # build tokenizer
    tokenizer, all_captions = build_tokenizer(cleaned_mapping, TOKENIZER_PATH)
    max_length = calculate_max_length(all_captions)

    return features, tokenizer, cleaned_mapping, max_length

# Prepare X, Y
def prepare_sequences(features, tokenizer, cleaned_mapping, max_length, embed_dim):
    """
    Build teacher-forcing input/output pairs from captions.

    For each caption of length T, produces T-1 (input_seq, cnn_feature) -> target_token
    pairs by shifting the sequence one step at a time.

    Args:
        features: dict mapping image_id to CNN feature vector.
        tokenizer: fitted Keras tokenizer.
        cleaned_mapping: dict mapping image_id to list of caption strings.
        max_length: sequence length to pad/truncate inputs to.
        embed_dim: unused here, kept for interface consistency.

    Returns:
        X: list of (cnn_feature, padded_input_seq) tuples.
        y: numpy array of target token indices, shape (N,).
    """
    vocab_size = len(tokenizer.word_index) + 1

    X, y = [], []

    for image_id, captions in cleaned_mapping.items():
        # ambil feature vector untuk gambar ini
        if image_id not in features:
            continue
        feature = features[image_id]  # shape: (2048,)

        for caption in captions:
            # tokenize caption
            seq = tokenizer.texts_to_sequences([caption])[0]

            # buat pasangan input-output dengan teacher forcing
            for i in range(1, len(seq)):
                in_seq  = seq[:i]         # input: semua token sebelum posisi i
                out_seq = seq[i]          # target: token di posisi i

                # padding input sequence
                in_seq = keras.preprocessing.sequence.pad_sequences(
                    [in_seq], maxlen=max_length, padding='post'
                )[0]

                X.append((feature, in_seq))
                y.append(out_seq)

    return X, np.array(y)

# Build Model
def build_keras_model(vocab_size, embed_dim, hidden_dim, max_length, cnn_feature_dim=2048):
    """
    Build and compile the Keras LSTM captioning model.

    Uses a pre-injection architecture: CNN features are projected to embed_dim
    and prepended to the embedded caption sequence before the LSTM.

    Args:
        vocab_size: total number of tokens in the vocabulary.
        embed_dim: embedding and CNN projection dimension.
        hidden_dim: LSTM hidden state size.
        max_length: fixed caption input length (tokens).
        cnn_feature_dim: dimension of the input CNN feature vector (default 2048 for InceptionV3).

    Returns:
        model: compiled Keras Model with Adam optimizer and sparse CCE loss.
    """
    # CNN feature input
    cnn_input    = Input(shape=(cnn_feature_dim,), name='cnn_input')
    cnn_proj     = Dense(embed_dim, activation='relu', name='cnn_proj')(cnn_input)
    cnn_proj     = Reshape((1, embed_dim))(cnn_proj)  # (1, embed_dim)

    # caption sequence input
    cap_input    = Input(shape=(max_length,), name='cap_input')
    cap_embed    = Embedding(vocab_size, embed_dim, mask_zero=True)(cap_input)

    # pre-inject: prepend CNN feature with correct mask propagation
    merged       = PrependCNN()([cnn_proj, cap_embed])

    # LSTM decoder
    lstm_out     = LSTM(hidden_dim, return_sequences=False)(merged)

    # output
    outputs      = Dense(vocab_size, activation='softmax')(lstm_out)

    model = Model(inputs=[cnn_input, cap_input], outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train
def train():
    """Full training pipeline: load data, prepare sequences, build model, train, and save weights."""
    print("Loading data...")
    features, tokenizer, cleaned_mapping, max_length = load_data()

    vocab_size = len(tokenizer.word_index) + 1
    print(f"vocab_size={vocab_size}, max_length={max_length}")

    print("Preparing sequences...")
    X, y = prepare_sequences(features, tokenizer, cleaned_mapping, max_length, EMBED_DIM)

    # split X jadi cnn_features dan cap_sequences
    cnn_features   = np.array([x[0] for x in X])  # (N, 2048)
    cap_sequences  = np.array([x[1] for x in X])  # (N, max_length)

    print(f"Training samples: {len(y)}")

    print("Building model...")
    model = build_keras_model(vocab_size, EMBED_DIM, HIDDEN_DIM, max_length)
    model.summary()

    print("Training...")
    model.fit(
        [cnn_features, cap_sequences],
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    Path('weights').mkdir(exist_ok=True)
    model.save_weights(WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")

if __name__ == '__main__':
    train()