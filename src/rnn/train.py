# Keras SimpleRNN training pipeline

import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Embedding, Concatenate
from tensorflow.keras.models import Model

from shared.preprocessing import (
	load_captions, clean_and_wrap_captions, build_tokenizer,
	calculate_max_length
)

# konfigurasi
CAPTIONS_PATH   = 'data/captions.txt'
FEATURES_PATH   = 'data/flickr8k_features.npy'
TOKENIZER_PATH  = 'data/tokenizer.json'
WEIGHTS_PATH    = 'weights/rnn_keras.weights.h5'

EMBED_DIM   = 256
HIDDEN_DIM  = 512
EPOCHS      = 20
BATCH_SIZE  = 64

# fungsi utk load CNN features dan membuat tokenizer dari caption
def load_data():
	# load CNN features
	features = np.load(FEATURES_PATH, allow_pickle=True).item()

	# load captions
	caption_mapping = load_captions(CAPTIONS_PATH)
	cleaned_mapping = clean_and_wrap_captions(caption_mapping)

	# build tokenizer
	tokenizer, all_captions = build_tokenizer(cleaned_mapping, TOKENIZER_PATH)
	max_length = calculate_max_length(all_captions)

	return features, tokenizer, cleaned_mapping, max_length

# fungsi utk membuat pasangan input-output utk teacher forcing
def prepare_sequences(features, tokenizer, cleaned_mapping, max_length, embed_dim):
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

# fungsi untuk membangun model Keras SimpleRNN
def build_rnn_keras_model(vocab_size, embed_dim, hidden_dim, max_length, cnn_feature_dim=2048):
	# CNN feature input
	cnn_input    = Input(shape=(cnn_feature_dim,), name='cnn_input')
	cnn_proj     = Dense(embed_dim, activation='relu', name='cnn_proj')(cnn_input)
	cnn_proj     = keras.layers.Reshape((1, embed_dim))(cnn_proj)  # (1, embed_dim)

	# caption sequence input
	caption_input    = Input(shape=(max_length,), name='caption_input')
	caption_embed    = Embedding(vocab_size, embed_dim, mask_zero=True)(caption_input)

	# pre-inject: concat CNN feature di depan sequence
	merged       = Concatenate(axis=1)([cnn_proj, caption_embed])  # (max_length+1, embed_dim)

	# RNN decoder
	rnn_out      = SimpleRNN(hidden_dim, return_sequences=False)(merged)

	# output
	outputs      = Dense(vocab_size, activation='softmax')(rnn_out)
    
	model = Model(inputs=[cnn_input, caption_input], outputs=outputs)
	
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
	return model

# fungsi training pipeline utama
def train():
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
	model = build_rnn_keras_model(vocab_size, EMBED_DIM, HIDDEN_DIM, max_length)
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
