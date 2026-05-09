import numpy as np
from rnn import RNN
from ..shared.dense import DenseLayer
from ..shared.embedding_layer import EmbeddingLayer
from ..shared.activation_functions import softmax

class RNNDecoder:
    def __init__(self, cnn_features_dim, embed_dim, hidden_dim, vocab_size):
        # Proyeksi dimensi fitur CNN ke dimensi embedding
        self.projection_layer = DenseLayer(input_dim=cnn_features_dim, output_size=embed_dim)

        # Embedding layer untuk token input
        self.embedding_layer = EmbeddingLayer(vocab_size=vocab_size, embed_dims=embed_dim)

        # RNN layer
        self.rnn_cell = RNN(input_size=embed_dim, hidden_size=hidden_dim)

        # Output layer
        self.output_layer = DenseLayer(input_size=hidden_dim, output_size=vocab_size)

        # dimensi tersembunyi untuk forward pass
        self.hidden_dim = hidden_dim

    def forward(self, image_features, caption_tokens):
        # inisialitasi hidden state awal (h_0)
        h_t = np.zeros((1, self.hidden_dim))

        # list probabilitas prediksi di setiap timestep
        predicted_probabilities = []

        # === fase 1: Pre-Inject gambar ===
        # ubah fitur gambar (2048) ke dimensi embedding
        x_image = self.projection_layer.forward(image_features)

        # input RNN untuk timestep pertama adalah embedding dari fitur gambar
        h_t = self.rnn_cell.forward(x_image, h_t)

        # === fase 2: pemrosesan sekuens teks ===
        for token in caption_tokens:
            # vektor embedding dari token input
            x_word = self.embedding_layer.forward(token)

            # Embedding layer mengembalikan vektor 1D
            # ubah menjadi 2D untuk perhitungan dot product dengan bobot RNN
            if x_word.ndim == 1:
                x_word = np.expand_dims(x_word, axis=0)  # ubah ke shape (1, embed_dim)

            # forward pass melalui RNN cell
            h_t = self.rnn_cell.forward(x_word, h_t)

            # prediksi distribusi probabilitas untuk token berikutnya
            logits = self.output_layer.forward(h_t)
            probs = softmax(logits)

            # simpan probabilitas prediksi untuk timestep ini
            predicted_probabilities.append(probs)
        
        return predicted_probabilities
    
    def backward(self, d_logits_list, caches_list, caption_tokens):
        # d_logits_list: list gradien dari output layer di setiap timestep
        # caches_list: list cache yang disimpan dari forward pass di setiap timestep
        # caption_tokens: list token input untuk setiap timestep
        
        # inisialisasi gradien akumulatif awal
        dw_xh_total = np.zeros_like(self.rnn_cell.w_xh)
        dw_hh_total = np.zeros_like(self.rnn_cell.w_hh)
        db_h_total = np.zeros_like(self.rnn_cell.b_h)

        # insialisasi gradien untuk hidden state berikutnya
        dh_next = np.zeros((1, self.hidden_dim))

        # panjang sekuens
        sequence_length = len(caches_list)

        # iterasi mundur dari timestep T ke 0 (Backpropagation Through Time)
        for t in reversed(range(sequence_length)):
            # backprop dense output layer
            d_out = d_logits_list[t]
            dh_output = self.output_layer.backward(d_out)

            # total dh di timestep ini: gabungan gradien dari output layer dan timestep berikutnya
            dh = dh_output + dh_next

            # backward pass RNN
            cache_t = caches_list[t]
            dx, dh_prev, dw_xh, dw_hh, db_h = self.rnn_cell.backward(dh, cache_t)

            # akumulasi gradien untuk bobot RNN
            dw_xh_total += dw_xh
            dw_hh_total += dw_hh
            db_h_total += db_h

            # update dh_next untuk timestep berikutnya
            dh_next = dh_prev

            # backward pass embedding layer
            token_t = caption_tokens[t]  # token input di timestep t
            self.embedding_layer.token_id = token_t  # set token_id untuk update gradien
            self.embedding_layer.backward(dx)

        return dw_xh_total, dw_hh_total, db_h_total