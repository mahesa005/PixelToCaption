import numpy as np
from ..shared.dense import DenseLayer
from ..shared.embedding_layer import EmbeddingLayer
from ..lstm.lstm import LSTM

class LSTMDecoder:
    """
    Input comment nanti
    """
    def __init__(
        self,
        lstm_keras_layer=None,
        embedding_keras_layer=None,
        dense_proj_keras_layer=None,
        dense_out_keras_layer=None,
        W=None, U=None, b=None,
        embed_dim=None,
        hidden_dim=None,
        vocab_size=None,
        dense_proj_input=None,
    ):
        self.lstm       = LSTM(lstm_keras_layer, W, U, b, embed_dim, hidden_dim)
        self.embedding  = EmbeddingLayer(embedding_keras_layer, vocab_size, embed_dim)
        self.dense_proj = DenseLayer(dense_proj_keras_layer, dense_proj_input, embed_dim)
        self.dense_out  = DenseLayer(dense_out_keras_layer, hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.lstm.hidden_dim
        self.embed_dim  = embed_dim if embed_dim is not None else self.lstm.embed_dim
        self.vocab_size = vocab_size

    def forward(self, cnn_features, caption_tokens):
        # Run CNN pre-injection
        pre_inject = self.dense_proj.forward(cnn_features)
        
        # initiate hidden state, cell state, and cache
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        cache = {}

        # Run timestep t = -1
        h, c, cache[-1] = self.lstm.forward(pre_inject, h, c)

        # Run many to many
        i = 0
        output = {}
        for token in caption_tokens:
            embedded_token = self.embedding.forward(token)
            h, c, cache[i] = self.lstm.forward(embedded_token, c, h)
            output[i] = self.dense_out.forward(h)
            i += 1
        return output, cache