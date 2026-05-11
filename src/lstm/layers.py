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
            h, c, cache[i] = self.lstm.forward(embedded_token, h, c)
            output[i] = self.dense_out.forward(h)
            i += 1
        return output, cache

    def backward(self, cache, grad_outputs):
        dh = np.zeros(self.hidden_dim,)
        dc = np.zeros(self.hidden_dim,)

        for t in reversed(range(len(cache) - 1)): # -1 because cache[-1] is for pre-injection
            dh = self.dense_out.backward(grad_outputs[t]) # gradient output from timestep t
            dx, dh, dc = self.lstm.backward(dh, dc, cache[t])
        dx, dh, dc = self.lstm.backward(dh, dc, cache[-1])
        self.dense_proj.backward(dx)
    
    def predict(self, cnn_features, start_token, end_token, max_length=20):
        # Run CNN pre-injection
        pre_inject = self.dense_proj.forward(cnn_features)
        
        # Initialize hidden state and cell state
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        # Run pre-injection
        h, c, _ = self.lstm.forward(pre_inject, c, h)

        # Run caption generation
        i = 0
        last_generated_token = None
        generated_tokens = []
        while i < max_length and last_generated_token != end_token:
            if i == 0:
                embedded_token = self.embedding.forward(start_token)
            else:
                embedded_token = self.embedding.forward(last_generated_token)
            h, c, _ = self.lstm.forward(embedded_token, c, h)
            output = self.dense_out.forward(h)
            token = np.argmax(output)
            generated_tokens.append(token)

            # Update loop conditions
            last_generated_token = token
            i += 1
        return generated_tokens



        
            
