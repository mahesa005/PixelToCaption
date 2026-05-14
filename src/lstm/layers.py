import numpy as np
from shared.dense import DenseLayer
from shared.embedding_layer import EmbeddingLayer
from lstm.lstm import LSTM
from shared.optimizer import AdamOptimizer
from shared.loss_function import SparseCategoricalCrossEntropy

class LSTMDecoder:
    """
    Image captioning decoder using a single LSTM cell.

    Architecture:
        1. CNN features are projected to embed_dim via dense_proj (pre-injection).
        2. The projected feature is fed as the first LSTM input (timestep -1).
        3. Caption tokens are embedded and fed sequentially through the LSTM.
        4. At each timestep, dense_out maps the hidden state to a vocab distribution.
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
        """
        Initialize the decoder from Keras layers or dimension specs for random init.

        Args:
            lstm_keras_layer: Keras LSTM layer (optional, for weight transfer).
            embedding_keras_layer: Keras Embedding layer (optional).
            dense_proj_keras_layer: Keras Dense layer for CNN projection (optional).
            dense_out_keras_layer: Keras Dense output layer (optional).
            W, U, b: explicit LSTM weight arrays.
            embed_dim: embedding dimension.
            hidden_dim: LSTM hidden state dimension.
            vocab_size: number of tokens in vocabulary.
            dense_proj_input: input size for the CNN projection dense layer.
        """
        self.lstm       = LSTM(lstm_keras_layer, W, U, b, embed_dim, hidden_dim)
        self.embedding  = EmbeddingLayer(embedding_keras_layer, vocab_size, embed_dim)
        self.dense_proj = DenseLayer(dense_proj_keras_layer, dense_proj_input, embed_dim)
        self.dense_out  = DenseLayer(dense_out_keras_layer, hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.lstm.hidden_dim
        self.embed_dim  = embed_dim if embed_dim is not None else self.lstm.embed_dim
        self.vocab_size = vocab_size

    def forward(self, cnn_features, caption_tokens):
        """
        Run a full forward pass over one (image, caption) pair.

        Args:
            cnn_features: CNN feature vector, shape (cnn_feature_dim,)
            caption_tokens: list of token indices (input side, e.g. starttoken..last-1)

        Returns:
            output: dict mapping timestep index to logit vector, shape (vocab_size,)
            cache: dict mapping timestep index to LSTM cache (used in backward).
                   Key -1 is the pre-injection timestep.
        """
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
        """
        Run BPTT over all timesteps and accumulate gradients into each layer.

        Args:
            cache: dict returned by forward (keys: -1, 0, 1, ..., T-1).
            grad_outputs: dict mapping timestep index to gradient of loss
                          w.r.t. the logit output at that timestep, shape (vocab_size,).
        """
        dh = np.zeros(self.hidden_dim,)
        dc = np.zeros(self.hidden_dim,)

        for t in reversed(range(len(cache) - 1)): # -1 because cache[-1] is for pre-injection
            dh = self.dense_out.backward(grad_outputs[t]) + dh # accumulate BPTT gradient
            dx, dh, dc = self.lstm.backward(dh, dc, cache[t])
        dx, dh, dc = self.lstm.backward(dh, dc, cache[-1])
        self.dense_proj.backward(dx)
    
    def predict(self, cnn_features, start_token, end_token, max_length=20):
        """
        Generate a caption via greedy decoding.

        Args:
            cnn_features: CNN feature vector, shape (cnn_feature_dim,)
            start_token: integer index of the start token.
            end_token: integer index of the end token (generation stops here).
            max_length: maximum number of tokens to generate.

        Returns:
            generated_tokens: list of predicted token indices (excludes start token).
        """
        # Run CNN pre-injection
        pre_inject = self.dense_proj.forward(cnn_features)
        
        # Initialize hidden state and cell state
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        # Run pre-injection
        h, c, _ = self.lstm.forward(pre_inject, h, c)

        # Run caption generation
        i = 0
        last_generated_token = None
        generated_tokens = []
        while i < max_length and last_generated_token != end_token:
            if i == 0:
                embedded_token = self.embedding.forward(start_token)
            else:
                embedded_token = self.embedding.forward(last_generated_token)
            h, c, _ = self.lstm.forward(embedded_token, h, c)
            output = self.dense_out.forward(h)
            token = np.argmax(output)
            generated_tokens.append(token)

            # Update loop conditions
            last_generated_token = token
            i += 1
        return generated_tokens

    def fit(self, features, captions, targets, epochs=10, lr=0.001, use_keras=True, keras_model=None):
        """
        Train the decoder using either Keras or the from-scratch implementation.

        Args:
            features: list of CNN feature vectors.
            captions: list of input token sequences (one per sample).
            targets: list of target token sequences (one per sample).
            epochs: number of training epochs.
            lr: learning rate for Adam (from-scratch mode only).
            use_keras: if True, delegate training to keras_model.fit.
            keras_model: compiled Keras Model (required when use_keras=True).
        """
        if use_keras:
            if keras_model is None:
                raise ValueError("keras_model required for Keras training")
            return keras_model.fit(features, targets, epochs=epochs)

        optimizer = AdamOptimizer(learning_rate=lr)
        loss_fn = SparseCategoricalCrossEntropy()

        for epoch in range(epochs):
            total_loss = 0

            for cnn_feature, caption_tokens, target_tokens in zip(features, captions, targets):
                self.lstm.zero_grad()

                # forward
                outputs, cache = self.forward(cnn_feature, caption_tokens)

                # compute loss + grad_outputs per timestep
                grad_outputs = {}
                for t in outputs:
                    total_loss += loss_fn.forward(outputs[t], target_tokens[t])
                    grad_outputs[t] = loss_fn.backward(outputs[t], target_tokens[t])

                # backward
                self.backward(cache, grad_outputs)

                # update weights
                params = {
                    'W_i': self.lstm.W_i, 'W_f': self.lstm.W_f,
                    'W_c': self.lstm.W_c, 'W_o': self.lstm.W_o,
                    'U_i': self.lstm.U_i, 'U_f': self.lstm.U_f,
                    'U_c': self.lstm.U_c, 'U_o': self.lstm.U_o,
                    'dense_out_W': self.dense_out.weights,
                    'dense_out_b': self.dense_out.bias,
                    'dense_proj_W': self.dense_proj.weights,
                    'dense_proj_b': self.dense_proj.bias,
                }
                grads = {
                    'W_i': self.lstm.dW_i, 'W_f': self.lstm.dW_f,
                    'W_c': self.lstm.dW_c, 'W_o': self.lstm.dW_o,
                    'U_i': self.lstm.dU_i, 'U_f': self.lstm.dU_f,
                    'U_c': self.lstm.dU_c, 'U_o': self.lstm.dU_o,
                    'dense_out_W': self.dense_out.dW,
                    'dense_out_b': self.dense_out.db,
                    'dense_proj_W': self.dense_proj.dW,
                    'dense_proj_b': self.dense_proj.db,
                }
                optimizer.step(params, grads)

            print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss:.4f}")

            
                
