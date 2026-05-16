import numpy as np


class EmbeddingLayer:
    """
    A layer that has the function of embedding tokens into vectors.
    """
    def __init__(self, keras_layer=None, vocab_size=None, embed_dim=None):
        """
        vocab_size: the total number of the words the model can learn
        embed_dim: the dimension of the vector embedding
        """
        if keras_layer is not None:
            self.weights = keras_layer.get_weights()[0]
        else:
            if vocab_size is not None and embed_dim is not None:
                self.weights = np.random.randn(vocab_size, embed_dim)
            else:
                raise ValueError("NO WEIGHTS TO INITIALIZE")
        
        self.embed_dim = embed_dim if embed_dim is not None else self.weights.shape[1]

    def forward(self, token: int):
        self.token = token
        return self.weights[token]

    def backward(self, grad_out):
        # matriks graiden kosong berukuran sama dengan matriks embedding
        self.dW = np.zeros_like(self.weights)

        # hanya update baris yang digunakan (token) dengan gradien yang diterima
        self.dW[self.token] = grad_out

        return None

        
        
        