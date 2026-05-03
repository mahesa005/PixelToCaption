import numpy as np


class EmbeddingLayer:
    """
    A layer that has the function of embedding tokens into vectors.
    """
    def __init__(self, keras_layer=None, vocab_size=None, embed_dims=None):
        """
        vocab_size: the total number of the words the model can learn
        embed_dims: the dimension of the vector embedding
        """
        if keras_layer is not None:
            self.weights = keras_layer.get_weights()[0] 
        else:
            if vocab_size is not None:
                self.weights = np.random.randn(vocab_size, embed_dims)
            else:
                raise ValueError("NO WEIGHTS TO INITIALIZE")
        if embed_dims is not None:
            self.embed_dims = embed_dims
        else: 
            self.embed_dims = self.weights.shape[1]

    def forward(self, token: int):
        self.token = token
        return self.weights[token]

    def backward(self, grad_out):
        pass
        # self.dW = np.zeros_like(self.weights)
        # self.dW[self.token_id] = grad_out  # update baris yang dipake aja
        # return self.dW

        
        
        