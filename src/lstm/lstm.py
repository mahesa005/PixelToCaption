import numpy as np

class LSTM:
    """
    W shape: (embed_dim, 4 * hidden_dim)
    U shape: (hidden_dim, 4 * hidden_dim)
    b shape: (4 * hidden_dim,)
    order: [i | f | g | o]
    """
    def __init__(self, keras_layer=None, W=None, U=None, b=None, embed_dim=None, hidden_dim=None):
        if keras_layer is not None:
            self.W, self.U, self.b = keras_layer.get_weights()
            self.hidden_dim = self.W.shape[1] // 4
        else:
            if W is not None and U is not None and b is not None:
                self.W, self.U, self.b = W, U, b
                self.hidden_dim = hidden_dim or self.W.shape[1] // 4
            elif embed_dim is not None and hidden_dim is not None:
                self.hidden_dim = hidden_dim
                self.W = np.random.randn(embed_dim, 4 * hidden_dim)
                self.U = np.random.randn(hidden_dim, 4 * hidden_dim)
                self.b = np.random.randn(4 * hidden_dim)
            else:
                raise ValueError("Incorrect Parameters")

        h = self.hidden_dim
        self.W_i, self.W_f, self.W_g, self.W_o = self.W[:, 0*h:1*h], self.W[:, 1*h:2*h], self.W[:, 2*h:3*h], self.W[:, 3*h:4*h]
        self.U_i, self.U_f, self.U_g, self.U_o = self.U[:, 0*h:1*h], self.U[:, 1*h:2*h], self.U[:, 2*h:3*h], self.U[:, 3*h:4*h]
        self.b_i, self.b_f, self.b_g, self.b_o = self.b[0*h:1*h], self.b[1*h:2*h], self.b[2*h:3*h], self.b[3*h:4*h]

    def forward(self, x, cell_state, hidden_state):
        pass


    def forget(self, x, cell_sate, hidden_state):
        