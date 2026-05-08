import pathlib
# pyrefly: ignore [missing-import]
import numpy as np
from ..shared.activation_functions import tanh, sigmoid

class LSTM:
    """
    W shape: (embed_dim, 4 * hidden_dim)
    U shape: (hidden_dim, 4 * hidden_dim)
    b shape: (4 * hidden_dim,)
    order: [i | f | c | o]
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
        self.W_i, self.W_f, self.W_c, self.W_o = self.W[:, 0*h:1*h], self.W[:, 1*h:2*h], self.W[:, 2*h:3*h], self.W[:, 3*h:4*h]
        self.U_i, self.U_f, self.U_c, self.U_o = self.U[:, 0*h:1*h], self.U[:, 1*h:2*h], self.U[:, 2*h:3*h], self.U[:, 3*h:4*h]
        self.b_i, self.b_f, self.b_c, self.b_o = self.b[0*h:1*h], self.b[1*h:2*h], self.b[2*h:3*h], self.b[3*h:4*h]

    def forward(self, x, c_prev, h_prev):
        f = self.forget_gate(x, h_prev)
        i = self.input_gate(x, h_prev)
        candidate = self.candidate(x, h_prev)
        o = self.output_gate(x, h_prev)
        c = c_prev * f + (candidate * i)
        h = tanh(c) * o
        return h, c

    def backward(self):
        pass

    def forget_gate(self, x, h_prev):
        output_x = x @ self.W_f
        output_h = h_prev @ self.U_f
        output = sigmoid(output_x + output_h + self.b_f)
        return output
    
    def input_gate(self, x, h_prev):
        output_x = x @ self.W_i
        output_h = h_prev @ self.U_i
        output = sigmoid(output_x + output_h + self.b_i)
        return output
    
    def candidate(self, x, h_prev):
        output_x = x @ self.W_c
        output_h = h_prev @ self.U_c
        output = tanh(output_x + output_h + self.b_c)
        return output

    def output_gate(self, x, h_prev):
        output_x = x @ self.W_o
        output_h = h_prev @ self.U_o
        output = sigmoid(output_x + output_h + self.b_o)
        return output
