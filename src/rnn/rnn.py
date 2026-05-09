import numpy as np
from ..shared.activation_functions import tanh

class RNN:
    def __init__(self, keras_layer=None, input_size=None, hidden_size=None):
        if keras_layer is not None:
            weights = keras_layer.get_weights()
            self.w_xh = weights[0]  # shape (input_size, hidden_size)
            self.w_hh = weights[1]  # shape (hidden_size, hidden_size)
            self.b_h = weights[2]   # shape (hidden_size,)
        else:
            if input_size is not None and hidden_size is not None:
                # Inisialisasi bobot awal dengan nilai kecil
                self.w_xh = np.random.randn(input_size, hidden_size) * 0.01
                self.w_hh = np.random.randn(hidden_size, hidden_size) * 0.01
                self.b_h = np.random.randn(hidden_size)
            else:
                raise ValueError("Incorrect Parameters")
            
    def forward(self, x, h_prev):
        # x shape: (1, input_size)
        # h_prev shape: (1, hidden_size)
        self.x = x
        self.h_prev = h_prev

        net_h = np.dot(x, self.w_xh) + np.dot(h_prev, self.w_hh) + self.b_h
        h = tanh(net_h)

        # cache untuk backpropagation
        cache = (x, h_prev, h)

        return h, cache
    
    def backward(self, dh, cache):
        x, h_prev, h = cache

        # backprop aktivasi tanh
        dtanh = dh * (1 - h**2) # shape (1, hidden_size)

        # gradien untuk bobot dan bias
        dw_xh = np.dot(x.T, dtanh)  # shape (input_size, hidden_size)
        dw_hh = np.dot(h_prev.T, dtanh)  # shape (hidden_size, hidden_size)
        db_h = np.sum(dtanh, axis=0, keepdims=True)  # shape (1, hidden_size)

        # gradien untuk input dan hidden state sebelumnya
        dh_prev = np.dot(dtanh, self.w_hh.T)  # shape (1, hidden_size)
        dx = np.dot(dtanh, self.w_xh.T)  # shape (1, input_size)

        return dx, dh_prev, dw_xh, dw_hh, db_h