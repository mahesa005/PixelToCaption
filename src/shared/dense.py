import numpy as np

class DenseLayer:
    def __init__(self, keras_layer=None, input_size=None, output_size=None):
        if keras_layer is not None:
            weights = keras_layer.get_weights()
            self.weights = weights[0]
            self.bias = weights[1] if len(weights) > 1 else np.zeros(weights[0].shape[1])
        else:
            if input_size is not None and output_size is not None:
                # inisialisasi bobot dengan metode He/Xavier initialization
                self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
                self.bias = np.zeros(output_size)  # bias diinisialisasi ke nol
            else:
                raise ValueError("NO WEIGHTS TO INITIALIZE")
        
        # gradien bobot dan bias
        self.dW = None
        self.db = None
        

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias

    def backward(self, grad_out):
        # gradien input x
        # nilai ini akan diteruskan ke layer sebelumnya (RNN/LSTM)
        dL_dx = grad_out @ self.weights.T

        # gradien bobot
        self.dW = self.x.T @ grad_out

        # gradien bias
        # jumlahkan gradien dari semua sampel dalam batch untuk setiap unit output
        self.db = grad_out.sum(axis=0)
        
        return dL_dx
