import numpy as np

class DenseLayer:
    def __init__(self, keras_layer=None, input_size=None, output_size=None):
        if keras_layer is not None:
            weights = keras_layer.get_weights()
            self.weights = weights[0]
            self.bias = weights[1] if len(weights) > 1 else np.zeros(weights[0].shape[1])
        else:
            if input_size is not None and output_size is not None:
                self.weights = np.random.randn(input_size, output_size)
                self.bias = np.random.randn(output_size)
            else:
                raise ValueError("NO WEIGHTS TO INITIALIZE")

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias

    def backward(self, grad_out):
        pass
        # dL_dW = self.x.T @ grad_out
        # dL_db = grad_out.sum()
        # dL_dx = grad_out @ self.weights.T
        # return dL_dx