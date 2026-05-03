import numpy as np

class DenseLayer:
    def __init__(self, keras_layer=None, input_size=None, output_size=None):
        if keras_layer is not None:
            self.weights, self.bias = keras_layer.get_weights()
        else:
            if input_size is not None and output_size is not None:
                self.weights, self.bias = np.random.randn(input_size, output_size), np.random.randn(output_size)
            else:
                raise ValueError("NO WEIGHTS TO INITIALIZE")
        

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias 

    def backward(self, grad_out):
        pass 
        # dL_dW = self.x.T @ grad_out
        # dL_db = grad_out.sum()
        # dL_dx = grad_out @ self.W.T  # ini yang diterusin ke layer sebelumnya
        # return dL_dx