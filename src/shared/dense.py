class DenseLayer:
    def __init__(self, keras_layer=None):
        if keras_layer is not None:
            self.weights, self.bias = keras_layer.get_weights()

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias 

    def backward(self, grad_out):
        pass 
        # dL_dW = self.x.T @ grad_out
        # dL_db = grad_out.sum()
        # dL_dx = grad_out @ self.W.T  # ini yang diterusin ke layer sebelumnya
        # return dL_dx