import numpy as np
from .layers import (
    Conv2DLayer, LocallyConnected2DLayer,
    MaxPooling2DLayer, AveragePooling2DLayer,
    GlobalMaxPooling2DLayer, GlobalAveragePooling2DLayer,
    FlattenLayer, ActivationLayer
)
from ..shared.dense import DenseLayer


class CNNScratch:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x):
        return int(np.argmax(self.forward(x)))

    def predict_batch(self, images, batch_size=32):
        N       = images.shape[0]
        results = []
        for start in range(0, N, batch_size):
            end        = min(start + batch_size, N)
            batch      = images[start:end]       # (batch_size, H, W, C)
            out        = self.forward(batch)     # (batch_size, num_classes)
            preds      = np.argmax(out, axis=-1) # (batch_size,)
            results.append(preds)
        return np.concatenate(results)

    def predict_proba(self, images, batch_size=32):
        N       = images.shape[0]
        results = []
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            batch = images[start:end]
            out   = self.forward(batch)
            results.append(out)
        return np.concatenate(results)

    def backward(self, grad_out):
        grad = grad_out
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad