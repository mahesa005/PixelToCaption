import numpy as np
from .layers import (
    Conv2DLayer, LocallyConnected2DLayer,
    MaxPooling2DLayer, AveragePooling2DLayer,
    GlobalMaxPooling2DLayer, GlobalAveragePooling2DLayer,
    FlattenLayer
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

    def predict_batch(self, images):
        return np.array([self.predict(img) for img in images])

    def predict_proba(self, images):
        return np.array([self.forward(img) for img in images])