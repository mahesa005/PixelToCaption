import numpy as np
from ..shared.activation_functions import relu, softmax


def get_activation(name):
    if name == 'relu':
        return relu
    elif name == 'softmax':
        return softmax
    elif name in ('linear', None):
        return lambda x: x
    else:
        raise ValueError(f"Unsupported activation: {name}")

class ActivationLayer:
    def __init__(self, activation):
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(x)

class Conv2DLayer:
    def __init__(self, keras_layer=None, kernel=None, bias=None,
                 activation='relu', strides=(1,1), padding='valid'):
        if keras_layer is not None:
            weights     = keras_layer.get_weights()
            self.kernel = weights[0]  # (kH, kW, C_in, C_out)
            self.bias   = weights[1]  # (C_out,)              
        else:
            self.kernel = kernel
            self.bias   = bias

        self.strides    = strides
        self.padding    = padding
        self.activation = get_activation(activation)
        self.kH, self.kW, self.C_in, self.C_out = self.kernel.shape

    def _pad(self, x):
        if self.padding == 'valid':
            return x
        H, W   = x.shape[0], x.shape[1]
        sH, sW = self.strides
        out_H  = int(np.ceil(H / sH))
        out_W  = int(np.ceil(W / sW))
        pH = max((out_H - 1) * sH + self.kH - H, 0)
        pW = max((out_W - 1) * sW + self.kW - W, 0)
        return np.pad(x, ((pH//2, pH - pH//2), (pW//2, pW - pW//2), (0,0)))

    def forward(self, x):
        x       = self._pad(x)
        H, W, _ = x.shape
        sH, sW  = self.strides
        out_H   = (H - self.kH) // sH + 1
        out_W   = (W - self.kW) // sW + 1
        out     = np.zeros((out_H, out_W, self.C_out))

        for i in range(out_H):
            for j in range(out_W):
                patch     = x[i*sH:i*sH+self.kH, j*sW:j*sW+self.kW, :]
                out[i, j] = np.einsum('hwc,hwck->k', patch, self.kernel) + self.bias

        return self.activation(out)


class LocallyConnected2DLayer:
    def __init__(self, keras_layer=None, kernel=None, bias=None,
                 activation='relu', strides=(1,1), padding='valid',
                 kernel_size=(3,3)):
        if keras_layer is not None:
            weights      = keras_layer.get_weights()
            self.kernel  = weights[0]  # (out_H*out_W, kH*kW*C_in, C_out) 
            self.bias    = weights[1]  # (out_H*out_W, C_out)             
        else:
            self.kernel  = kernel
            self.bias    = bias

        self.kH, self.kW = kernel_size
        self.strides     = strides
        self.padding     = padding
        self.activation  = get_activation(activation)
        self.C_out       = self.kernel.shape[2]

    def _pad(self, x):
        if self.padding == 'valid':
            return x
        H, W   = x.shape[0], x.shape[1]
        sH, sW = self.strides
        out_H  = int(np.ceil(H / sH))
        out_W  = int(np.ceil(W / sW))
        pH = max((out_H - 1) * sH + self.kH - H, 0)
        pW = max((out_W - 1) * sW + self.kW - W, 0)
        return np.pad(x, ((pH//2, pH - pH//2), (pW//2, pW - pW//2), (0,0)))

    def forward(self, x):
        x          = self._pad(x)
        H, W, C_in = x.shape
        sH, sW     = self.strides
        kH, kW     = self.kH, self.kW
        out_H      = (H - kH) // sH + 1
        out_W      = (W - kW) // sW + 1
        out        = np.zeros((out_H, out_W, self.C_out))

        for i in range(out_H):
            for j in range(out_W):
                patch     = x[i*sH:i*sH+kH, j*sW:j*sW+kW, :]
                flat      = patch.flatten()
                pos       = i * out_W + j
                out[i, j] = flat @ self.kernel[pos] + self.bias[pos]

        return self.activation(out)


class MaxPooling2DLayer:
    def __init__(self, keras_layer=None, pool_size=(2,2), strides=None):
        if keras_layer is not None:
            cfg            = keras_layer.get_config()
            self.pool_size = tuple(cfg['pool_size'])
            self.strides   = tuple(cfg['strides']) if cfg['strides'] else self.pool_size
        else:
            self.pool_size = pool_size
            self.strides   = strides if strides else pool_size

    def forward(self, x):
        H, W, C = x.shape
        pH, pW  = self.pool_size
        sH, sW  = self.strides
        out_H   = (H - pH) // sH + 1
        out_W   = (W - pW) // sW + 1
        out     = np.zeros((out_H, out_W, C))

        for i in range(out_H):
            for j in range(out_W):
                out[i, j] = np.max(x[i*sH:i*sH+pH, j*sW:j*sW+pW, :], axis=(0,1))

        return out


class AveragePooling2DLayer:
    def __init__(self, keras_layer=None, pool_size=(2,2), strides=None):
        if keras_layer is not None:
            cfg            = keras_layer.get_config()
            self.pool_size = tuple(cfg['pool_size'])
            self.strides   = tuple(cfg['strides']) if cfg['strides'] else self.pool_size
        else:
            self.pool_size = pool_size
            self.strides   = strides if strides else pool_size

    def forward(self, x):
        H, W, C = x.shape
        pH, pW  = self.pool_size
        sH, sW  = self.strides
        out_H   = (H - pH) // sH + 1
        out_W   = (W - pW) // sW + 1
        out     = np.zeros((out_H, out_W, C))

        for i in range(out_H):
            for j in range(out_W):
                out[i, j] = np.mean(x[i*sH:i*sH+pH, j*sW:j*sW+pW, :], axis=(0,1))

        return out


class GlobalMaxPooling2DLayer:
    def forward(self, x):
        return np.max(x, axis=(0,1))


class GlobalAveragePooling2DLayer:
    def forward(self, x):
        return np.mean(x, axis=(0,1))


class FlattenLayer:
    def forward(self, x):
        return x.flatten(order='C')