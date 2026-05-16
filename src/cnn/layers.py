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
        self.activation_name = activation
        self.activation      = get_activation(activation)

    def forward(self, x):
        self.x   = x
        self.out = self.activation(x)
        return self.out

    def backward(self, grad_out):
        if self.activation_name == 'relu':
            return grad_out * (self.x > 0).astype(float)
        elif self.activation_name == 'softmax':
            s   = self.out
            dot = np.sum(grad_out * s, axis=-1, keepdims=True)
            return s * (grad_out - dot)
        else:
            return grad_out


class Conv2DLayer:
    def __init__(self, keras_layer=None, kernel=None, bias=None,
                 activation='relu', strides=(1,1), padding='valid'):
        if keras_layer is not None:
            weights     = keras_layer.get_weights()
            self.kernel = weights[0]
            self.bias   = weights[1]
        else:
            self.kernel = kernel
            self.bias   = bias

        self.strides         = strides
        self.padding         = padding
        self.activation      = get_activation(activation)
        self.activation_name = activation
        self.kH, self.kW, self.C_in, self.C_out = self.kernel.shape
        self.dkernel         = None
        self.dbias           = None

    def _pad(self, x):
        if self.padding == 'valid':
            return x
        if x.ndim == 4:
            H, W = x.shape[1], x.shape[2]
        else:
            H, W = x.shape[0], x.shape[1]
        sH, sW = self.strides
        out_H  = int(np.ceil(H / sH))
        out_W  = int(np.ceil(W / sW))
        pH = max((out_H - 1) * sH + self.kH - H, 0)
        pW = max((out_W - 1) * sW + self.kW - W, 0)
        if x.ndim == 4:
            return np.pad(x, ((0,0), (pH//2, pH-pH//2), (pW//2, pW-pW//2), (0,0)))
        return np.pad(x, ((pH//2, pH-pH//2), (pW//2, pW-pW//2), (0,0)))

    def _forward_single(self, x):
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
        return out

    def forward(self, x):
        if x.ndim == 3:
            self.x_input = x
            out          = self._forward_single(x)
            self.pre_act = out
            return self.activation(out)
        self.x_input = x
        out          = np.stack([self._forward_single(x[n]) for n in range(x.shape[0])])
        self.pre_act = out
        return self.activation(out)

    def backward(self, grad_out):
        single = grad_out.ndim == 3
        if single:
            grad_out = grad_out[np.newaxis]
            x_input  = self.x_input[np.newaxis]
            pre_act  = self.pre_act[np.newaxis]
        else:
            x_input = self.x_input
            pre_act = self.pre_act

        if self.activation_name == 'relu':
            grad_out = grad_out * (pre_act > 0)

        N      = grad_out.shape[0]
        sH, sW = self.strides

        self.dkernel = np.zeros_like(self.kernel)
        self.dbias   = np.zeros_like(self.bias)
        dx_list      = []

        for n in range(N):
            x_n             = x_input[n]
            x_pad           = self._pad(x_n)
            H_pad, W_pad, _ = x_pad.shape
            dx_pad          = np.zeros_like(x_pad)
            out_H, out_W    = grad_out.shape[1], grad_out.shape[2]

            for i in range(out_H):
                for j in range(out_W):
                    patch         = x_pad[i*sH:i*sH+self.kH, j*sW:j*sW+self.kW, :]
                    g             = grad_out[n, i, j, :]
                    self.dkernel += np.einsum('hwc,k->hwck', patch, g)
                    self.dbias   += g
                    dx_pad[i*sH:i*sH+self.kH, j*sW:j*sW+self.kW, :] += \
                        np.einsum('k,hwck->hwc', g, self.kernel)

            if self.padding == 'same':
                H, W   = x_n.shape[0], x_n.shape[1]
                pH     = H_pad - H
                pW     = W_pad - W
                pt, pb = pH//2, pH - pH//2
                pl, pr = pW//2, pW - pW//2
                dx_pad = dx_pad[pt:H_pad-pb, pl:W_pad-pr, :]
            dx_list.append(dx_pad)

        dx = np.stack(dx_list)
        return dx[0] if single else dx


class LocallyConnected2DLayer:
    def __init__(self, keras_layer=None, kernel=None, bias=None,
                 activation='relu', strides=(1,1), padding='valid',
                 kernel_size=(3,3)):
        if keras_layer is not None:
            weights      = keras_layer.get_weights()
            self.kernel  = weights[0]
            self.bias    = weights[1]
        else:
            self.kernel  = kernel
            self.bias    = bias

        self.kH, self.kW     = kernel_size
        self.strides         = strides
        self.padding         = padding
        self.activation      = get_activation(activation)
        self.activation_name = activation
        self.C_out           = self.kernel.shape[2]
        self.dkernel         = None
        self.dbias           = None

    def _pad(self, x):
        if self.padding == 'valid':
            return x
        if x.ndim == 4:
            H, W = x.shape[1], x.shape[2]
        else:
            H, W = x.shape[0], x.shape[1]
        sH, sW = self.strides
        out_H  = int(np.ceil(H / sH))
        out_W  = int(np.ceil(W / sW))
        pH = max((out_H - 1) * sH + self.kH - H, 0)
        pW = max((out_W - 1) * sW + self.kW - W, 0)
        if x.ndim == 4:
            return np.pad(x, ((0,0), (pH//2, pH-pH//2), (pW//2, pW-pW//2), (0,0)))
        return np.pad(x, ((pH//2, pH-pH//2), (pW//2, pW-pW//2), (0,0)))

    def _forward_single(self, x):
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
        return out

    def forward(self, x):
        if x.ndim == 3:
            self.x_input = x
            out          = self._forward_single(x)
            self.pre_act = out
            return self.activation(out)
        self.x_input = x
        out          = np.stack([self._forward_single(x[n]) for n in range(x.shape[0])])
        self.pre_act = out
        return self.activation(out)

    def backward(self, grad_out):
        single = grad_out.ndim == 3
        if single:
            grad_out = grad_out[np.newaxis]
            x_input  = self.x_input[np.newaxis]
            pre_act  = self.pre_act[np.newaxis]
        else:
            x_input = self.x_input
            pre_act = self.pre_act

        if self.activation_name == 'relu':
            grad_out = grad_out * (pre_act > 0)

        N      = grad_out.shape[0]
        sH, sW = self.strides
        kH, kW = self.kH, self.kW

        self.dkernel = np.zeros_like(self.kernel)
        self.dbias   = np.zeros_like(self.bias)
        dx_list      = []

        for n in range(N):
            x_n             = x_input[n]
            x_pad           = self._pad(x_n)
            H_pad, W_pad, _ = x_pad.shape
            dx_pad          = np.zeros_like(x_pad)
            out_H, out_W    = grad_out.shape[1], grad_out.shape[2]

            for i in range(out_H):
                for j in range(out_W):
                    patch             = x_pad[i*sH:i*sH+kH, j*sW:j*sW+kW, :]
                    flat              = patch.flatten()
                    pos               = i * out_W + j
                    g                 = grad_out[n, i, j, :]
                    self.dkernel[pos] += np.outer(flat, g)
                    self.dbias[pos]   += g
                    dx_flat            = self.kernel[pos] @ g
                    dx_pad[i*sH:i*sH+kH, j*sW:j*sW+kW, :] += \
                        dx_flat.reshape(kH, kW, -1)

            if self.padding == 'same':
                H, W   = x_n.shape[0], x_n.shape[1]
                pH     = H_pad - H
                pW     = W_pad - W
                pt, pb = pH//2, pH - pH//2
                pl, pr = pW//2, pW - pW//2
                dx_pad = dx_pad[pt:H_pad-pb, pl:W_pad-pr, :]
            dx_list.append(dx_pad)

        dx = np.stack(dx_list)
        return dx[0] if single else dx


class MaxPooling2DLayer:
    def __init__(self, keras_layer=None, pool_size=(2,2), strides=None):
        if keras_layer is not None:
            cfg            = keras_layer.get_config()
            self.pool_size = tuple(cfg['pool_size'])
            self.strides   = tuple(cfg['strides']) if cfg['strides'] else self.pool_size
        else:
            self.pool_size = pool_size
            self.strides   = strides if strides else pool_size

    def _forward_single(self, x):
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

    def forward(self, x):
        if x.ndim == 3:
            self.x_input = x
            return self._forward_single(x)
        self.x_input = x
        return np.stack([self._forward_single(x[n]) for n in range(x.shape[0])])

    def backward(self, grad_out):
        single = grad_out.ndim == 3
        if single:
            grad_out = grad_out[np.newaxis]
            x_input  = self.x_input[np.newaxis]
        else:
            x_input = self.x_input

        N       = x_input.shape[0]
        pH, pW  = self.pool_size
        sH, sW  = self.strides
        dx_list = []

        for n in range(N):
            x        = x_input[n]
            dx       = np.zeros_like(x)
            out_H, out_W = grad_out.shape[1], grad_out.shape[2]

            for i in range(out_H):
                for j in range(out_W):
                    window  = x[i*sH:i*sH+pH, j*sW:j*sW+pW, :]
                    max_val = np.max(window, axis=(0,1), keepdims=True)
                    mask    = (window == max_val).astype(float)
                    mask   /= (mask.sum(axis=(0,1), keepdims=True) + 1e-8)
                    dx[i*sH:i*sH+pH, j*sW:j*sW+pW, :] += \
                        mask * grad_out[n, i, j, :]
            dx_list.append(dx)

        dx = np.stack(dx_list)
        return dx[0] if single else dx


class AveragePooling2DLayer:
    def __init__(self, keras_layer=None, pool_size=(2,2), strides=None):
        if keras_layer is not None:
            cfg            = keras_layer.get_config()
            self.pool_size = tuple(cfg['pool_size'])
            self.strides   = tuple(cfg['strides']) if cfg['strides'] else self.pool_size
        else:
            self.pool_size = pool_size
            self.strides   = strides if strides else pool_size

    def _forward_single(self, x):
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

    def forward(self, x):
        if x.ndim == 3:
            self.x_input = x
            return self._forward_single(x)
        self.x_input = x
        return np.stack([self._forward_single(x[n]) for n in range(x.shape[0])])

    def backward(self, grad_out):
        single = grad_out.ndim == 3
        if single:
            grad_out = grad_out[np.newaxis]
            x_input  = self.x_input[np.newaxis]
        else:
            x_input = self.x_input

        N       = x_input.shape[0]
        pH, pW  = self.pool_size
        sH, sW  = self.strides
        dx_list = []

        for n in range(N):
            dx       = np.zeros_like(x_input[n])
            out_H, out_W = grad_out.shape[1], grad_out.shape[2]
            for i in range(out_H):
                for j in range(out_W):
                    dx[i*sH:i*sH+pH, j*sW:j*sW+pW, :] += \
                        grad_out[n, i, j, :] / (pH * pW)
            dx_list.append(dx)

        dx = np.stack(dx_list)
        return dx[0] if single else dx


class GlobalMaxPooling2DLayer:
    def forward(self, x):
        self.x_input = x
        if x.ndim == 3:
            return np.max(x, axis=(0,1))
        return np.max(x, axis=(1,2))

    def backward(self, grad_out):
        single = grad_out.ndim == 1
        x      = self.x_input
        if single:
            x        = x[np.newaxis]
            grad_out = grad_out[np.newaxis]

        N, H, W, C = x.shape
        dx          = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                max_val        = np.max(x[n, :, :, c])
                mask           = (x[n, :, :, c] == max_val).astype(float)
                mask          /= (mask.sum() + 1e-8)
                dx[n, :, :, c] = mask * grad_out[n, c]

        return dx[0] if single else dx


class GlobalAveragePooling2DLayer:
    def forward(self, x):
        self.x_input = x
        if x.ndim == 3:
            return np.mean(x, axis=(0,1))
        return np.mean(x, axis=(1,2))

    def backward(self, grad_out):
        single = grad_out.ndim == 1
        x      = self.x_input
        if single:
            x        = x[np.newaxis]
            grad_out = grad_out[np.newaxis]

        N, H, W, C = x.shape
        dx          = np.zeros_like(x)
        for n in range(N):
            dx[n] = grad_out[n] / (H * W)

        return dx[0] if single else dx


class FlattenLayer:
    def forward(self, x):
        self.x_shape = x.shape
        if x.ndim == 3:
            return x.flatten(order='C')
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_out):
        return grad_out.reshape(self.x_shape)