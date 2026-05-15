import numpy as np
from shared.activation_functions import tanh, sigmoid

class LSTM:
    """
    Single LSTM cell with forward and backward (BPTT) support.

    Weight layout (all gates concatenated along axis 1):
        W: (embed_dim, 4 * hidden_dim)  -- input weights
        U: (hidden_dim, 4 * hidden_dim) -- recurrent weights
        b: (4 * hidden_dim,)            -- biases
    Gate order: [i | f | c | o]
    """
    def __init__(self, keras_layer=None, W=None, U=None, b=None, embed_dim=None, hidden_dim=None):
        """
        Initialize weights from a Keras layer, explicit arrays, or random values.

        Args:
            keras_layer: Keras LSTM layer to copy weights from.
            W, U, b: explicit weight arrays (all three required together).
            embed_dim: input embedding dimension (used for random init).
            hidden_dim: number of hidden units (used for random init).
        """
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
        self.embed_dim = self.W.shape[0]
        self.zero_grad()

    def forward(self, x, h_prev, c_prev):
        """
        Run one LSTM timestep.

        Args:
            x: input vector, shape (embed_dim,)
            h_prev: previous hidden state, shape (hidden_dim,)
            c_prev: previous cell state, shape (hidden_dim,)

        Returns:
            h: new hidden state, shape (hidden_dim,)
            c: new cell state, shape (hidden_dim,)
            cache: dict of intermediate values needed for backward.
        """
        f = self.forget_gate(x, h_prev)
        i = self.input_gate(x, h_prev)
        candidate = self.candidate(x, h_prev)
        o = self.output_gate(x, h_prev)
        c = c_prev * f + (candidate * i)
        tanh_c = tanh(c)
        h = tanh_c * o

        # cache the current timestep for later use in backprop
        cache = {
            'x':         x,           # for dW_f, dW_i, dW_c, dW_o
            'h_prev':    h_prev,         # for dU_f, dU_i, dU_c, dU_o
            'c_prev':    c_prev,         # for dc/df
            'f':         f,           # for sigmoid derivative
            'i':         i,
            'candidate': candidate,
            'o':         o,
            'c':         c,
            'tanh_c':    tanh_c,         # for tanh derivative
        }
        return h, c, cache

    def backward(self, dh, dc, cache):
        """
        Backpropagate through one LSTM timestep and accumulate weight gradients.

        Gradients are accumulated into dW_*, dU_*, db_* (reset via zero_grad).

        Args:
            dh: gradient of loss w.r.t. hidden state h, shape (hidden_dim,)
            dc: gradient of loss w.r.t. cell state c (from next timestep), shape (hidden_dim,)
            cache: dict returned by the corresponding forward call.

        Returns:
            dx: gradient w.r.t. input x, shape (embed_dim,)
            dh_prev: gradient w.r.t. h_prev, shape (hidden_dim,)
            dc_prev: gradient w.r.t. c_prev, shape (hidden_dim,)
        """
        x         = cache['x']
        h_prev    = cache['h_prev']
        c_prev    = cache['c_prev']
        f         = cache['f']
        i         = cache['i']
        candidate = cache['candidate']
        o         = cache['o']
        c         = cache['c']
        tanh_c    = cache['tanh_c']

        # step 1 — gradient from h = tanh_c * o
        d_tanh_c = dh * o
        d_o      = dh * tanh_c

        # step 2 — gradient through cell state
        dc_total = d_tanh_c * (1 - tanh_c**2) + dc

        # step 3 — gradient through each gate (chain rule through sigmoid/tanh)
        d_pre_o = d_o * o * (1 - o)
        d_pre_f = (dc_total * c_prev) * f * (1 - f)
        d_pre_i = (dc_total * candidate) * i * (1 - i)
        d_pre_c = (dc_total * i) * (1 - candidate**2)

        # step 4 — weight gradients (accumulated across timesteps; reset by zero_grad())
        self.dW_o += x.reshape(-1,1) @ d_pre_o.reshape(1,-1)
        self.dW_f += x.reshape(-1,1) @ d_pre_f.reshape(1,-1)
        self.dW_i += x.reshape(-1,1) @ d_pre_i.reshape(1,-1)
        self.dW_c += x.reshape(-1,1) @ d_pre_c.reshape(1,-1)

        self.dU_o += h_prev.reshape(-1,1) @ d_pre_o.reshape(1,-1)
        self.dU_f += h_prev.reshape(-1,1) @ d_pre_f.reshape(1,-1)
        self.dU_i += h_prev.reshape(-1,1) @ d_pre_i.reshape(1,-1)
        self.dU_c += h_prev.reshape(-1,1) @ d_pre_c.reshape(1,-1)

        self.db_o += d_pre_o
        self.db_f += d_pre_f
        self.db_i += d_pre_i
        self.db_c += d_pre_c

        # step 5 — pass gradients to previous timestep
        dx      = d_pre_o @ self.W_o.T + d_pre_f @ self.W_f.T + d_pre_i @ self.W_i.T + d_pre_c @ self.W_c.T
        dh_prev = d_pre_o @ self.U_o.T + d_pre_f @ self.U_f.T + d_pre_i @ self.U_i.T + d_pre_c @ self.U_c.T
        dc_prev = dc_total * f

        return dx, dh_prev, dc_prev


    def forget_gate(self, x, h_prev):
        """Compute forget gate: sigmoid(x @ W_f + h_prev @ U_f + b_f)."""
        output_x = x @ self.W_f
        output_h = h_prev @ self.U_f
        output = sigmoid(output_x + output_h + self.b_f)
        return output
    
    def input_gate(self, x, h_prev):
        """Compute input gate: sigmoid(x @ W_i + h_prev @ U_i + b_i)."""
        output_x = x @ self.W_i
        output_h = h_prev @ self.U_i
        output = sigmoid(output_x + output_h + self.b_i)
        return output
    
    def candidate(self, x, h_prev):
        """Compute candidate cell: tanh(x @ W_c + h_prev @ U_c + b_c)."""
        output_x = x @ self.W_c
        output_h = h_prev @ self.U_c
        output = tanh(output_x + output_h + self.b_c)
        return output

    def output_gate(self, x, h_prev):
        """Compute output gate: sigmoid(x @ W_o + h_prev @ U_o + b_o)."""
        output_x = x @ self.W_o
        output_h = h_prev @ self.U_o
        output = sigmoid(output_x + output_h + self.b_o)
        return output

    def zero_grad(self):
        """Reset all weight gradient accumulators to zero. Call before each new sample."""
        self.dW_o = np.zeros_like(self.W_o)
        self.dW_f = np.zeros_like(self.W_f)
        self.dW_i = np.zeros_like(self.W_i)
        self.dW_c = np.zeros_like(self.W_c)
        self.dU_o = np.zeros_like(self.U_o)
        self.dU_f = np.zeros_like(self.U_f)
        self.dU_i = np.zeros_like(self.U_i)
        self.dU_c = np.zeros_like(self.U_c)
        self.db_o = np.zeros_like(self.b_o)
        self.db_f = np.zeros_like(self.b_f)
        self.db_i = np.zeros_like(self.b_i)
        self.db_c = np.zeros_like(self.b_c)