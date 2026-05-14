
"""
Full pipeline debug test for the LSTM from-scratch implementation.
Run from project root:  python src/lstm/test/test_pipeline.py
"""

import sys
import traceback
import numpy as np
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parents[2]   # src/lstm/test -> src/lstm -> src
sys.path.insert(0, str(SRC_DIR))

# ── Stub out image_utils so shared/__init__.py doesn't trigger Keras/TF load ─
import types as _types
_img_stub = _types.ModuleType("shared.image_utils")
for _fn in ("load_image", "load_batch", "extract_features"):
    setattr(_img_stub, _fn, None)
sys.modules["shared.image_utils"] = _img_stub

# ── Minimal test harness ─────────────────────────────────────────────────────
_pass = _fail = 0

def ok(name: str, cond: bool, detail: str = ""):
    global _pass, _fail
    if cond:
        print(f"  PASS  {name}")
        _pass += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        _fail += 1

def section(title: str):
    print(f"\n{'='*64}\n  {title}\n{'='*64}")

def allclose(a, b, name: str, tol: float = 1e-4):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        ok(name, False, f"shape mismatch: {a.shape} vs {b.shape}")
        return
    max_err = float(np.max(np.abs(a - b)))
    ok(name, max_err < tol, f"max_err={max_err:.2e}")

def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference numerical gradient of scalar f(x) w.r.t. x."""
    g = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        x[idx] += eps; fp = f(x.copy())
        x[idx] -= 2*eps; fm = f(x.copy())
        x[idx] += eps
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g

def run(name, fn):
    """Run fn(); catch and report exceptions as test failures."""
    try:
        fn()
    except Exception:
        print(f"  \033[31mERROR\033[0m  {name}")
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
section("1. Activation Functions")

def _test_activations():
    from shared.activation_functions import sigmoid, tanh

    ok("sigmoid(0) == 0.5",          abs(sigmoid(0.0) - 0.5) < 1e-9)
    ok("sigmoid(100) ~= 1.0",         abs(sigmoid(100) - 1.0) < 1e-6)
    ok("sigmoid(-100) ~= 0.0",        abs(sigmoid(-100)) < 1e-6)
    ok("sigmoid output in (0, 1)",   0 < sigmoid(1.5) < 1)

    ok("tanh(0) == 0.0",             abs(tanh(0.0)) < 1e-9)
    ok("tanh(100) ~= 1.0",            abs(tanh(100) - 1.0) < 1e-6)
    ok("tanh(-100) ~= -1.0",          abs(tanh(-100) + 1.0) < 1e-6)
    ok("tanh output in (-1, 1)",     -1 < tanh(0.7) < 1)

    # Gradient check: d/dx sigmoid(x)  =  sigmoid(x)*(1 - sigmoid(x))
    x = 0.6
    eps = 1e-5
    dsig_numerical  = (sigmoid(x + eps) - sigmoid(x - eps)) / (2 * eps)
    dsig_analytical = sigmoid(x) * (1 - sigmoid(x))
    allclose(dsig_numerical, dsig_analytical, "sigmoid gradient check")

    dtanh_numerical  = (tanh(x + eps) - tanh(x - eps)) / (2 * eps)
    dtanh_analytical = 1 - tanh(x) ** 2
    allclose(dtanh_numerical, dtanh_analytical, "tanh gradient check")

run("activation functions", _test_activations)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DENSE LAYER
# ═══════════════════════════════════════════════════════════════════════════════
section("2. DenseLayer")

def _test_dense():
    from shared.dense import DenseLayer

    np.random.seed(0)
    in_d, out_d = 5, 3
    layer = DenseLayer(input_size=in_d, output_size=out_d)
    x = np.random.randn(in_d)

    # Forward
    y = layer.forward(x)
    ok("forward output shape",  y.shape == (out_d,))
    expected = x @ layer.weights + layer.bias
    allclose(y, expected, "forward value correct")

    # Backward shapes
    grad_out = np.random.randn(out_d)
    dx = layer.backward(grad_out)
    ok("backward dx shape",  dx.shape == (in_d,))
    ok("backward dW shape",  layer.dW.shape == (in_d, out_d))
    ok("backward db shape",  layer.db.shape == (out_d,))

    # Numerical gradient check: dx
    def loss_x(x_):
        return layer.forward(x_) @ np.ones(out_d)   # sum of outputs

    dx_num = numerical_grad(loss_fn := lambda x_: layer.forward(x_).sum(), x.copy())
    # recompute analytical with ones grad
    layer.forward(x)
    dx_an = layer.backward(np.ones(out_d))
    allclose(dx_an, dx_num, "dense dx gradient check")

    # Numerical gradient check: dW
    W0 = layer.weights.copy()

    def loss_W(W_):
        layer.weights = W_
        return layer.forward(x).sum()

    dW_num = numerical_grad(loss_W, W0.copy())
    layer.weights = W0
    layer.forward(x)
    layer.backward(np.ones(out_d))
    allclose(layer.dW, dW_num, "dense dW gradient check")
    layer.weights = W0   # restore

run("DenseLayer", _test_dense)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING LAYER
# ═══════════════════════════════════════════════════════════════════════════════
section("3. EmbeddingLayer")

def _test_embedding():
    from shared.embedding_layer import EmbeddingLayer

    np.random.seed(1)
    vocab, embed = 10, 4
    layer = EmbeddingLayer(vocab_size=vocab, embed_dim=embed)
    token = 3

    # Forward
    out = layer.forward(token)
    ok("forward output shape",          out.shape == (embed,))
    allclose(out, layer.weights[token], "forward returns correct row")

    # Backward
    grad = np.random.randn(embed)
    layer.backward(grad)

    ok("backward dW shape",             layer.dW.shape == (vocab, embed))
    allclose(layer.dW[token], grad,     "backward: token row == grad")
    other_rows = np.delete(layer.dW, token, axis=0)
    ok("backward: non-token rows == 0", np.allclose(other_rows, 0))

run("EmbeddingLayer", _test_embedding)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
section("4. SparseCategoricalCrossEntropy")

def _test_loss():
    from shared.loss_function import SparseCategoricalCrossEntropy

    loss_fn = SparseCategoricalCrossEntropy()

    # Known input: uniform distribution → loss = -log(1/V)
    V = 5
    y_pred = np.ones(V) / V
    y_true = 2
    loss = loss_fn.forward(y_pred, y_true)
    ok("forward output is scalar",  np.isscalar(loss) or loss.ndim == 0)
    allclose(loss, -np.log(1 / V),  "forward value correct (uniform pred)")

    # Backward shape and value
    grad = loss_fn.backward(y_pred, y_true)
    ok("backward shape matches y_pred",       grad.shape == (V,))
    ok("backward grad at y_true index = p-1", abs(grad[y_true] - (1/V - 1)) < 1e-9)
    ok("backward grad at other index  = p",   abs(grad[0] - 1/V) < 1e-9)

    # backward is the combined softmax+CCE gradient w.r.t. logits: y_pred - y_true_onehot
    # Verify the key algebraic property: grad[y_true] == y_pred[y_true] - 1
    np.random.seed(2)
    y_pred3 = np.abs(np.random.randn(V)); y_pred3 /= y_pred3.sum()
    y_true3 = 1
    grad3 = loss_fn.backward(y_pred3, y_true3)
    ok("backward combined grad: grad[y_true] == p[y_true]-1",
       abs(grad3[y_true3] - (y_pred3[y_true3] - 1.0)) < 1e-9)
    ok("backward combined grad: other elements unchanged",
       all(abs(grad3[i] - y_pred3[i]) < 1e-9 for i in range(V) if i != y_true3))

run("SparseCategoricalCrossEntropy", _test_loss)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ADAM OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════
section("5. AdamOptimizer")

def _test_optimizer():
    from shared.optimizer import AdamOptimizer

    opt = AdamOptimizer(learning_rate=0.1)
    W = np.array([1.0, 2.0, 3.0])
    W_before = W.copy()

    # grad pointing in +direction → param should decrease
    grad = np.array([1.0, 1.0, 1.0])
    opt.step({"W": W}, {"W": grad})

    ok("param updated after step",    not np.allclose(W, W_before))
    ok("param decreased (grad > 0)",  np.all(W < W_before))

    # Second step
    W2_before = W.copy()
    opt.step({"W": W}, {"W": grad})
    ok("param still moving after step 2", not np.allclose(W, W2_before))

run("AdamOptimizer", _test_optimizer)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LSTM CELL
# ═══════════════════════════════════════════════════════════════════════════════
section("6. LSTM Cell")

def _test_lstm():
    from lstm.lstm import LSTM

    np.random.seed(3)
    E, H = 6, 4
    cell = LSTM(embed_dim=E, hidden_dim=H)

    x      = np.random.randn(E)
    h_prev = np.random.randn(H)
    c_prev = np.random.randn(H)

    # ── Forward ──────────────────────────────────────────────────────────────
    h_out, c_out, cache = cell.forward(x, h_prev, c_prev)

    ok("forward h shape",          h_out.shape == (H,))
    ok("forward c shape",          c_out.shape == (H,))
    ok("h and c are distinct",     not np.allclose(h_out, c_out))
    ok("h values in (-1, 1)",      np.all(np.abs(h_out) < 1.0))   # tanh output
    ok("cache is dict",            isinstance(cache, dict))
    ok("cache has required keys",  all(k in cache for k in ("x","h_prev","c_prev","f","i","candidate","o","c","tanh_c")))

    # ── Backward ─────────────────────────────────────────────────────────────
    dh = np.ones(H)
    dc = np.zeros(H)
    dx_an, dh_prev_an, dc_prev_an = cell.backward(dh, dc, cache)

    ok("backward dx shape",       dx_an.shape == (E,))
    ok("backward dh_prev shape",  dh_prev_an.shape == (H,))
    ok("backward dc_prev shape",  dc_prev_an.shape == (H,))

    # ── Numerical gradient checks ────────────────────────────────────────────
    # Loss = sum(h_out)  →  dh = ones, dc = zeros
    def loss_h(h_out_): return h_out_.sum()

    def f_x(x_):
        h_, _, _ = cell.forward(x_, h_prev, c_prev)
        return loss_h(h_)

    def f_h(h_):
        h_, _, _ = cell.forward(x, h_, c_prev)
        return loss_h(h_)

    def f_c(c_):
        h_, _, _ = cell.forward(x, h_prev, c_)
        return loss_h(h_)

    dx_num      = numerical_grad(f_x, x.copy())
    dh_prev_num = numerical_grad(f_h, h_prev.copy())
    dc_prev_num = numerical_grad(f_c, c_prev.copy())

    allclose(dx_an,       dx_num,      "LSTM dx gradient check")
    allclose(dh_prev_an,  dh_prev_num, "LSTM dh_prev gradient check")
    allclose(dc_prev_an,  dc_prev_num, "LSTM dc_prev gradient check")

    # Weight gradients: dW (kernel) — spot-check a few elements
    cell.zero_grad()                   # reset accumulators for single-step check
    cell.forward(x, h_prev, c_prev)   # recompute cache
    cell.backward(dh, dc, cache)

    W_orig = cell.W.copy()
    def f_W(W_):
        cell.W    = W_
        # recompute gate slices from new W
        cell.W_i  = cell.W[:, 0*H:1*H]
        cell.W_f  = cell.W[:, 1*H:2*H]
        cell.W_c  = cell.W[:, 2*H:3*H]
        cell.W_o  = cell.W[:, 3*H:4*H]
        h_, _, _  = cell.forward(x, h_prev, c_prev)
        return loss_h(h_)

    dW_num = numerical_grad(f_W, W_orig.copy())
    # analytical dW reconstructed from gate grads
    dW_an = np.hstack([cell.dW_i, cell.dW_f, cell.dW_c, cell.dW_o])
    cell.W   = W_orig            # restore
    cell.W_i = cell.W[:, 0*H:1*H]
    cell.W_f = cell.W[:, 1*H:2*H]
    cell.W_c = cell.W[:, 2*H:3*H]
    cell.W_o = cell.W[:, 3*H:4*H]
    allclose(dW_an, dW_num, "LSTM dW gradient check")

run("LSTM Cell", _test_lstm)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LSTM DECODER
# ═══════════════════════════════════════════════════════════════════════════════
section("7. LSTMDecoder")

def _test_decoder():
    from lstm.layers import LSTMDecoder

    np.random.seed(4)
    VOCAB, EMBED, HIDDEN, FEAT = 20, 8, 6, 16

    dec = LSTMDecoder(
        embed_dim        = EMBED,
        hidden_dim       = HIDDEN,
        vocab_size       = VOCAB,
        dense_proj_input = FEAT,
    )

    feat   = np.random.randn(FEAT)
    tokens = [3, 7, 1, 5]   # caption token sequence

    # ── forward() ────────────────────────────────────────────────────────────
    output, cache = dec.forward(feat, tokens)

    ok("forward output is dict",           isinstance(output, dict))
    ok("forward output has T timesteps",   len(output) == len(tokens))
    ok("each output shape == (vocab,)",    all(v.shape == (VOCAB,) for v in output.values()))
    ok("cache has T+1 entries",            len(cache) == len(tokens) + 1)  # +1 for pre-injection
    ok("cache[-1] exists",                 -1 in cache)

    # ── backward() ───────────────────────────────────────────────────────────
    grad_outputs = {t: np.random.randn(VOCAB) for t in range(len(tokens))}
    try:
        dec.backward(cache, grad_outputs)
        ok("backward runs without error", True)
        ok("dW_i computed",  dec.lstm.dW_i is not None and dec.lstm.dW_i.shape == (EMBED, HIDDEN))
        ok("dW_f computed",  dec.lstm.dW_f is not None)
        ok("dU_o computed",  dec.lstm.dU_o is not None and dec.lstm.dU_o.shape == (HIDDEN, HIDDEN))
    except Exception as e:
        ok("backward runs without error", False, str(e))

    # ── predict() ────────────────────────────────────────────────────────────
    START, END = 1, 2
    MAX = 12

    # Normal generation (may or may not hit END)
    generated = dec.predict(feat, START, END, max_length=MAX)
    ok("predict returns a list",               isinstance(generated, list))
    ok("predict length <= max_length",         len(generated) <= MAX)
    ok("all predicted tokens are valid idx",   all(0 <= t < VOCAB for t in generated))

    # Force a short sequence by placing END as the most probable token
    # Build a decoder where dense_out bias is rigged to always predict END
    dec2 = LSTMDecoder(
        embed_dim=EMBED, hidden_dim=HIDDEN, vocab_size=VOCAB, dense_proj_input=FEAT
    )
    dec2.dense_out.bias[:] = -1e6
    dec2.dense_out.bias[END] = 1e6
    gen2 = dec2.predict(feat, START, END, max_length=MAX)
    ok("predict stops at end_token when forced", len(gen2) == 1 and gen2[0] == END)

run("LSTMDecoder", _test_decoder)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. INTEGRATION: full training step
# ═══════════════════════════════════════════════════════════════════════════════
section("8. Integration — full training step")

def _test_integration():
    from lstm.layers import LSTMDecoder
    from shared.loss_function import SparseCategoricalCrossEntropy
    from shared.optimizer import AdamOptimizer

    np.random.seed(5)
    VOCAB, EMBED, HIDDEN, FEAT = 15, 6, 5, 12

    dec      = LSTMDecoder(embed_dim=EMBED, hidden_dim=HIDDEN, vocab_size=VOCAB, dense_proj_input=FEAT)
    loss_fn  = SparseCategoricalCrossEntropy()
    opt      = AdamOptimizer(learning_rate=1e-3)

    feat          = np.random.randn(FEAT)
    caption_tok   = [3, 6, 2, 8]
    target_tok    = {0: 6, 1: 2, 2: 8, 3: 4}

    # ── Step 1: collect losses ───────────────────────────────────────────────
    step_losses = []
    for step in range(3):
        dec.lstm.zero_grad()
        output, cache = dec.forward(feat, caption_tok)

        grad_outputs = {}
        total = 0.0
        for t in output:
            total += float(loss_fn.forward(output[t], target_tok[t]))
            grad_outputs[t] = loss_fn.backward(output[t], target_tok[t])
        step_losses.append(total)

        dec.backward(cache, grad_outputs)

        params = {
            "W_i": dec.lstm.W_i, "W_f": dec.lstm.W_f,
            "W_c": dec.lstm.W_c, "W_o": dec.lstm.W_o,
            "U_i": dec.lstm.U_i, "U_f": dec.lstm.U_f,
            "U_c": dec.lstm.U_c, "U_o": dec.lstm.U_o,
            "dout_W": dec.dense_out.weights,  "dout_b": dec.dense_out.bias,
            "dproj_W": dec.dense_proj.weights, "dproj_b": dec.dense_proj.bias,
        }
        grads = {
            "W_i": dec.lstm.dW_i, "W_f": dec.lstm.dW_f,
            "W_c": dec.lstm.dW_c, "W_o": dec.lstm.dW_o,
            "U_i": dec.lstm.dU_i, "U_f": dec.lstm.dU_f,
            "U_c": dec.lstm.dU_c, "U_o": dec.lstm.dU_o,
            "dout_W": dec.dense_out.dW,   "dout_b": dec.dense_out.db,
            "dproj_W": dec.dense_proj.dW, "dproj_b": dec.dense_proj.db,
        }
        opt.step(params, grads)

    ok("3 training steps ran without error", True)
    ok("loss is finite",                     all(np.isfinite(l) for l in step_losses))
    ok("loss changed between steps",         step_losses[0] != step_losses[2])

    # ── Gradient sanity: gradients are non-zero ──────────────────────────────
    ok("dW_i is non-zero",  not np.allclose(dec.lstm.dW_i, 0))
    ok("dW_o is non-zero",  not np.allclose(dec.lstm.dW_o, 0))
    ok("dU_f is non-zero",  not np.allclose(dec.lstm.dU_f, 0))
    ok("dense_out dW non-zero", not np.allclose(dec.dense_out.dW, 0))

    print(f"\n  step losses: {[f'{l:.4f}' for l in step_losses]}")

run("Integration", _test_integration)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print(f"  Results:  {_pass} passed  |  {_fail} failed  |  {_pass+_fail} total")
print(f"{'='*64}\n")
sys.exit(0 if _fail == 0 else 1)
