import numpy as np

# Loss function Sparse Categorical Cross-Entropy untuk prediksi kata berikutnya dalam caption
class SparseCategoricalCrossEntropy:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true_index):
        """
        y_pred: numpy array shape (1, vocab_size) hasil dari Softmax
        y_true_index: indeks dari kata yang benar
        """
        # clipping untuk mencegah log(0) yang menghasilkan NaN
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15).ravel()

        # probabilitas tebakan pada kelas yang benar
        correct_class_probabilities = y_pred_clipped[y_true_index]
        
        # Loss: -log likelihood
        loss = -np.log(correct_class_probabilities)

        return loss


    def backward(self, y_pred, y_true_index):
        """
        Turunan Softmax + CCE: dZ = Y_pred - Y_true
        """
        d_logits = y_pred.copy().ravel()

        # karena y_true adalah 1 hanya pada y_true_index,
        # (y_pred - y_true) sama dengan mengurangi 1 di indeks yang benar
        d_logits[y_true_index] -= 1.0
        
        return d_logits