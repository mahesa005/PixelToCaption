import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, classification_report


# evaluasi
def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def evaluate_keras_model(model, X, y_true, class_names=None):
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    return {
        'macro_f1': compute_macro_f1(y_true, y_pred),
        'report':   classification_report(y_true, y_pred, target_names=class_names, digits=4),
        'y_pred':   y_pred
    }

def evaluate_scratch_model(scratch_model, X, y_true, class_names=None):
    y_pred = scratch_model.predict_batch(X)
    return {
        'macro_f1': compute_macro_f1(y_true, y_pred),
        'report':   classification_report(y_true, y_pred, target_names=class_names, digits=4),
        'y_pred':   y_pred
    }


# plotting
def plot_history(history, title='Training History', save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{title} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True)

    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'],     label='Train Acc')
        axes[1].plot(history.history['val_accuracy'], label='Val Acc')
        axes[1].set_title(f'{title} — Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_f1_comparison(labels, f1_scores, title='Macro F1-Score Comparison', save_path=None):
    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    bars = plt.bar(labels, f1_scores, color='steelblue', edgecolor='black')
    plt.bar_label(bars, fmt='%.4f', padding=3)
    plt.title(title)
    plt.ylabel('Macro F1-Score')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()