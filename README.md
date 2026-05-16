# PixelToCaption

Implementation of CNN (image classification) and RNN/LSTM (image captioning) from scratch using NumPy — IF3270 Machine Learning, ITB 2025/2026.

This project builds and evaluates deep learning models at two levels:
- **Reference model** — built with Keras/TensorFlow
- **From-scratch model** — reimplemented using only NumPy, with weights transferred from the reference model

---

## Project Structure

```
PixelToCaption/
├── data/                     # Dataset files (not tracked by git)
│   ├── captions.txt          # Flickr8k captions
│   ├── flickr8k_features.npy # Pre-extracted InceptionV3 features
│   └── tokenizer.json        # Fitted tokenizer (generated on first run)
├── weights/                  # Saved model weights (not tracked by git)
├── src/
│   ├── shared/               # Shared utilities (preprocessing, layers, activations)
│   ├── cnn/                  # CNN image classification
│   │   ├── cnn.py            # From-scratch CNN model
│   │   ├── layers.py         # Conv2D, Pooling, Flatten layers
│   │   ├── train.py          # Keras reference training script
│   │   └── utils.py
│   ├── rnn/                  # RNN image captioning
│   │   ├── rnn.py            # From-scratch RNN cell
│   │   ├── layers.py         # RNNDecoder (multi-layer, from scratch)
│   │   ├── train.py          # Keras reference training script
│   │   └── notebook.ipynb    # Experiments & evaluation
│   └── lstm/                 # LSTM image captioning
│       ├── lstm.py           # From-scratch LSTM cell (BPTT)
│       ├── layers.py         # LSTMDecoder (multi-layer, from scratch)
│       ├── train.py          # Keras reference training script
│       └── notebook.ipynb    # Experiments & evaluation
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- (Optional) CUDA-compatible GPU for faster training

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd PixelToCaption

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

**CNN — Intel Image Classification**
1. Download from [Kaggle: Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
2. Extract and place under `data/intel/` with structure:
   ```
   data/intel/
   ├── seg_train/seg_train/
   └── seg_test/seg_test/
   ```

**RNN/LSTM — Flickr8k**
1. Download from [Kaggle: Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Place files in `data/`:
   ```
   data/
   ├── captions.txt
   └── Images/
   ```
3. Pre-extract InceptionV3 features (required before training):
   ```bash
   python src/shared/image_utils.py
   ```
   This generates `data/flickr8k_features.npy`.

---

## Running the Models

### CNN

```bash
# Train Keras reference model
python src/cnn/train.py

# Run full experiments (Keras vs. from-scratch, layer ablation)
jupyter notebook src/cnn/notebook.ipynb
```

### RNN

```bash
# Train Keras reference model
python src/rnn/train.py

# Run experiments
jupyter notebook src/rnn/notebook.ipynb
```

### LSTM

```bash
# Train Keras reference model
python src/lstm/train.py

# Run full experiments (multi-layer, hidden size, BLEU-4/METEOR evaluation)
jupyter notebook src/lstm/notebook.ipynb
```

---

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| CNN (image classification) | Accuracy, per-class accuracy |
| RNN/LSTM (image captioning) | BLEU-4 (corpus), METEOR, inference time |

---

## Team Members

| Name | NIM | Responsibilities |
|------|-----|-----------------|
| Jonathan Kenan Budianto | 13523139 | CNN from-scratch implementation, CNN experiments & evaluation |
| Mahesa Fadhillah Andre | 13523140 | LSTM from-scratch implementation, shared layers & preprocessing, LSTM experiments & evaluation |
| Muhammad Dzaky Atha Fadhilah | 18223124 | RNN from-scratch implementation, shared layers & preprocessing, RNN experiments & evaluation |

---

## Course

IF3270 — Machine Learning
Institut Teknologi Bandung, Semester 6, 2025/2026

## Attachment
https://drive.google.com/drive/folders/1hbqDSzFquf_ZXHe9QKcc2jIv2HXUtCPO?usp=drive_link
