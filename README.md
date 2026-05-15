# Hate Speech Classification using Simple RNN

## Problem Statement
Detect hate speech in tweets using a binary classifier built with a Simple Recurrent Neural Network (RNN). Given a tweet, the model predicts whether it contains hate speech (label = 1) or not (label = 0).

---

## Dataset
- **File:** `imbalanced_data.csv`
- **Size:** 31,962 tweets
- **Columns used:** `tweet`, `label`
- **Classes:** 0 = Non-Hate, 1 = Hate (imbalanced distribution)

---

## Data Preprocessing Steps

1. **Lowercase conversion** — all tweet text is lowercased for consistency.
2. **Remove @mentions** — user handles (e.g., `@username`) are stripped using regex `@\w+`.
3. **Remove URLs** — HTTP/HTTPS links and `www.*` patterns are removed.
4. **Remove special characters & numbers** — only lowercase letters and spaces are retained (`[^a-z\s]` removed).
5. **Collapse extra spaces** — multiple consecutive spaces are reduced to a single space and leading/trailing whitespace is stripped.

---

## Vocabulary Creation Approach

- **Tokenizer:** Keras `Tokenizer` fitted on the training set only (no data leakage).
- **Vocabulary size:** Top **5,000** most frequent words.
- **Special tokens:**
  - `<PAD>` → index **0** (reserved by Keras for padding)
  - `<OOV>` → index **1** (out-of-vocabulary words are mapped here)
- Tweets are converted to integer sequences and padded/truncated to a fixed length of **25 tokens** (`padding='post'`, `truncating='post'`).

---

## Model Architecture

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| Input | — | (25,) | — |
| Embedding | `Embedding` | (25, 64) | 320,000 |
| RNN | `SimpleRNN` | (64,) | 8,256 |
| Output | `Dense` (sigmoid) | (1,) | 65 |

- **No LSTM or GRU** — only Simple RNN as required.
- **No pre-trained embeddings** — embedding weights are learned from scratch.

---

## Training Details

| Setting | Value |
|---|---|
| Loss | Binary Crossentropy |
| Optimizer | Adam |
| Epochs | 10 |
| Batch Size | 32 |
| Validation Split | 10% of training data |
| Train/Test Split | 80% / 20% (stratified) |

---

## Final Results

| Metric | Value |
|---|---|
| Test Accuracy | **94.93%** |
| Test Loss | 0.2010 |
| Non-Hate Precision | 0.97 |
| Non-Hate Recall | 0.98 |
| Non-Hate F1-Score | 0.97 |
| Hate Precision | 0.66 |
| Hate Recall | 0.57 |
| Hate F1-Score | 0.61 |
| Macro avg F1 | 0.79 |
| Weighted avg F1 | 0.95 |

**Confusion Matrix:**
```
                 Predicted Non-Hate   Predicted Hate
Actual Non-Hate        5814               131
Actual Hate             193               255
```

**Key Observations:**
- The model achieves 94.93% overall accuracy.
- It performs very well on the Non-Hate class (F1 = 0.97) and achieves meaningful Hate detection (F1 = 0.61).
- Out of 448 actual hate tweets, 255 were correctly detected (recall = 0.57).
- The dataset is heavily imbalanced (29,720 Non-Hate vs 2,242 Hate — a 93:7 ratio), which naturally biases models toward the majority class.
- The macro F1-score (0.79) reflects solid balanced performance across both classes.

---

## How to Run

Open `hate_speech_rnn_final.ipynb` in Google Colab, place `imbalanced_data.csv` in a Google Drive folder named `Hate Speech RNN`, and run all cells top to bottom.
