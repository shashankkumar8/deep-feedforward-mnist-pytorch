# 🧠 Deep Feedforward Neural Network

## Handwritten Digit Recognition — MNIST

![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.16%25-2E7D32)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Parameters](https://img.shields.io/badge/Parameters-568%2C970-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Results

| Metric            | Score        |
| ----------------- | ------------ |
| **Test Accuracy** | **98.16%**   |
| Test Loss         | 0.0654       |
| Best Val Accuracy | 98.24%       |
| Train-Val Gap     | 0.08%        |
| Total Parameters  | 568,970      |
| Avg Epoch Time    | 44.79s (CPU) |

## 🏗️ Architecture

Input (784) ← 28×28 pixel image flattened
↓ Linear(784→512) + BatchNorm + ReLU + Dropout(0.3)
Hidden (512)
↓ Linear(512→256) + BatchNorm + ReLU + Dropout(0.2)
Hidden (256)
↓ Linear(256→128) + BatchNorm + ReLU
Hidden (128)
↓ Linear(128→10)
Output (10) → Softmax → Predicted Digit (0-9)

text

## 📊 Training Progress

| Epoch  | Train Loss | Train Acc  | Val Loss   | Val Acc    |
| ------ | ---------- | ---------- | ---------- | ---------- |
| 1      | 0.2654     | 92.11%     | 0.1161     | 96.39%     |
| 3      | 0.1062     | 96.58%     | 0.0862     | 97.38%     |
| 5      | 0.0780     | 97.49%     | 0.0749     | 97.71%     |
| 7      | 0.0619     | 97.99%     | 0.0709     | 97.86%     |
| 9      | 0.0541     | 98.28%     | 0.0636     | 98.04%     |
| **10** | **0.0480** | **98.43%** | **0.0597** | **98.24%** |

## 🎓 Per-Class Test Accuracy

| Digit | Accuracy | Digit | Accuracy |
| ----- | -------- | ----- | -------- |
| 0     | 98.67%   | 5     | 97.98%   |
| 1     | 99.21%   | 6     | 98.33%   |
| 2     | 98.93%   | 7     | 97.67%   |
| 3     | 98.02%   | 8     | 97.74%   |
| 4     | 97.96%   | 9     | 96.93%   |

## 🧮 Key ML Concepts Demonstrated

| Concept                 | Implementation                    |
| ----------------------- | --------------------------------- |
| **Neural Networks**     | 4-layer feedforward: y = Wx + b   |
| **Linear Algebra**      | Matrix multiplications per layer  |
| **Backpropagation**     | Chain rule via PyTorch autograd   |
| **Gradient Descent**    | Adam optimizer (lr=0.001)         |
| **Cross-Entropy Loss**  | −log(P(true class))               |
| **Softmax**             | Converts logits to probabilities  |
| **Dropout**             | p=0.3, 0.2 — prevents overfitting |
| **Batch Normalization** | Stabilizes training               |
| **Bias-Variance**       | Train-Val gap = 0.08%             |
| **Model Evaluation**    | Accuracy, Confusion Matrix        |

## 📁 Output Files

| File                       | Description                         |
| -------------------------- | ----------------------------------- |
| `training_curves.png`      | Loss + Accuracy curves (4 subplots) |
| `confusion_matrix.png`     | 10×10 digit confusion matrix        |
| `sample_predictions.png`   | 25 test samples with confidence     |
| `weight_visualization.png` | 32 learned first-layer filters      |
| `mnist_model.pth`          | Saved best model weights            |

## ▶️ How to Run

```bash
pip install torch torchvision matplotlib
python train.py
📦 Dataset
MNIST — 70,000 handwritten digit images
Training: 48,000 | Validation: 12,000 | Test: 10,000
Input: 28×28 grayscale → flattened to 784-dim vector
Classes: 10 digits (0–9)
🔑 Key Finding
Digit 1 achieved highest accuracy (99.21%) due to
its distinctive vertical stroke. Digit 9 was
hardest to classify (96.93%), most often confused
with 3 and 4 — consistent with human visual
ambiguity between these digit shapes.
```
