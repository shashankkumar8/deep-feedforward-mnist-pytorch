# 🧠 Deep Multilayer Perceptron for MNIST

**From Theory to Insight: A Production-Style Neural Network Built from Scratch**

![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-98.16%25-2E7D32?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white&style=for-the-badge)
![Parameters](https://img.shields.io/badge/Parameters-568K-0D47A1?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-2E7D32?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-F57C00?style=for-the-badge)

**A clean, heavily commented Deep Multilayer Perceptron (MLP) implemented purely in PyTorch that achieves 98.16% test accuracy on the MNIST benchmark while visually explaining every fundamental ML concept — exactly the foundations Amazon ML Summer School teaches.**

This project bridges classroom theory (Probability, Statistics, Linear Algebra, Optimization) with practical implementation and rich visualizations.

[📊 Results](#-results) • [🏗️ Architecture](#️-architecture) • [🔬 Design Choices](#-why-these-design-choices) • [▶️ How to Run](#️-how-to-run)

---

## ✨ Features

- **98.16% Test Accuracy** with only 0.08% train-val gap
- **Rich Visualizations** — training curves, confusion matrix, sample predictions with confidence, and learned weight filters
- **Heavy Mathematical Comments** explaining Neural Networks, Backpropagation, Gradient Descent, Linear Algebra, Softmax, Cross-Entropy, BatchNorm, and Dropout
- **Per-class accuracy analysis** with human-like error patterns
- **Production-style code** with clean structure, early stopping, learning rate scheduling, and model checkpointing
- **Zero GPU required** — runs efficiently on CPU in ~7.5 minutes

---

## 🎯 Results

| Metric                   | Value        | Insight                            |
| ------------------------ | ------------ | ---------------------------------- |
| **Test Accuracy**        | **98.16%**   | Strong generalization              |
| Test Loss                | 0.0654       | Low confidence error               |
| Best Validation Accuracy | **98.24%**   | Excellent convergence              |
| Train-Val Gap            | **0.08%**    | Near-perfect bias-variance balance |
| Total Parameters         | 568,970      | Efficient architecture             |
| Average Epoch Time       | 44.79s (CPU) | ~7.5 minutes total training        |

---

## 🏗️ Architecture

**Deep Feedforward Neural Network (MLP)**

| Layer    | Neurons | Activation | Regularization           | Weight Shape |
| -------- | ------- | ---------- | ------------------------ | ------------ |
| Input    | 784     | -          | Flattened 28×28 image    | -            |
| Hidden 1 | 512     | ReLU       | BatchNorm + Dropout(0.3) | (512 × 784)  |
| Hidden 2 | 256     | ReLU       | BatchNorm + Dropout(0.2) | (256 × 512)  |
| Hidden 3 | 128     | ReLU       | BatchNorm                | (128 × 256)  |
| Output   | 10      | Softmax    | -                        | (10 × 128)   |

**Total Learnable Parameters: 568,970**

Every forward pass follows `y = Wx + b`. Gradients are computed via backpropagation using the chain rule.

---

## 📊 Training Progress

| Epoch  | Train Loss | Train Acc  | Val Loss   | Val Acc    | Notes                  |
| ------ | ---------- | ---------- | ---------- | ---------- | ---------------------- |
| 1      | 0.2654     | 92.11%     | 0.1161     | 96.39%     | Rapid initial learning |
| 5      | 0.0780     | 97.49%     | 0.0749     | 97.71%     | Strong convergence     |
| **10** | **0.0480** | **98.43%** | **0.0597** | **98.24%** | Final best model       |

---

## 🎓 Per-Class Accuracy & Human-Like Errors

| Digit | Accuracy   | Difficulty  | Most Confused With |
| ----- | ---------- | ----------- | ------------------ |
| 1     | **99.21%** | Very Easy   | —                  |
| 2     | 98.93%     | Easy        | 7                  |
| 0     | 98.67%     | Easy        | —                  |
| **9** | **96.93%** | **Hardest** | **3 & 4**          |

> **🧠 Why is digit 9 the hardest?**  
> Digit 9 shares visual features with both 3 (curved bottom) and 4 (angular stroke patterns). This mirrors human visual cognition — the same ambiguities humans struggle with appear as classification errors in the neural network. This validates that the model learned **genuine visual structure** rather than simply memorizing pixel patterns.

---

## 🧮 ML Concepts Demonstrated

| Concept                 | Implementation                                          | Why It Matters                            |
| ----------------------- | ------------------------------------------------------- | ----------------------------------------- |
| Neural Networks         | 4-layer Deep MLP                                        | Universal function approximator           |
| Linear Algebra          | Matrix multiplications (`y = Wx + b`)                   | Foundation of every forward pass          |
| Backpropagation         | PyTorch autograd + chain rule                           | How networks actually learn               |
| Gradient Descent        | Adam optimizer                                          | Efficient loss minimization               |
| Softmax + Cross-Entropy | Output layer & loss function                            | Mathematically optimal for classification |
| Regularization          | BatchNorm + Dropout                                     | Prevents overfitting                      |
| Model Evaluation        | Confusion matrix, per-class accuracy, confidence scores | Real-world performance insight            |

---

## 🔬 Why These Design Choices?

**Q: Why Adam and not plain SGD?**  
**A:** Adam adapts learning rate per parameter using momentum + RMSprop. It converges ~3× faster than SGD on MNIST.

**Q: Why Batch Normalization?**  
**A:** Without BatchNorm, deeper layers suffer from internal covariate shift (changing input distributions). BatchNorm normalizes activations to mean=0, variance=1 per batch — leading to faster and more stable training.

**Q: Why Dropout rates of 0.3 and 0.2?**  
**A:** Layer 1 has the highest number of parameters (≈401K), so stronger regularization (0.3) is applied. Layer 2 uses 0.2 as representations become more abstract. No dropout before the output layer — it would hurt final decision-making.

**Q: Why Cross-Entropy instead of MSE?**  
**A:** MSE treats classification as regression and leads to vanishing gradients. Cross-Entropy directly measures probability divergence and is the maximum likelihood estimator for multi-class classification.

---

## 📁 Generated Visualizations

| File                       | Description                               | Insight                              |
| -------------------------- | ----------------------------------------- | ------------------------------------ |
| `training_curves.png`      | 2×2 grid: Loss & Accuracy (Train vs Val)  | Smooth convergence, minimal gap      |
| `confusion_matrix.png`     | 10×10 heatmap of predictions              | Clear view of digit 9 confusion      |
| `sample_predictions.png`   | 5×5 grid of test images with confidence % | Green = correct, Red = wrong         |
| `weight_visualization.png` | First-layer weights as 32 learned filters | Network learns stroke/edge detectors |

---

## ▶️ How to Run

````bash
# Clone the repository
git clone https://github.com/shashankkumar8/deep-mlp-mnist-pytorch.git
cd deep-mlp-mnist-pytorch

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py
All plots and the best model (mnist_model.pth) are automatically saved. Expected runtime: ~7.5 minutes on CPU.

📦 Dataset
MNIST — 70,000 handwritten digit images

Training: 48,000 | Validation: 12,000 | Test: 10,000
Input: 28×28 grayscale → flattened to 784 dimensions
Classes: 10 digits (0–9)
💡 Motivation & Learnings
This project transformed abstract concepts from my coursework into working, visual code. Seeing the first-layer weights emerge as stroke detectors was a breakthrough moment — it showed the network was truly learning visual features, not just memorizing.

The tiny 0.08% train-val gap gave me confidence in building reliable systems. However, I still crave deeper theoretical guidance on scaling these foundations to real-world problems at Amazon scale. This is why I am applying to Amazon ML Summer School 2026 — to learn directly from Amazon Scientists and strengthen my core ML understanding.

Built with curiosity, rigorous self-learning, and a desire to master foundational Machine Learning.

Author: Shashank Kumar
B.Tech Computer Science | ABES Engineering College | Expected 2028

GitHub: shashankkumar8
LinkedIn: linkedin.com/in/shashank-kumar-2b574228b
<div align="center"> ⭐ If this repository helped you understand neural networks better, please star it!
Built from scratch — no pretrained models, no shortcuts, pure gradient descent.

</div> ```
````
