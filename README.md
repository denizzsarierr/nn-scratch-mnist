# MNIST Neural Network from Scratch

A fully connected neural network built from scratch using **NumPy** to classify handwritten digits from the MNIST dataset.

No deep learning frameworks (TensorFlow / PyTorch) were used.

---

## Features
- Forward and backpropagation implemented manually
- ReLU (hidden layers) + Softmax (output)
- Mini-batch gradient descent
- Hyperparameter tuning (grid search)
- Model saving and loading using `.npy`

---

## Model

**Architecture:**  
784 → 256 → 128 → 10  

**Activation Functions:**  
- ReLU (hidden layers)  
- Softmax (output layer)  

**Parameter Initialization:**  
- He initialization for weights  
- Biases initialized to zero  

**Training:**  
- Mini-batch gradient descent  

**Loss Function:**  
- Categorical Cross-Entropy  

---

## Results

- Test Accuracy: **98.0%**
- Dataset: MNIST (70,000 samples)
- Train/Test Split: 80% / 20%
- Training Time: ~1–2 minutes  

**Best Hyperparameters:**
- Architecture: 784 → 256 → 128 → 10  
- Epochs: 30  
- Learning Rate: 0.02  
- Batch Size: 16  

---

## ⚙️ Usage

```bash
python tune.py      # hyperparameter tuning
python train.py     # train final model
python evaluate.py  # evaluate saved model
