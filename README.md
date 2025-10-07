🧠 MNIST Digit Classifier (NumPy From Scratch)

A lightweight deep learning pipeline built entirely with NumPy — no PyTorch, no TensorFlow. The project focuses on understanding core DL mechanics by implementing every major component manually.

✅ What’s Inside

Models

Softmax Regression

2-Layer MLP with ReLU/Sigmoid

Core Features

Forward & backward propagation (manual)

Batch data loader + train/val split

Cross-entropy loss + softmax

Accuracy tracking

Learning curve visualization

Optimizer

Vanilla SGD with L2 regularization

Bias terms excluded from weight decay

🗂️ Structure
models/         # Custom NN implementations
optim/          # SGD + regularization utilities
utils.py        # Data loading, batching, plotting
configs/        # YAML-based experiments
main.py         # Training loop
tests/          # Unit tests for each module

▶️ Run
python main.py --config configs/two_layer.yaml


Dataset setup:

cd data
sh get_data.sh
cd ..

🔍 Highlights

Manual backprop + param management

Config-driven experimentation

Minimal dependencies (NumPy + Matplotlib)

Training & validation metrics per epoch

Learning curves via Matplotlib

✅ Tech Stack

Python 3

NumPy

Matplotlib

YAML configs
