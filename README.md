<div align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Keras-Sequential_CNN-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/Dataset-MNIST-6A1B9A?style=for-the-badge&logo=databricks&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-~99%25-00C853?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>

<br/><br/>

# ✍️ Handwritten Digit Recognition Model
### *CNN · MNIST · TensorFlow · Keras*

**A deep learning pipeline that classifies handwritten digits (0–9) with ~99% accuracy, built on the MNIST benchmark dataset using a Convolutional Neural Network — complete with preprocessing, training visualizations, and full model evaluation.**

<br/>

[![GitHub Stars](https://img.shields.io/github/stars/ibtesaamaslam/Handwritten-Digit-Recognition-Model?style=social)](https://github.com/ibtesaamaslam/Handwritten-Digit-Recognition-Model/stargazers)
&nbsp;
[![GitHub Forks](https://img.shields.io/github/forks/ibtesaamaslam/Handwritten-Digit-Recognition-Model?style=social)](https://github.com/ibtesaamaslam/Handwritten-Digit-Recognition-Model/network/members)
&nbsp;
[![GitHub Issues](https://img.shields.io/github/issues/ibtesaamaslam/Handwritten-Digit-Recognition-Model)](https://github.com/ibtesaamaslam/Handwritten-Digit-Recognition-Model/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Real-World Applications](#-real-world-applications)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [System Architecture](#-system-architecture)
- [Model Architecture](#-model-architecture)
- [Deep Learning Pipeline](#-deep-learning-pipeline)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Results & Performance](#-results--performance)
- [Visualizations](#-visualizations)
- [Roadmap & Future Enhancements](#-roadmap--future-enhancements)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

This project implements a **Convolutional Neural Network (CNN)** to recognize and classify handwritten digits (0–9) from the **MNIST dataset** — one of the most widely studied benchmarks in computer vision and deep learning. The model achieves **~99% test accuracy** using a compact yet powerful architecture consisting of two convolutional blocks, max pooling, dropout regularization, and a softmax classification head.

The entire pipeline — from raw image loading through preprocessing, training, evaluation, and visualization — is contained in a single, well-commented Jupyter Notebook designed to be accessible for beginners while remaining rigorous enough for practitioners.

> 💡 **Why CNN over a simple dense network?** Dense (fully connected) networks treat each pixel independently and lose spatial context. CNNs learn local patterns — edges, curves, stroke shapes — that are exactly what distinguish one digit from another. Even a shallow 2-layer CNN dramatically outperforms fully connected networks on image tasks.

---

## 🌍 Real-World Applications

| Application | How This Model Applies |
|-------------|----------------------|
| 📮 Postal code recognition | Sort mail by reading handwritten zip codes on envelopes |
| 🏦 Bank cheque processing | Automatically read dollar amounts written on paper cheques |
| 📝 Digitized form entry | Convert handwritten fields in scanned forms to structured data |
| 🧾 Invoice processing | Extract handwritten quantities or totals from paper invoices |
| 🏫 Automated exam grading | Read and score handwritten numeric answers |
| 📱 Mobile OCR | Foundational component in on-device handwriting recognition |

---

## 🧰 Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| [Python](https://www.python.org/) | 3.8+ | Core programming language |
| [TensorFlow](https://www.tensorflow.org/) | 2.x | Deep learning framework |
| [Keras](https://keras.io/) | (via TensorFlow) | High-level neural network API |
| [NumPy](https://numpy.org/) | 1.x | Array operations and numerical computing |
| [Matplotlib](https://matplotlib.org/) | 3.x | Training curves and image visualizations |
| [Seaborn](https://seaborn.pydata.org/) | 0.x | Confusion matrix heatmap visualization |
| [Scikit-learn](https://scikit-learn.org/) | 1.x | Classification report and confusion matrix utilities |
| [Jupyter Notebook](https://jupyter.org/) | — | Interactive development environment |

---

## 📊 Dataset

**Name:** MNIST Handwritten Digits Dataset  
**Source:** Loaded directly from `tensorflow.keras.datasets.mnist` — no external download required  
**Creator:** Yann LeCun, Corinna Cortes, Christopher Burges

| Split | Samples | Shape | Labels |
|-------|---------|-------|--------|
| Training | 60,000 images | 28 × 28 × 1 (grayscale) | 0–9 |
| Testing | 10,000 images | 28 × 28 × 1 (grayscale) | 0–9 |
| **Total** | **70,000 images** | — | 10 classes |

**Class distribution:** Balanced — approximately 6,000–7,000 samples per digit class in training.

**Pixel values:** Originally `uint8` in range `[0, 255]` → normalized to `float32` in range `[0.0, 1.0]` during preprocessing.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│       MNIST dataset — 28×28 grayscale pixel images               │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                               │
│  Normalize [0,255] → [0,1]  ·  Reshape → (28,28,1)              │
│  One-hot encode labels → (10,) binary vectors                    │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CNN MODEL (Sequential)                        │
│                                                                  │
│  Conv2D(32, 3×3, ReLU) → MaxPool(2×2)                           │
│  Conv2D(64, 3×3, ReLU) → MaxPool(2×2)                           │
│  Dropout(0.25)                                                   │
│  Flatten → Dense(128, ReLU) → Dropout(0.5)                      │
│  Dense(10, Softmax)                                              │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       TRAINING                                   │
│  Loss: Categorical Crossentropy  ·  Optimizer: Adam              │
│  Epochs: 10–20  ·  Batch size: 128  ·  Validation split: 10%    │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      EVALUATION                                  │
│  Test accuracy ~99%  ·  Confusion matrix  ·  Classification report│
│  Prediction visualizations  ·  Misclassified digit analysis      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture

```
CNN Sequential Model
│
├── Conv2D         →  32 filters, 3×3 kernel, ReLU, input: (28, 28, 1)
├── MaxPooling2D   →  Pool size: 2×2
│
├── Conv2D         →  64 filters, 3×3 kernel, ReLU
├── MaxPooling2D   →  Pool size: 2×2
│
├── Dropout        →  Rate: 0.25  (reduces overfitting in conv block)
├── Flatten        →  Converts 2D feature maps → 1D vector
│
├── Dense          →  128 neurons, ReLU
├── Dropout        →  Rate: 0.5   (stronger regularization before output)
│
└── Dense          →  10 neurons, Softmax  (one per digit class 0–9)
```

| Component | Value | Reason |
|-----------|-------|--------|
| Loss function | Categorical Crossentropy | Standard for multi-class classification |
| Optimizer | Adam | Adaptive learning rate; stable and fast convergence |
| Output activation | Softmax | Outputs a probability distribution over 10 classes |
| Hidden activation | ReLU | Prevents vanishing gradient; efficient training |
| Regularization | Dropout (0.25 + 0.5) | Prevents memorization of training samples |
| Input shape | (28, 28, 1) | Grayscale single-channel images |
| Output shape | (10,) | One probability score per digit class |

---

## 🔬 Deep Learning Pipeline

### Step 1 — Data Loading & Visualization

The MNIST dataset is loaded directly from Keras, requiring no external files:

```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train shape: (60000, 28, 28) — y_train shape: (60000,)
```

Sample images are visualized from the training set to inspect digit styles, stroke widths, and class diversity before any preprocessing.

### Step 2 — Preprocessing

Three operations prepare the raw data for the CNN:

```python
# 1. Normalize pixel values from [0, 255] to [0.0, 1.0]
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# 2. Reshape to add channel dimension (required by Conv2D)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# 3. One-hot encode labels (e.g., digit 3 → [0,0,0,1,0,0,0,0,0,0])
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)
```

### Step 3 — Model Construction

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### Step 4 — Compilation & Training

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1
)
```

### Step 5 — Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred_classes))
# Confusion matrix visualized as a Seaborn heatmap
```

---

## 📂 Project Structure

```
Handwritten-Digit-Recognition-Model/
│
├── Hand_Written_Digit_Recognition_Model.ipynb   # Main notebook — full pipeline
├── README.md                                    # Project documentation (this file)
├── requirements.txt                             # All Python dependencies
└── model/                                       # (optional) Saved model weights
    └── digit_recognition_model.keras
```

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Option A — Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/ibtesaamaslam/Handwritten-Digit-Recognition-Model.git
cd Handwritten-Digit-Recognition-Model

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### Option B — Quick pip install

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter
```

> **No dataset download needed.** MNIST is fetched automatically by Keras on first run and cached locally at `~/.keras/datasets/`.

---

## ▶ How to Run

1. Open **`Hand_Written_Digit_Recognition_Model.ipynb`** in Jupyter Notebook or JupyterLab.
2. Select **Kernel → Restart & Run All**.
3. The notebook will automatically download MNIST, preprocess the data, train the CNN, and display all evaluation outputs.
4. Training takes approximately **2–4 minutes** on a standard CPU.

> **GPU acceleration:** If you have a CUDA-compatible GPU and TensorFlow-GPU installed, training time drops to under 60 seconds.

---

## 📈 Results & Performance

| Metric | Value |
|--------|-------|
| Training accuracy | ~99.5% |
| Validation accuracy | ~99.2% |
| **Test accuracy** | **~99%** |
| Test loss | < 0.05 |
| Training time (CPU) | ~2–4 minutes |
| Training time (GPU) | < 60 seconds |
| Epochs | 10–20 |
| Batch size | 128 |

### Per-Class Performance (Typical)

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 0.99 | 0.99 | 0.99 |
| 1 | 0.99 | 0.99 | 0.99 |
| 2 | 0.99 | 0.99 | 0.99 |
| 3 | 0.99 | 0.99 | 0.99 |
| 4 | 0.99 | 0.99 | 0.99 |
| 5 | 0.99 | 0.98 | 0.99 |
| 6 | 0.99 | 0.99 | 0.99 |
| 7 | 0.99 | 0.99 | 0.99 |
| 8 | 0.98 | 0.99 | 0.99 |
| 9 | 0.99 | 0.98 | 0.99 |

> Most common confusions are between visually similar digit pairs: **4↔9**, **3↔8**, and **5↔6**.

---

## 📉 Visualizations

The notebook generates the following plots automatically:

| Visualization | Description |
|---------------|-------------|
| **Sample grid** | 5×5 grid of random training images with true labels |
| **Pixel intensity histogram** | Distribution of raw pixel values before normalization |
| **Training accuracy curve** | Train vs. validation accuracy over all epochs |
| **Training loss curve** | Train vs. validation loss over all epochs |
| **Confusion matrix heatmap** | 10×10 Seaborn heatmap of predicted vs. true labels |
| **Prediction samples** | Grid showing test images with predicted and true labels |
| **Misclassified digits** | Gallery of digits the model got wrong, for error analysis |

---

## 🗺 Roadmap & Future Enhancements

- [ ] **Streamlit web app** — Draw a digit in-browser and get a live prediction
- [ ] **Webcam / drawing pad input** — Real-time recognition via OpenCV or canvas
- [ ] **Advanced architectures** — Benchmark against LeNet-5, ResNet, and MobileNetV2
- [ ] **Data augmentation** — Random rotations, shifts, and zoom for robustness to noisy inputs
- [ ] **Model export** — Save as `.keras`, `.tflite` (mobile), and ONNX (cross-platform)
- [ ] **Batch prediction CLI** — Script to run predictions on a folder of digit images
- [ ] **Extended dataset** — Fine-tune on EMNIST (letters + digits) or custom handwriting
- [ ] **Explainability** — Grad-CAM heatmaps showing which pixels the model focuses on

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/[USER-NAME]/Handwritten-Digit-Recognition-Model.git

# 3. Create a feature branch
git checkout -b feature/add-streamlit-app

# 4. Make your changes and commit
git add .
git commit -m "feat: add Streamlit live digit recognition app"

# 5. Push and open a Pull Request
git push origin feature/add-streamlit-app
```

All contributions — bug fixes, new features, documentation improvements, additional visualizations — are appreciated.

---

## 👤 Author

<div align="center">

**Ibtesaam Aslam**

[![GitHub](https://img.shields.io/badge/GitHub-ibtesaamaslam-181717?style=for-the-badge&logo=github)](https://github.com/ibtesaamaslam)

*Machine Learning Engineer & Computer Vision Enthusiast*

</div>

---

## 📜 License

```
MIT License

Copyright (c) 2024 Ibtesaam Aslam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### License Permissions at a Glance

| Permission | Allowed? |
|-----------|----------|
| ✅ Commercial use | Yes |
| ✅ Modification | Yes |
| ✅ Distribution | Yes |
| ✅ Private use | Yes |
| ❌ Liability | No warranty provided |
| ❌ Trademark use | Not granted |

---

## 🙏 Acknowledgements

- **[Yann LeCun, Corinna Cortes & Christopher Burges](http://yann.lecun.com/exdb/mnist/)** — For creating and maintaining the MNIST dataset, the definitive benchmark for handwritten digit recognition.
- **[TensorFlow & Keras teams at Google](https://www.tensorflow.org/)** — For building an open-source deep learning ecosystem that makes CNN research accessible to everyone.
- **[Scikit-learn](https://scikit-learn.org/)** — For the evaluation utilities that complement Keras's training API.
- The **open-source Python community** — For Matplotlib, Seaborn, NumPy, and every library that powers this pipeline.

---

<div align="center">

**⭐ If this project was useful to you, please consider starring it on GitHub!**

[![Star on GitHub](https://img.shields.io/github/stars/ibtesaamaslam/Handwritten-Digit-Recognition-Model?style=social)](https://github.com/ibtesaamaslam/Handwritten-Digit-Recognition-Model)

*Made with ❤️ by [Ibtesaam Aslam](https://github.com/ibtesaamaslam)*

</div>
