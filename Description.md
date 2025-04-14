
## ✍️ Handwritten Digit Recognition Model

This project builds an efficient **deep learning model to recognize handwritten digits (0–9)** from the popular **MNIST dataset**. Using Convolutional Neural Networks (CNNs), this model can accurately classify digits written by hand — a foundational task in computer vision with real-world applications like postal code recognition, bank check processing, and digitized form entries.

---

### 🚀 Project Highlights

- **Dataset Used**: [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/)
- **Tech Stack**:
  - Language: Python
  - Framework: TensorFlow & Keras
  - Libraries: `numpy`, `matplotlib`, `seaborn`, `sklearn`

---

### 🧠 Deep Learning Pipeline

#### 1. **Data Loading & Visualization**
- Loaded MNIST dataset directly from Keras.
- Visualized samples from the training set to understand digit styles.
- Checked shape, class balance, and pixel intensity distribution.

#### 2. **Preprocessing**
- Normalized pixel values to range `[0, 1]` for faster convergence.
- Reshaped data to fit the input format expected by CNNs.
- One-hot encoded the labels (0–9).

#### 3. **Model Architecture**
- Implemented a simple yet powerful CNN using Keras Sequential API:
  - 2 Convolutional Layers
  - MaxPooling Layers
  - Dropout for regularization
  - Flatten & Dense layers with Softmax output

#### 4. **Model Training**
- Used `categorical_crossentropy` loss and `adam` optimizer.
- Trained over multiple epochs (e.g., 10–20) with validation split.
- Tracked metrics like accuracy and loss on training and validation sets.

#### 5. **Model Evaluation**
- Evaluated on test dataset.
- Visualized predictions, confusion matrix, and misclassified digits.
- Achieved high accuracy (~98–99%).

---

### 📊 Results

- **Final Test Accuracy**: ~99%
- **Model Type**: CNN (2 Conv Layers + Dropout)
- **Training Time**: ~2–4 minutes (depending on hardware)

---

### 📁 Repository Structure

```bash
Handwritten-Digit-Recognition/
│
├── Hand_Written_Digit_Recognition_Model.ipynb   # Main Jupyter notebook
├── README.md                                    # Project overview and guide
├── requirements.txt                             # All dependencies
└── model/                                       # (optional) Folder to save trained model
```

---

### ✅ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook "Hand_Written_Digit_Recognition_Model.ipynb"
   ```

---

### 📌 Future Enhancements

- Deploy model using Flask or Streamlit for live predictions
- Integrate with webcam or drawing pad input
- Use advanced architectures like LeNet, ResNet, or MobileNet
- Train on augmented or noisy digit images for robustness
