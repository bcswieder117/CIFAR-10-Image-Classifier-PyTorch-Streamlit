# CIFAR-10 Image Classifier with PyTorch & Streamlit

This project demonstrates an end-to-end image classification pipeline using a custom Convolutional Neural Network (CNN) built with PyTorch and deployed interactively using Streamlit. The model is trained to classify 32x32 color images into one of 10 categories from the CIFAR-10 dataset.

---

## Project Overview

- **Goal**: Classify images into one of 10 CIFAR-10 classes
- **Model**: Custom 4-layer CNN built in PyTorch
- **Application**: Interactive image classifier using Streamlit
- **Output**: Top-3 predictions with class labels and confidence scores

---

## Project Structure

```
Atreyus AI Project/
├── Atreyus AI Project.ipynb        # Main notebook with training and evaluation
├── model_code.py                   # CNN model definition
├── streamlit_app.py                # Streamlit web application
├── cifar10_cnn.pth                 # Trained model weights
└── README.md                       # Project documentation
```

---

## Model Architecture

```
Conv2D (3 → 32) → ReLU → MaxPool
Conv2D (32 → 64) → ReLU → MaxPool
Flatten → Fully Connected (512) → ReLU → Dropout
Fully Connected (10 outputs) → Softmax (via CrossEntropyLoss)
```

---

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/bcswieder117/CIFAR-10-Image-Classifier-PyTorch-Streamlit.git
    cd CIFAR-10-Image-Classifier-PyTorch-Streamlit
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

---

## Requirements

- Python 3.8+
- torch
- torchvision
- streamlit
- pillow
- pandas

---

## Notes

- The dataset itself is **not required** to run the Streamlit app. The model has already been trained and is saved as `cifar10_cnn.pth`.
- If you wish to retrain the model, download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and follow the notebook `CIFAR 10 Image Classifier.ipynb`.

---

## Author

**Blaine Swieder**  
GitHub: [bcswieder117](https://github.com/bcswieder117)

---
