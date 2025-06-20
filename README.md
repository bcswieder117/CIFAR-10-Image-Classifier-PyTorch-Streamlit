# CIFAR-10 Image Classifier with PyTorch & Streamlit

This project demonstrates an end-to-end image classification pipeline using a custom Convolutional Neural Network (CNN) built with PyTorch and deployed interactively using Streamlit. The model is trained to classify 32×32 color images into one of 10 categories from the CIFAR-10 dataset.

---

## Project Overview

* **Goal**: Classify images into one of 10 CIFAR-10 classes
* **Model**: Custom 4-layer CNN built in PyTorch
* **Application**: Interactive image classifier using Streamlit
* **Output**: Top-3 predictions with class labels and confidence scores

---

## Project Structure

```
Atreyus AI Project/
├── Atreyus AI Project.ipynb        # Main notebook with training and evaluation
├── model_code.py                   # CNN model definition
├── streamlit_app.py                # Streamlit web application
├── cifar10_cnn.pth                 # Trained model weights
├── README.md                       # Project documentation
├── CNN Classifier Workflow.png     # Workflow diagram
├── CNN Classifier Results Figure.png
├── CNN Images_Output.png           # Sample gallery
├── Streamlit Ex1.png               # Streamlit home screen
├── Streamlit Ex2.png               # Streamlit prediction result
├── My CNN Architecture.png         # Detailed model diagram
├── CNN Show Top3 Predictions.png   # CLI predictions
└── CNN Image Classifier Results.png# Console output snippet
```

---

## Model Architecture

```
Conv2D (3 → 32) → ReLU → MaxPool
Conv2D (32 → 64) → ReLU → MaxPool
Flatten → Fully Connected (512) → ReLU → Dropout
Fully Connected (10 outputs) → Softmax (via CrossEntropyLoss)
```

<p align="center">
  <img src="CNN Classifier Workflow.png" alt="Workflow Diagram" width="600" />
</p>

---

## Data Preprocessing

* **Normalization**: Mean/std scaling per channel
* **Augmentation**: Random horizontal flips, random crops, and rotations during training

---

## Training Details

* **Epochs**: 5
* **Optimizer**: Adam (LR = 0.001)
* **Loss Function**: CrossEntropyLoss

<p align="center">
  <img src="CNN Image Classifier Results.png" alt="Training Loop Console Output" width="600" />
</p>

---

## Results

<p align="center">
  <img src="CNN Classifier Results Figure.png" alt="Training Progress and Overall Accuracy" width="600" />
</p>

* **Overall Test Accuracy**: **82.90%**
* **Training Accuracy by Epoch**:

  1. 52.73%
  2. 67.32%
  3. 73.44%
  4. 78.27%
  5. 82.90%
* **Training Loss by Epoch**:

  1. 2054.92
  2. 1453.81
  3. 1172.51
  4. 962.14
  5. 755.43

---

## Sample Predictions

Below are random test samples with top-3 predicted classes and confidence scores.

<p align="center">
  <img src="CNN Show Top3 Predictions.png" alt="CLI Top-3 Predictions" width="600" />
</p>

<p align="center">
  <img src="CNN Images_Output.png" alt="Sample Predictions Gallery" width="600" />
</p>

| Image       | 1st Prediction | 2nd Prediction | 3rd Prediction |
| ----------- | -------------- | -------------- | -------------- |
| `1.png`     | deer — 67.61%  | cat — 13.07%   | bird — 11.50%  |
| `10.png`    | dog — 46.20%   | cat — 30.65%   | deer — 13.32%  |
| `100.png`   | deer — 91.00%  | bird — 8.13%   | frog — 0.58%   |
| `1000.png`  | truck — 32.44% | frog — 19.21%  | deer — 14.17%  |
| `10000.png` | frog — 64.52%  | bird — 20.17%  | deer — 8.42%   |

---

## Workflow Diagram

Here's the end-to-end pipeline overview:

<p align="center">
  <img src="My CNN Architecture.png" alt="Detailed Model Architecture" width="600" />
</p>

---

## Streamlit App

The interactive web app automatically resizes uploads to 32×32 and displays top-3 predictions:

<table>
  <tr>
    <td align="center"><img src="Streamlit Ex1.png" alt="Streamlit Home Screen" width="300" /></td>
    <td align="center"><img src="Streamlit Ex2.png" alt="Streamlit Prediction Result" width="300" /></td>
  </tr>
  <tr>
    <td align="center">Home Screen</td>
    <td align="center">Prediction Result</td>
  </tr>
</table>

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

* Python 3.8+
* torch
* torchvision
* streamlit
* pillow
* pandas

---

## Notes

* The dataset itself is **not required** to run the Streamlit app. The model has already been trained (`cifar10_cnn.pth`).
* To retrain, download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and follow the notebook.

---

## Author

**Blaine Swieder**
GitHub: [bcswieder117](https://github.com/bcswieder117)


