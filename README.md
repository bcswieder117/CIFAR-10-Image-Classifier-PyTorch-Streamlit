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

    CIFAR-10-Image-Classifier-PyTorch-Streamlit/
    ├── Atreyus AI Project.ipynb            # Main notebook with training & evaluation
    ├── model_code.py                       # CNN model definition
    ├── streamlit_app.py                    # Streamlit web application
    ├── cifar10_cnn.pth                     # Trained model weights
    ├── README.md                           # Project documentation
    ├── CNN Classifier Workflow.png         # Workflow diagram
    ├── CNN Classifier Results Figure.png   # Metric summary dashboard
    ├── CNN Images_Output.png               # Sample gallery grid
    ├── Streamlit Ex 1.png                  # Streamlit home/upload screen
    ├── Streamlit Ex 2.png                  # Streamlit prediction result view
    ├── My CNN Architecture.png             # Detailed model diagram
    ├── CNN Show Top3 Predictions.png       # CLI top-3 predictions printout
    └── CNN Image Classifier Results.png    # Training console output

---

## Model Architecture

    Conv2D (3 → 32) → ReLU → MaxPool  
    Conv2D (32 → 64) → ReLU → MaxPool  
    Flatten → Fully Connected (512) → ReLU → Dropout  
    Fully Connected (10 outputs) → Softmax (via CrossEntropyLoss)

![Detailed Model Architecture](./My%20CNN%20Architecture.png)

---

## Data Preprocessing

* **Normalization**: Mean/std scaling per channel  
* **Augmentation**: Random horizontal flips, random crops, rotations  

---

## Training Details

* **Epochs**: 5  
* **Optimizer**: Adam (LR = 0.001)  
* **Loss Function**: CrossEntropyLoss  

![Training Console Output](./CNN%20Image%20Classifer%20Results.png)

---

## Results

![Training Progress and Overall Accuracy](./CNN%20Classifier%20Results%20Figure.png)

* **Overall Test Accuracy:** 82.90%  
* **Training Accuracy by Epoch:**  
  1. 52.73%  
  2. 67.32%  
  3. 73.44%  
  4. 78.27%  
  5. 82.90%  
* **Training Loss by Epoch:**  
  1. 2054.92  
  2. 1453.81  
  3. 1172.51  
  4. 962.14  
  5. 755.43  

---

## Sample Predictions

![CLI Top-3 Predictions](./CNN%20Show%20Top3%20Predictions.png)  
![Sample Predictions Gallery](./CNN%20Images_Output.png)

| Image       | 1st Prediction   | 2nd Prediction  | 3rd Prediction |
| ----------- | ---------------- | --------------- | -------------- |
| `1.png`     | deer — 67.61%    | cat — 13.07%    | bird — 11.50%  |
| `10.png`    | dog — 46.20%     | cat — 30.65%    | deer — 13.32%  |
| `100.png`   | deer — 91.00%    | bird — 8.13%    | frog — 0.58%   |
| `1000.png`  | truck — 32.44%   | frog — 19.21%   | deer — 14.17%  |
| `10000.png` | frog — 64.52%    | bird — 20.17%   | deer — 8.42%   |

---

## Workflow Pipeline

![CNN Classifier Workflow](./CNN%20Classifier%20Workflow.png)



---

## Streamlit App

The interactive web app automatically resizes uploads to 32×32 and displays top-3 predictions:

| Home Screen                                 | Prediction View                                    |
| ------------------------------------------- | -------------------------------------------------- |
| ![Streamlit Home](./Streamlit%20Ex%201.png) | ![Streamlit Prediction](./Streamlit%20Ex%202.png)  |
| *Upload interface & class chooser*          | *Resized image & top-3 confidence scores*          |

---

## Getting Started

1. **Clone** the repository  
    ```bash
    git clone https://github.com/bcswieder117/CIFAR-10-Image-Classifier-PyTorch-Streamlit.git
    cd CIFAR-10-Image-Classifier-PyTorch-Streamlit
    ```

2. **Install** dependencies  

    ```bash
    pip install -r requirements.txt
    ```

3. **Run** the Streamlit app  

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

* No dataset download required—the pretrained weights (`cifar10_cnn.pth`) are included.  
* To retrain, download CIFAR-10 from [the official site](https://www.cs.toronto.edu/~kriz/cifar.html) and run the notebook.  

---

## Author

**Blaine Swieder**  
GitHub: [bcswieder117](https://github.com/bcswieder117)
