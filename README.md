# CIFAR-10 Image Classifier with PyTorch & Streamlit

This project demonstrates an end-to-end image classification pipeline using a custom Convolutional Neural Network (CNN) built with PyTorch and deployed interactively using Streamlit. The model is trained to classify 32x32 color images into one of 10 categories from the CIFAR-10 dataset.

## Project Overview

- Goal: Classify images into one of 10 CIFAR-10 classes
- Model: Custom 4-layer CNN in PyTorch
- Application: Interactive image classifier using Streamlit
- Output: Top-3 predictions with class labels and confidence scores

## Project Structure

Atreyus AI Project/
├── Atreyus AI Project.ipynb        # Main notebook with training and evaluation
├── model_code.py                   # CNN model definition
├── streamlit_app.py                # Streamlit web application
├── cifar10_cnn.pth                 # Trained model weights
└── README.md                       # Project documentation

## Model Architecture

Conv2D (3 → 32) → ReLU → MaxPool  
Conv2D (32 → 64) → ReLU → MaxPool  
Flatten → Fully Connected (512) → ReLU → Dropout  
Fully Connected (10 classes) → Softmax (via CrossEntropyLoss)

## Streamlit App Instructions

To run the Streamlit app locally:

1. Install dependencies

   pip install -r requirements.txt

2. Run the app

   streamlit run streamlit_app.py

3. Upload a PNG or JPG image. The app will resize the image to 32x32, preprocess it, and classify it into the top 3 most probable classes.

Example output:

1. dog — 85.32%  
2. cat — 10.14%  
3. horse — 4.01%

## Dataset

This project uses the CIFAR-10 dataset:

- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Dataset loading and preprocessing steps are implemented in the notebook and Streamlit app.

## Author

Blaine Swieder  
GitHub: https://github.com/bcswieder117

## License

MIT License

## Future Improvements

- Add Grad-CAM or saliency map visualization
- Improve accuracy using ResNet or other pre-trained models
- Deploy on Streamlit Cloud or Hugging Face Spaces
