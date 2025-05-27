import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model_code import SimpleCNN
import os

# --- Custom Style ---
st.markdown("""
<style>
    .reportview-container {
        background: #111;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #222;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- CIFAR-10 Class Labels ---
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# --- Load Trained Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
model.to(device)
model.eval()

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Sidebar ---
st.sidebar.title(" Project Info ")
st.sidebar.markdown("""
**CIFAR-10 Image Classifier**  
This model classifies images into one of 10 categories using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

**Model:** Custom 4-layer CNN  
**Framework:** PyTorch + Streamlit  
**Author:** Blaine Swieder  
""")

with st.sidebar.expander(" CIFAR-10 Classes "):
    for i, label in enumerate(classes):
        st.write(f"{i}. {label}")

# --- App Header ---
st.title(" CIFAR-10 Image Classifier ")
st.markdown(" Upload a PNG or JPG image to see the model's prediction with top-3 confidence scores. ")

# --- Layout Columns ---
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Warn about resizing
        if image.size != (32, 32):
            st.warning(f"Resizing image from {image.size} to 32x32 to match model input.")

        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            top_probs, top_idxs = torch.topk(probs, 3)

        st.subheader("Top 3 Predictions")
        for i in range(3):
            label = classes[top_idxs[0][i]]
            prob = top_probs[0][i].item() * 100
            st.write(f"{i+1}. **{label}** — {prob:.2f}%")
    else:
        st.info("Upload an image to get predictions.")
