import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms

# Configuración de la página
st.set_page_config(page_title="Clasificador COVID", layout="wide")

# Cargar modelo ONNX
onnx_model_path = "covid_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Etiquetas
class_names = ["COVID-19", "Normal", "Viral Pneumonia"]

# Título
st.title("🩻 Clasificador de Radiografías de Tórax")

# Subida múltiple de imágenes
uploaded_files = st.file_uploader("Sube una o más radiografías", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 3))  # Mostrar hasta 3 por fila

    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).numpy()

        # Inferencia
        outputs = session.run(None, {"input": img_tensor})
        pred = np.argmax(outputs[0])
        pred_label = class_names[pred]

        with cols[i % 3]:  # Mostrar en columnas
            st.image(image, caption=f"Predicción: {pred_label}", use_container_width=True)
