import os
import cv2
import time
import torch
import tempfile
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
confidence_threshold = 0.5

class CustomEfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

@st.cache_resource
def load_models():
    yolo_model = YOLO('yolov11n-face.pt')
    emotion_model = CustomEfficientNetClassifier(num_classes).to(device)
    emotion_model.load_state_dict(torch.load("efficientnet_b0_20_0.0001_32.pt", map_location=device))
    emotion_model.eval()
    return yolo_model, emotion_model

modelyolov11, model = load_models()
class_labels = torch.load("emotion_labels.pt")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def crop_face_pad_and_resize(frame, box, output_size=224, fill_color=(0, 0, 0)):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    face = frame[y1:y2, x1:x2]
    pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    w, h = pil_face.size
    delta_w = max(h - w, 0)
    delta_h = max(w - h, 0)
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    face_square = ImageOps.expand(pil_face, padding, fill=fill_color)
    return face_square.resize((output_size, output_size), Image.LANCZOS)

def predict_emotion(face_img):
    img_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_index = output.argmax(1).item()
        return class_labels[pred_index]

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = modelyolov11.predict(frame, conf=confidence_threshold)
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_img = crop_face_pad_and_resize(frame, (x1, y1, x2, y2))
                emotion = predict_emotion(face_img)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
    time.sleep(1)
    return output_path

st.title("🎭 Facial Emotion Detection (EfficientNet-B0)")
st.write("Upload a video to detect facial emotions.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mpeg4"])

if uploaded_file is not None:

    if (
        "processed_b0_path" not in st.session_state or
        "last_uploaded_filename_b0" not in st.session_state or
        st.session_state.last_uploaded_filename_b0 != uploaded_file.name
    ):
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_file.read())
        temp_input.close()

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output.close()

        with st.spinner("Processing video..."):
            processed_path = process_video(temp_input.name, temp_output.name)
            st.session_state.processed_b0_path = processed_path
            st.session_state.last_uploaded_filename_b0 = uploaded_file.name
            st.success("✅ Processing complete!")

    with open(st.session_state.processed_b0_path, "rb") as file:
        st.download_button(
            label="📥 Download Processed Video",
            data=file,
            file_name="emotion_detection_b0.mp4",
            mime="video/mp4"
        )
