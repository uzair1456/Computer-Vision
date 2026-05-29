import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --- 1. Dashboard Customization & Theme ---
st.set_page_config(page_title="AI Traffic Intelligence", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTitle { 
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; font-family: 'Trebuchet MS'; font-size: 50px; font-weight: bold;
    }
    .stSidebar { background-color: #111827; border-right: 1px solid #00d2ff; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='stTitle'>🚦 AI Traffic Enforcement System</h1>", unsafe_allow_html=True)

# --- 2. Model Loading (Aapke 5 Specific Models) ---
@st.cache_resource
def load_models():
    # SMIT Assignment Models
    return {
        "car": YOLO('car model.pt'),
        "bike": YOLO('Bike detection.pt'),
        "belt": YOLO('seatbelt.pt'),
        "helmet": YOLO('helmet model.pt'),
        "plate": YOLO('license plate model.pt')
    }

try:
    models = load_models()
    st.sidebar.success("✅ System Ready: Models Active")
except Exception as e:
    st.sidebar.error(f"❌ Model Missing: {e}")

# --- 3. Advanced Detection Logic ---
def analyze_frame(frame, conf_val):
    # Perform Detections
    res = {k: v(frame, conf=conf_val, verbose=False)[0] for k, v in models.items()}
    
    # Store sub-object boxes
    belt_boxes = res['belt'].boxes.xyxy.cpu().numpy()
    helm_boxes = res['helmet'].boxes.xyxy.cpu().numpy()
    plate_boxes = res['plate'].boxes.xyxy.cpu().numpy()

    # --- A. Car & Seatbelt Logic ---
    for car in res['car'].boxes:
        c = car.xyxy[0].cpu().numpy().astype(int)
        # Buffer check: overlapping car area with belt detections
        has_belt = any((b[0] < c[2] and b[2] > c[0] and b[1] < c[3] and b[3] > c[1]) for b in belt_boxes)
        
        status = "With Seatbelt" if has_belt else "Without Seatbelt"
        color = (0, 255, 0) if has_belt else (0, 0, 255) # Green vs Red
        
        cv2.rectangle(frame, (c[0], c[1]), (c[2], c[3]), color, 3)
        cv2.putText(frame, status, (c[0], c[1]-12), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    # --- B. Bike & Helmet Logic ---
    for bike in res['bike'].boxes:
        bk = bike.xyxy[0].cpu().numpy().astype(int)
        # Check if helmet box exists inside bike coordinates
        has_helm = any((h[0] < bk[2] and h[2] > bk[0] and h[1] < bk[3] and h[3] > bk[1]) for h in helm_boxes)
        
        status = "With Helmet" if has_helm else "Without Helmet"
        color = (0, 255, 0) if has_helm else (0, 0, 255)
        
        cv2.rectangle(frame, (bk[0], bk[1]), (bk[2], bk[3]), color, 3)
        cv2.putText(frame, status, (bk[0], bk[1]-12), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    # --- C. License Plate Logic ---
    for p in plate_boxes:
        pb = p.astype(int)
        # Yellow bounding box for plate
        cv2.rectangle(frame, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 255), 4)
        cv2.putText(frame, "LP: DETECTED-PK", (pb[0], pb[3]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame

# --- 4. Main UI Interaction ---
st.sidebar.header("📁 Control Panel")
upload = st.sidebar.file_uploader("Upload Image or Video", type=['jpg','png','jpeg','mp4'])
sens = st.sidebar.slider("Detection Sensitivity", 0.05, 1.0, 0.18)

if upload:
    if upload.name.endswith(('.mp4', '.avi')):
        # Video Handling
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            processed = analyze_frame(frame, sens)
            st_frame.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()
        os.unlink(tfile.name)
    else:
        # Image Handling
        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        output = analyze_frame(img, sens)
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("Sidebar se file upload karein aur sensitivity 0.15-0.20 ke darmiyan rakhen.")