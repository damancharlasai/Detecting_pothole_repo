from flask import Flask, request, render_template, redirect, url_for
import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO 

app = Flask(__name__)

# Load PyTorch model
model_path_pt = "models/best.pt"
model_path_onnx = "models/best.onnx"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_model = YOLO(model_path_pt)  # Adjust based on your model structure

# Load ONNX model
onnx_session = ort.InferenceSession(model_path_onnx)

# Helper function to process image and detect potholes
def detect_potholes(image_path, use_onnx=False):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    if use_onnx:
        input_data = cv2.resize(image_np, (640, 640)).astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))[np.newaxis, ...]  # HWC to NCHW
        outputs = onnx_session.run(None, {"input": input_data})
        # Adjust based on ONNX output format
    else:
        results = torch_model(image_path)  # Run inference with YOLO
        for box in results[0].boxes:  # Assuming results[0].boxes contains bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save output image
    output_path = "static/output.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return output_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if not image:
            return "No file uploaded", 400
        
        # Save uploaded image
        image_path = "static/input.jpg"
        image.save(image_path)
        
        # Detect potholes
        use_onnx = request.form.get("use_onnx", "false") == "true"
        output_path = detect_potholes(image_path, use_onnx=use_onnx)
        
        return render_template("result.html", output_image=output_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
