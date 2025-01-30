# Pothole_Detection

![image](https://github.com/user-attachments/assets/1dccdc56-b037-44dd-80f0-fdb77c7c288b)

![image](https://github.com/user-attachments/assets/92b7dedd-c0d6-472e-9ebe-06d9c0cee90c)


## Description
This project demonstrates a pothole detection system. The application uses a deep learning-based object detection model to detect potholes in uploaded road images and provides feedback by highlighting detected potholes with bounding boxes.
The app is implemented in Python using Flask for the web interface, and the models are trained using PyTorch and ONNX for inference.

## Features
Upload an image of a road through a simple web interface.

Detect potholes in the uploaded image and highlight them with bounding boxes.

Provide a message, "Potholes detected. Drive safely," for better user experience.

Beautiful and responsive web design for user-friendly interaction.

**Updated Model:** Includes augmented and resized training data to improve detection accuracy and robustness.

## Prerequisites
Python 3.11
Virtual environment setup (recommended)

## Installation

Clone the repository:
bash : Copy code : 
git clone https://github.com/TharushiHansika/Pothole_Detection.git
cd PotholeDetection

Create a virtual environment:
bash : Copy code : 
python -m venv venv

pip install -r requirements.txt

Activate the virtual environment:
#### On Windows:
bash : Copy code : 
venv\Scripts\activate

## Usage

Start the Flask application:
bash : Copy code : 
python app.py

Open your browser and navigate to:
Copy code : 
http://127.0.0.1:5000

## Model Information
best.pt: A PyTorch-based YOLO model trained to detect potholes.

best.onnx: An ONNX-converted model for cross-platform inference.

## Technologies Used
Backend: Python, Flask

Frontend: HTML, CSS

Deep Learning: PyTorch, ONNX

Dependencies: OpenCV, Pillow, NumPy, Ultralytics YOLO

## Author
Tharushi Hansika - Initial Work
