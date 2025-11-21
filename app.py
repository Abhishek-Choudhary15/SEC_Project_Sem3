from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
from utils import load_checkpoint

# ------------------------------
# MODEL LOADING (FAST)
# ------------------------------

MODEL_PATH = "checkpoints/best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def prepare_transform():
    size = 224
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

transform = prepare_transform()

# Load model
model = get_model(model_name="resnet18", pretrained=False, num_classes=1).to(DEVICE)
state = load_checkpoint(MODEL_PATH, DEVICE)

for key in ("model_state", "model_state_dict", "state_dict"):
    if key in state:
        model.load_state_dict(state[key])
        break
else:
    model.load_state_dict(state)

model.eval()

# ------------------------------
# PREDICT FUNCTION
# ------------------------------

def predict_single_image(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x).item()
        prob = 1 / (1 + np.exp(-out))

    label = "deepfake" if prob >= 0.5 else "real"
    return label, float(prob)

# ------------------------------
# FLASK API + CORS ENABLED
# ------------------------------

app = Flask(__name__)
CORS(app)  # ★ IMPORTANT — allows frontend to call backend

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_path = "uploaded.jpg"
    file.save(file_path)

    label, confidence = predict_single_image(file_path)

    return jsonify({
        "label": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
