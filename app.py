from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
from utils import load_checkpoint
import clip
import torch.nn.functional as F

# ===========================================================
# DEVICE CONFIG
# ===========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================================
# LOAD RESNET DEEPFAKE MODEL
# ===========================================================
MODEL_PATH = "checkpoints/best.pth"

def prepare_transform():
    size = 224
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

transform = prepare_transform()

model = get_model(model_name="resnet18", pretrained=False, num_classes=1).to(DEVICE)
state = load_checkpoint(MODEL_PATH, DEVICE)

# flexible key loader
for key in ("model_state", "model_state_dict", "state_dict"):
    if key in state:
        model.load_state_dict(state[key])
        break
else:
    model.load_state_dict(state)

model.eval()

# ===========================================================
# LOAD CLIP AI IMAGE DETECTOR
# ===========================================================
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# ===========================================================
# CLIP AI / DEEPFAKE DETECTOR
# ===========================================================
def clip_detector(image_path):
    image = Image.open(image_path).convert("RGB")
    img_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)

    text_labels = [
        "a real human photograph",
        "an AI generated face",
        "a synthetic human face",
        "a digitally created human face",
        "a deepfake human face"
    ]

    text_tokens = clip.tokenize(text_labels).to(DEVICE)

    with torch.no_grad():
        img_feat = clip_model.encode_image(img_input)
        txt_feat = clip_model.encode_text(text_tokens)

        logits = (img_feat @ txt_feat.T) * 100
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    real_conf = probs[0]                # score only for real
    fake_conf = max(probs[1:])          # take highest fake label score

    return real_conf, fake_conf         # both 0–1 range


# ===========================================================
# RESNET DEEPFAKE DETECTOR
# ===========================================================
def resnet_detector(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x).item()
        prob = 1 / (1 + np.exp(-out))

    real_prob = prob
    fake_prob = 1 - prob
    return real_prob, fake_prob


# ===========================================================
# FINAL FUSED DETECTOR
# ===========================================================
def predict_single_image(image_path):

    # RESNET deepfake detection
    r_real, r_fake = resnet_detector(image_path)

    # CLIP detection
    c_real, c_fake = clip_detector(image_path)

    # Weighted fusion (CLIP stronger)
    final_fake = (0.7 * c_fake) + (0.3 * r_fake)
    final_real = (0.7 * c_real) + (0.3 * r_real)

    if final_fake > final_real:
        label = "deepfake"
        confidence = final_fake
    else:
        label = "real"
        confidence = final_real

    return label, round(confidence, 2)   # convert to percentage


# ===========================================================
# FLASK BACKEND
# ===========================================================
app = Flask(__name__)
CORS(app)

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
