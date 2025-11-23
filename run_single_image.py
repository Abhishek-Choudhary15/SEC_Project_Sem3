import torch
import timm
from PIL import Image
from torchvision import transforms

MODEL_NAME = "resnet18"
CHECKPOINT = "checkpoints/best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Preprocess
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load model
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

# Input
img_path = input("Enter image path: ").strip()
img = Image.open(img_path).convert("RGB")
img_t = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    prob_fake = torch.sigmoid(model(img_t)).item()

print(f"\nFake Probability: {prob_fake:.4f}")
print("Prediction:", "FAKE" if prob_fake >= 0.5 else "REAL")
