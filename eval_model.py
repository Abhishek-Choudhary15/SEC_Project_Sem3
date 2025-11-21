# eval_model.py
import torch
import timm
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# ✅ Paths
DATA_DIR = "dataset/test"
CHECKPOINT = "checkpoints/best.pth"
MODEL_NAME = "resnet18"

# ✅ transforms (same as val)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ✅ Load dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

print("class_to_idx:", dataset.class_to_idx)

# ✅ Load model
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

probs = []
labels = []

# ✅ inference
with torch.no_grad():
    for imgs, labs in tqdm(loader):
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        probs.extend(p)
        labels.extend(labs.numpy().tolist())

# ✅ convert predictions
preds = [1 if x >= 0.5 else 0 for x in probs]

# ✅ metrics
print("\nAUC:", roc_auc_score(labels, probs))
print("\nClassification Report:\n", classification_report(labels, preds, target_names=dataset.classes))
print("\nConfusion Matrix:\n", confusion_matrix(labels, preds))

# ✅ save predictions to CSV
df = pd.DataFrame({
    "path": [p for p, _ in dataset.samples],
    "label": labels,
    "prob_fake": probs,
    "pred": preds
})
df.to_csv("test_predictions.csv", index=False)
print("\n✅ Saved test_predictions.csv")
