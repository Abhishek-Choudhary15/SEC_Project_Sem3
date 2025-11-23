# utils.py
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

def save_checkpoint(state, filename='checkpoints/checkpoint.pth'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename, device='cpu'):
    state = torch.load(filename, map_location=device)
    return state

def compute_metrics(y_true, y_scores, threshold=0.5):
    # y_scores are probabilities or logits (if logits, send through sigmoid first)
    probs = 1 / (1 + np.exp(-y_scores)) if (y_scores.min() < 0 or y_scores.max() > 1) else y_scores
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, preds)
    return {'auc': auc, 'acc': acc}

def evaluate(model, dataloader, device):
    model.eval()
    ys = []
    ys_scores = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Eval", leave=False):
            imgs = imgs.to(device)
            out = model(imgs)
            out = out.view(-1).cpu().numpy()
            ys_scores.extend(out.tolist())
            ys.extend(labels.numpy().tolist())
    return compute_metrics(np.array(ys), np.array(ys_scores))
