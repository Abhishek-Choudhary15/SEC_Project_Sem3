# infer.py
import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import get_model
from utils import load_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    p.add_argument('--image', type=str, required=True, help='path to image or folder')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--model', type=str, default='resnet18')
    p.add_argument('--batch-size', type=int, default=0, help='if >0 run batched inference using DataLoader')
    p.add_argument('--limit', type=int, default=0, help='limit number of images to run (0 = no limit)')
    p.add_argument('--out-csv', type=str, default='', help='optional CSV path to write results')
    return p.parse_args()

def prepare_transform():
    size = 224
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def infer_one(model, img_path, transform, device):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        out = out.item()
        prob = 1.0 / (1.0 + np.exp(-out))  # sigmoid
    return prob


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        x = self.transform(img)
        return x, p

def main():
    args = parse_args()
    device = torch.device(args.device)
    # create model and load (be tolerant to different checkpoint dict key names)
    model = get_model(model_name=args.model, pretrained=False, num_classes=1).to(device)
    state = load_checkpoint(args.checkpoint, device)
    # try a few common keys used for saved model state
    for k in ('model_state', 'model_state_dict', 'state_dict'):
        if k in state:
            model.load_state_dict(state[k])
            break
    else:
        # maybe the checkpoint *is* a raw state_dict
        try:
            model.load_state_dict(state)
        except Exception as e:
            raise KeyError(f"Could not find model weights in checkpoint: available keys={list(state.keys())}") from e

    transform = prepare_transform()

    if os.path.isdir(args.image):
        imgs = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    else:
        imgs = [args.image]

    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]

    results = []
    # batched path using DataLoader
    if args.batch_size and args.batch_size > 0 and len(imgs) > 1:
        ds = ImagePathDataset(imgs, transform)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dl, desc='Infer'):
                xs, paths = batch
                xs = xs.to(device)
                outs = model(xs).view(-1).cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-outs))
                for pth, prob in zip(paths, probs):
                    label = 'fake' if prob >= 0.5 else 'real'
                    results.append((pth, float(prob), label))
                    print(f"{os.path.basename(pth)} -> prob_fake={prob:.4f} -> {label}")
    else:
        for pth in imgs:
            prob = infer_one(model, pth, transform, device)
            label = 'fake' if prob >= 0.5 else 'real'
            results.append((pth, prob, label))
            print(f"{os.path.basename(pth)} -> prob_fake={prob:.4f} -> {label}")

    # optional CSV output
    if args.out_csv:
        try:
            import csv
            with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['path', 'prob_fake', 'label'])
                for pth, prob, label in results:
                    w.writerow([pth, f"{prob:.6f}", label])
            print(f"Wrote results to {args.out_csv}")
        except Exception as e:
            print(f"Failed to write CSV: {e}")

if __name__ == '__main__':
    main()
