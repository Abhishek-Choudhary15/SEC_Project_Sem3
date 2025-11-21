import os
import argparse
import csv
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from utils import load_checkpoint, compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, required=True, help='root folder with class subfolders (ImageFolder)')
    p.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    p.add_argument('--model', type=str, default='resnet18')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--out-csv', type=str, default='eval_results.csv')
    p.add_argument('--limit', type=int, default=0, help='limit number of samples (0=no limit)')
    return p.parse_args()


def prepare_transform():
    size = 224
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])


def main():
    args = parse_args()
    device = torch.device(args.device)

    transform = prepare_transform()
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)

    # build mapping to interpret which class index corresponds to 'fake'
    class_to_idx = dataset.class_to_idx
    fake_idx = None
    for k, v in class_to_idx.items():
        if k.lower() == 'fake':
            fake_idx = v
            break
    if fake_idx is None:
        # try common alternatives
        for k, v in class_to_idx.items():
            if 'fake' in k.lower():
                fake_idx = v
                break

    print('Class to idx:', class_to_idx, '-> interpreted fake_idx=', fake_idx)

    # optionally limit samples
    samples = dataset.samples
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
        # create a small subset dataset-like structure: use the paths list for manual loading

    # DataLoader for batched inference on the dataset (we'll re-create a lightweight loader if limited)
    if args.limit and args.limit > 0:
        # create a simple list of (path, target) pairs
        paths = [p for p, t in samples]
        targets = [t for p, t in samples]
        # manual batching
        batched = [paths[i:i+args.batch_size] for i in range(0, len(paths), args.batch_size)]
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(args.workers>0))

    # create model and load checkpoint
    model = get_model(model_name=args.model, pretrained=False, num_classes=1).to(device)
    state = load_checkpoint(args.checkpoint, device)
    for k in ('model_state', 'model_state_dict', 'state_dict'):
        if k in state:
            model.load_state_dict(state[k])
            break
    else:
        try:
            model.load_state_dict(state)
        except Exception:
            raise KeyError(f'Could not find model weights in checkpoint: keys={list(state.keys())}')

    model.eval()

    results = []
    ys = []
    ys_scores = []

    if args.limit and args.limit > 0:
        with torch.no_grad():
            for batch_paths in tqdm(batched, desc='Eval'):
                imgs = []
                for p in batch_paths:
                    from PIL import Image
                    img = Image.open(p).convert('RGB')
                    x = transform(img)
                    imgs.append(x)
                xs = torch.stack(imgs).to(device)
                outs = model(xs).view(-1).cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-outs))
                for p, prob in zip(batch_paths, probs):
                    # determine true label from original dataset samples mapping
                    # find index in dataset.samples
                    idx = next(i for i, s in enumerate(samples) if s[0] == p)
                    true_idx = samples[idx][1]
                    if fake_idx is None:
                        # assume target 1 == fake
                        y_true = 1 if true_idx == 1 else 0
                    else:
                        y_true = 1 if true_idx == fake_idx else 0
                    ys.append(y_true)
                    ys_scores.append(prob)
                    label = 'fake' if prob >= 0.5 else 'real'
                    results.append((p, y_true, float(prob), label))
    else:
        with torch.no_grad():
            for imgs, targets in tqdm(loader, desc='Eval'):
                imgs = imgs.to(device)
                outs = model(imgs).view(-1).cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-outs))
                for p_idx, prob in zip(targets, probs):
                    # In this loop we don't have file paths directly; instead use dataset.samples order
                    pass
            # simpler way: iterate through dataset with loader indices
        # For full dataset, we'll iterate samples directly to capture paths and targets
        with torch.no_grad():
            for p, t in tqdm(dataset.samples, desc='EvalSamples'):
                from PIL import Image
                img = Image.open(p).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                out = model(x).view(-1).cpu().item()
                prob = 1.0 / (1.0 + np.exp(-out))
                if fake_idx is None:
                    y_true = 1 if t == 1 else 0
                else:
                    y_true = 1 if t == fake_idx else 0
                ys.append(y_true)
                ys_scores.append(prob)
                label = 'fake' if prob >= 0.5 else 'real'
                results.append((p, y_true, float(prob), label))

    metrics = compute_metrics(np.array(ys), np.array(ys_scores))
    print('Eval metrics:', metrics)

    # write CSV
    try:
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['path', 'true_is_fake', 'prob_fake', 'pred_label'])
            for p, y_true, prob, label in results:
                w.writerow([p, y_true, f"{prob:.6f}", label])
        print('Wrote eval CSV to', args.out_csv)
    except Exception as e:
        print('Failed to write CSV:', e)


if __name__ == '__main__':
    main()
