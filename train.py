# train.py
import os
import argparse
import time
import random
import shutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import get_model
from utils import save_checkpoint, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='dataset', help='dataset root with train/val subfolders')
    p.add_argument('--model', type=str, default='resnet18')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=2)   # reduced to 2
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=0, help='dataloader workers (set 0 on Windows)')
    p.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--fine-tune', action='store_true', help='fine-tune whole network')
    return p.parse_args()


def find_existing_dir(root, names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    return None


def create_val_from_train(train_dir, val_dir, val_frac=0.10, seed=42):
    random.seed(seed)
    os.makedirs(val_dir, exist_ok=True)
    for cls in os.listdir(train_dir):
        src_cls = os.path.join(train_dir, cls)
        if not os.path.isdir(src_cls):
            continue
        dst_cls = os.path.join(val_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)
        imgs = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if not imgs:
            continue
        n_move = max(1, int(len(imgs) * val_frac))
        random.shuffle(imgs)
        for f in imgs[:n_move]:
            shutil.move(os.path.join(src_cls, f), os.path.join(dst_cls, f))


def make_dataloaders(data_dir, batch_size, workers):
    size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dir = find_existing_dir(data_dir, ['train','Train','TRAIN'])
    val_dir   = find_existing_dir(data_dir, ['val','Val','VAL','validation','Validation'])

    if train_dir is None:
        raise FileNotFoundError(f"No train folder found under {data_dir}. Expected e.g. {os.path.join(data_dir,'train')}")

    if val_dir is None:
        print(f"No validation folder found under {data_dir}. Creating a small validation split (10%) from {train_dir}.")
        val_dir = os.path.join(data_dir, 'val')
        create_val_from_train(train_dir, val_dir, val_frac=0.10)

    def report(path):
        classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        if not classes:
            raise RuntimeError(f"No class subfolders found in {path}. Expected folders like 'fake' and 'real'.")
        total = 0
        per = {}
        for c in classes:
            files = [f for f in os.listdir(os.path.join(path, c)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            per[c] = len(files)
            total += len(files)
        print(f"Data dir: {path} | classes: {classes} | total_images: {total} | per_class: {per}")
        return classes

    report(train_dir)
    report(val_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=(workers>0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=(workers>0))

    print("ImageFolder class_to_idx (train):", train_dataset.class_to_idx)
    print("ImageFolder class_to_idx (val):", val_dataset.class_to_idx)
    return train_loader, val_loader


def train():
    args = parse_args()
    print("Args:", vars(args))
    device = torch.device(args.device)
    print("Using device:", device)

    train_loader, val_loader = make_dataloaders(args.data_dir, args.batch_size, args.workers)

    model = get_model(model_name=args.model, pretrained=True, num_classes=1).to(device)

    if not args.fine_tune:
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Training only classifier layers (use --fine-tune to unfreeze).")
    else:
        print("Fine-tuning entire model.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=args.lr, weight_decay=args.weight_decay)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        start_epoch = time.time()
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = imgs.size(0)
            epoch_loss += loss.item() * batch_size
            n_samples += batch_size
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

        epoch_loss /= max(1, n_samples)
        metrics = evaluate(model, val_loader, device)
        val_auc, val_acc = metrics.get('auc', 0.0), metrics.get('acc', 0.0)
        print(f"Epoch {epoch}/{args.epochs} — loss: {epoch_loss:.4f} — val_auc: {val_auc:.4f} — val_acc: {val_acc:.4f} — time: {time.time()-start_epoch:.1f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_auc': best_auc,
                'args': vars(args)
            }, filename=args.checkpoint)
            print(f"✅ Saved best checkpoint (AUC={best_auc:.4f})")

    # Also save a last checkpoint (useful for resuming or debugging)
    try:
        last_path = os.path.join(os.path.dirname(args.checkpoint), 'last.pth')
        save_checkpoint({
            'epoch': args.epochs,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_auc': best_auc,
            'args': vars(args)
        }, filename=last_path)
        print(f"💾 Saved last checkpoint to {last_path}")
    except Exception:
        # non-critical: don't fail the script if last save fails
        pass

    print("🎯 Training complete — Best AUC:", best_auc)


if __name__ == '__main__':
    train()
