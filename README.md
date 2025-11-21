# Fake detector — training and inference

Quick notes to run training and inference on Windows (PowerShell).

1) Install dependencies (use a suitable torch build for your CUDA/CPU):

```powershell
# Recommended: create and activate a venv, then install
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
# If you need a specific torch+cuda wheel, follow https://pytorch.org/get-started/locally/
```

2) Train (example, CPU, small run):

```powershell
Set-Location 'C:\\Users\\rajam\\Desktop\\Project'
python .\\train.py --data-dir .\\dataset --model resnet18 --batch-size 16 --epochs 5 --workers 0 --device cpu
```

This saves the best checkpoint to `checkpoints/best.pth` and always writes a final `checkpoints/last.pth`.

3) Inference on a folder or single image:

```powershell
# infer on all images in the Fake validation folder
python .\\infer.py --checkpoint .\\checkpoints\\best.pth --image .\\dataset\\val\\Fake --device cpu --model resnet18

# infer on a single image
python .\\infer.py --checkpoint .\\checkpoints\\best.pth --image .\\path\\to\\image.jpg --device cpu --model resnet18
```

Notes:
- `infer.py` now uses `utils.load_checkpoint` and supports multiple common checkpoint key names (`model_state`, `model_state_dict`, `state_dict`) as well as raw state_dict files.
- Adjust `--device` to `cuda` if you have a GPU and installed the CUDA build of PyTorch.
