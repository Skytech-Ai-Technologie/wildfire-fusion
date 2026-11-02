# Reproducibility — exact commands that produced every artifact in this folder

Environment: Python 3.11.9, PyTorch 2.14.1+cu130, Ultralytics 8.4.170, NVIDIA RTX 5070
Laptop GPU (8 GB), Windows 11. Full versions in `environment.txt`. All commands run
from the repository root `C:\Users\LENOVO\Desktop\forfire1\wildfire-fusion`.

```bat
:: 0. environment
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130

:: 1. datasets (provenance markers written to data\<name>\.done)
python scripts\get_datasets.py --dataset dfire
python scripts\get_datasets.py --dataset figlib
python scripts\get_datasets.py --dataset flame

:: 2. leakage-safe splits (32-frame numeric-stem buckets; zero group straddling asserted)
python scripts\build_splits.py --dataset dfire

:: 3. training — single-stream YOLOv8n, 150 epochs (this run: runs\single_rgb_dfire_1791130238)
python src\train_perceptual.py --dataset dfire --arch single --mode rgb

:: 4. benchmark — Table IV on the held-out test split
python benchmark\run_benchmark.py --table iv --dataset dfire ^
    --weights runs\single_rgb_dfire_1791130238\weights\best.pt

:: 5. contextual tier on real embeddings
python -c "import sys; sys.path.insert(0,'.'); from src.train_contextual import extract_embeddings; extract_embeddings(r'runs\single_rgb_dfire_1791130238\weights\best.pt', [l.strip() for l in open(r'data\dfire\train.txt')], r'runs\emb\dfire_frames.npz')"
python -c "import sys; sys.path.insert(0,'.'); from src.train_contextual import build_windows; build_windows(r'data\dfire\split_manifest.csv', r'runs\emb\dfire_frames.npz', r'runs\emb\windows.npz', 32)"
python src\train_contextual.py --embeddings runs\emb\windows.npz

:: 6. synthetic field-sim tables (Tables V / VI — labeled SYNTHETIC)
python benchmark\run_benchmark.py --table v --events 3
python benchmark\ablation.py --events 3

:: 7. regenerate every artifact in this folder
python scripts\generate_paper_artifacts.py
```

Split-generation seed: 42 (`build_splits.py --seed 42` default). Checkpoint:
`runs\single_rgb_dfire_1791130238\weights\best.pt`. Training config logged alongside
the run in `runs\single_rgb_dfire_1791130238\args.yaml`.
