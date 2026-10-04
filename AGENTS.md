# Base44 Dev Environment

This repo is a collection of academic AI/ML projects — Python scripts and Jupyter
notebooks. There is **no web server**. The Base44 preview runs a **JupyterLab**
server on host port 3000 (container port 8888) with the repo bind-mounted at
`/work`, so notebooks can be opened and run in the browser preview.

## Running

```
docker compose -f docker-compose.base44.yml up -d --build
```

- JupyterLab UI: `http://localhost:3000/lab` (no token / no password — open server).
- Config: `.base44/jupyter_lab_config.py` (auth + XSRF + CSP disabled for the
  cross-origin preview iframe).
- Image: `Dockerfile.base44` (python:3.12-slim + jupyterlab + numpy/pandas/
  matplotlib/seaborn/scikit-learn/pygame). Source is bind-mounted, so edits are
  live; only dependency changes require a rebuild.

## Projects

1. **DNA Center Finding** (`Azizi.DNA.py`, `Simulated_Annealing/DNA_Center.ipynb`)
   — Hill Climbing + Simulated Annealing. Fully self-contained (generates random
   DNA). **This is the notebook that runs end-to-end in the preview.**
2. **Linear Regression** (`Azizi.regression.py`, `Linear_Regression/`)
   — Reads `./NASA_JPL_Dataset.csv`, which is **not in the repo** (see
   `Linear_Regression/dataset_url.txt` for a Google Drive link). The notebook
   `Linear_Regression/linear_regression.ipynb` is an empty 0-byte file (invalid
   notebook). Download the CSV next to `Azizi.regression.py` to run it.
3. **Search Algorithms** (`Search_Algorithms/`, `Azizi.Algoreitm.ipynb`)
   — PyGame grid search (Angry Birds: Star Wars). `env.play()` opens a real
   PyGame window via `pygame.display.set_mode`, which **cannot render in a
   headless browser iframe**. The notebook/algorithms are viewable, but the
   visual `play()` runner needs a local desktop with a display.

## Verifying it works

```
docker compose -f docker-compose.base44.yml ps
curl -fsS http://localhost:3000/api/status
```
Then open the preview and run the `Simulated_Annealing/DNA_Center.ipynb` cells.

## Notes

- No external secrets are required.
- Jupyter security (token, password, XSRF, CSP frame-ancestors) is intentionally
  disabled for the preview — this is a local dev environment only.
