# WeatherVisionNN

This repository contains a set of Jupyter notebooks and helper scripts implementing Neural Network architectures (CNNs and Vision Transformers) for weather data analysis and prediction. The models work with air temperature data, demonstrating both classification (predicting months from temperature maps) and prediction (forecasting future temperatures) tasks using state-of-the-art deep learning architectures.

## Contents

- `CNN_predictor.ipynb` — U-Net based predictor (convolutional encoder-decoder)
  - Downloads a NetCDF temperature dataset (via `huggingface_hub`).
  - Preprocesses the data into input / target pairs for prediction horizons (1, 3 and 7 timesteps ahead).
  - Builds a U-Net style convolutional model and trains separate models for each horizon.
  - Tracks training and test losses, computes RMSE, and plots per-pixel RMSE maps and sample predictions.

- `CNN_classifier.ipynb` — CNN classifier for month prediction
  - Loads the same temperature dataset and builds a classification dataset where each sample is labeled with its month (0..11).
  - Defines a small CNN classifier and trains it to predict the month from a single temperature map.
  - Includes train/validation/test split, training loop, accuracy evaluation and visualization helpers.
  - Uses the `plotlib.py` helper to display a grid of maps and predicted/true month labels.

- `Transformer.ipynb` — Vision Transformer experiments
  - Implements components of a Vision Transformer (Multi-Head Attention, patch embedding, ViT model) adapted for the weather maps.
  - Prepares data, trains and evaluates a transformer-based model, and visualizes results (loss curves, example predictions and error maps).
  - Checkpoints for experiments may be present in the repo (e.g. `vit_best.pth`, `vit_final.pth`).

- `plotlib.py` — plotting helper
  - Utility for visualizing batches of maps with predicted and true month labels. Used by `CNN_classifier.ipynb`.

## Quick start / Installation

1. (Recommended) Create and activate a Python virtual environment:

   python -m venv .venv
   source .venv/bin/activate

2. Install the Python dependencies listed in `requirements.txt`:

   pip install -r requirements.txt

Notes about PyTorch:
- The notebooks use `torch`. For best support of CUDA or Apple MPS, install the `torch` wheel that matches your platform following the official installation instructions — the simple `pip install -r requirements.txt` will attempt to install a compatible CPU/CUDA build but you may prefer to follow the instructions on https://pytorch.org to choose the correct build for GPU or MPS acceleration.

## Usage

- Start Jupyter Lab / Notebook from the repository root:

  jupyter lab

- Open the notebook you want to run (`CNN_predictor.ipynb`, `CNN_classifier.ipynb`, or `Transformer.ipynb`).

- Each notebook contains cells that download the dataset automatically (via `huggingface_hub`) and perform preprocessing. Run the cells in order. Long training cells are implemented with progress reporting and plotting.

## Notes & tips

- Data download: the notebooks download a NetCDF (air temperature) file at runtime. Ensure you have an internet connection the first time you run them.
- Hardware: training may be slow on CPU. If you have a CUDA GPU or Apple Silicon with MPS, the notebooks automatically select `cuda` or `mps` when available.
- Checkpoints: if you have pre-trained checkpoints (`vit_best.pth`, `vit_final.pth`) in the repository, the Transformer notebook may load them — check the notebook cells for exact filenames and loading logic.

## Files to consider adding to `.gitignore` before publishing

- Dataset files downloaded at runtime (if you want to avoid checking them into Git): `*.nc`
- Large model checkpoints: `*.pth`, `*.pt`
- Virtual environment: `.venv/`

## Contact

If you want help reproducing results, converting these notebooks into scripts, or preparing a cleaned demo for GitHub Pages, open an issue or contact the repository owner.
