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

## Motivation for CNNs and the Transition to Vision Transformers

CNNs were chosen as the initial modeling approach because they are highly effective at extracting local spatial patterns in geophysical data such as temperature fields. Their inductive bias—spatial locality and translation invariance—makes them well-suited for tasks like short-term forecasting and seasonal classification, where fine-grained structures (fronts, gradients, small-scale anomalies) carry meaningful signals.
However, during training of the CNN-based predictor, signs of mild overfitting emerged at longer horizons (e.g., at 21 hours), where the training loss continued to decrease while the validation loss began to rise. This suggests that the model’s reliance on local spatial priors may limit its ability to generalize when longer-range dependencies become dominant.
Vision Transformers (ViTs) address this limitation by replacing convolutional locality with global self-attention, enabling the model to learn relationships across the entire temperature field. The improved performance of the ViT at the 21-hour horizon highlights the advantage of models capable of capturing global spatial context without being constrained by convolutional receptive fields.


## Some Results

The following summarizes the performance of the models implemented in this repository.  
All improvements are measured with respect to a **trivial persistence baseline** namely predicting the future temperature map as identical to the latest observed one.

### CNN Classifier — Temporal Attribution
A CNN classifier was trained to infer the month of the year from a single temperature map, effectively converting weather maps into a temporal-classification dataset.

- **Result:** Achieved **99.32% accuracy** on the test set.

![classifier_pred](images/classifier_pred.png)

### U-Net Temperature Forecasting
U-Net–based architectures were trained on NetCDF temperature data to predict future maps at multiple horizons.

- **Result:** Improved **9-hour forecasts by 22%** relative to the persistence baseline.

![CNN_pred](images/CNN_pred.png)
![CNN_RMSE](images/CNN_RMSE.png)


### Vision Transformer Forecasting
A Vision Transformer architecture was tested on the same forecasting task.

- **Result:** Achieved a **27% improvement** over the baseline for **21-hour forecasts**.

![ViT_pred](images/ViT_pred.png)
![ViT_RMSE](images/ViT_RMSE.png)


## Contact

If you want help reproducing results, converting these notebooks into scripts, or preparing a cleaned demo for GitHub Pages, open an issue or contact me.
