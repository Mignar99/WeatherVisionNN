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

### Why CNNs First — and Why Transition to Vision Transformers

**CNNs were the natural starting point**: they excel at capturing **local spatial patterns** in temperature fields, leveraging strong inductive biases such as **spatial locality** and **translation invariance**. These properties make them particularly effective for  
- short-term temperature forecasting  
- season/month classification  
- detecting fine-scale structures (fronts, gradients, local anomalies)

**However, CNN forecasting showed early signs of overfitting at longer horizons** (notably at **21 hours**):  
- training loss kept decreasing  
- validation loss began to increase  
This indicates that purely local feature extraction limits the model when **large-scale atmospheric structures** drive the evolution of temperature.

**Vision Transformers** overcome this limitation by using **global self-attention**, allowing the model to capture relationships **across the entire spatial field**.  
This makes ViTs better suited for medium-range forecasting, where global context matters more than local texture.  
The result: **the ViT outperformed the CNN at 21-hour predictions**, validating the transition to a more globally aware architecture.



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
