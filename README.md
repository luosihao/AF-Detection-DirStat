## Introduction

Welcome to the repository for our novel End-to-End Atrial Fibrillation Detection Model, inspired by directional statistics. This model leverages Sobolev test statistics to quantify irregularities and variability in  R-R intervals. It's designed for interpretability, computational efficiency, and excellent generalization capabilities. 

## Model Description

`My_e2e.py` contains a PyTorch model designed for processing RRI.

Note: We have provided an example script for training a model, named `training_model_example.py`. You will need to modify the file path and the method of data reading to accommodate your own dataset.

## Input Format

The expected input for the model is an RRI (normalized with frequency, converted to a numerical value in seconds) PyTorch tensor with the shape (batch_size, 1, time_series_length), where:

- `batch_size` is the number of samples in your batch.
- `1` represents a single-channel input, as typically required for time-series data.
- `time_series_length` is the length of your time series data for each sample.

The model parameters saved in `Saved_weights` are adapted for a `time_series_length` of 32. You are free to use a different `time_series_length` if you are training the model yourself.



## Citation

If you use this model or scripts in your research, please cite our paper:



```
Luo, C., Zhang, K., Hu, Y., Li, X., Cao, S., Jin, Y., Ren, P., Rao, N. (2024). 
Directional statistics-inspired end-to-end atrial fibrillation detection model based on ECG rhythm. 
Expert Systems with Applications, 123112. https://doi.org/10.1016/j.eswa.2023.123112
```

For any further questions or issues, please open an issue in this repository or contact us via email at [luosicheng945@gmail.com](mailto:luosicheng945@gmail.com), and we will get back to you as soon as possible.
