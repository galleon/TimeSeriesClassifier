# Project Name

This repository contains the source code for a PyTorch implementation of a Time-Serie classifier model.

## Dataset

For training we used the UCI HAR Dataset, which can be downloaded from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip).
We used the raw data: total acceleration, body acceleration, and body gyroscope. Each has three axes of data. This means that there are a total of nine variables for each time step.
The data is split into time windows containing 128 time steps. The observations where recorded at 50Hz, so each time step is 0.02 seconds long. The time windows are overlapping, so there is 50% overlap between them.

The classes/labels are the following:
```
    - 0 Walking
    - 1 Walking upstairs
    - 2 Walking downstairs
    - 3 Sitting
    - 4 Standing
    - 5 Laying
```

# API
There is no API for this package.

# Setup instructions & Usage

To install the package, run the following commands:
```bash
# Clone the repository
git clone https://github.com/gallon/TimeSeriesClassifier.git
# Setup the environment
pyenv virtualenv 3.8.13 TimeSeriesClassifier
pyenv local TimeSeriesClassifier
pip install -e .
```

# Usage

To run the experiments, run the following commands:
```bash
# Install the notebook dependencies
pip install -qU pip ipython jupyter
# Run the notebook
jupyter notebooks/TimeSeriesClassifier.ipynb
```
