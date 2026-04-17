### Glucose Data Preprocessing for PPGR Analysis

This repository contains the preprocessing script used to analyze glucose data and isolate Postprandial Glucose Response (PPGR) signals. The script is implemented as a Jupyter notebook and is intended to be used with data from the following Zenodo repository: at https://doi.org/10.5281/zenodo.15196914 (https://doi.org/10.5281/zenodo.15196914).

## Dataset: Zenodo - T1D-UOM – A Longitudinal Multimodal Dataset of Type 1 Diabetes

## Repository Contents
Preprocessing.ipynb – The main notebook that:
Loads and cleans the glucose monitoring dataset.
Extracts key glucose dynamics and features.
Identifies and isolates PPGR episodes from continuous glucose monitoring (CGM) data.
Performs basic statistical analysis and preprocessing for downstream modeling.

## Overview of the Workflow
* Data Loading
		Reads in data files from the Zenodo dataset.
* Signal Processing
		Detects peaks and relevant features using scipy.signal.
		Computes deltas, durations, and other metrics for glucose 		excursions.
* Data Normalization and Scaling
		Supports standardization (StandardScaler) and min-max 		scaling (MinMaxScaler) for model-ready features.
* Feature Engineering
		Calculates measures like peak glucose, area under the 		curve (AUC), time to peak, and more.
* Visualization
		Provides summary plots for glucose trends and PPGR 			profiles.

## Requirements
Make sure to install the following Python packages before running the notebook:
pip install pandas numpy matplotlib seaborn scikit-learn scipy

## Using the Dataset
Download the data files directly from the Zenodo DOI and place them in a directory accessible to the notebook. Update the relevant file paths in Preprocessing.ipynb accordingly.

## Getting Started
To run the notebook:
* Clone this repository or download the files.
* Download the dataset from Zenodo.
* Launch Jupyter Notebook or JupyterLab.
* Open and run Preprocessing.ipynb cell-by-cell.

## Notes
The notebook includes in-line comments and visualizations to help interpret results.
Designed as a starting point for more advanced modeling (e.g., meal response prediction, clustering, or personalized nutrition).
