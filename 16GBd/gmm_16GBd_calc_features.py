#!/usr/bin/env python
# coding: utf-8

# Libraries
import os
from collections import defaultdict
from itertools import product

import numpy as np
import polars as pl
from pathlib import Path

import sofa
import gmm_utils

# Globals
LOCAL_ROOT = sofa.find_root()
GLOBAL_ROOT = LOCAL_ROOT.parent
DATABASE_DIR = f"{GLOBAL_ROOT}/databases"
GLOBAL_RESULTS_DIR = f"{GLOBAL_ROOT}/results"

# Create results directory if it doesn't exist
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/gmm_16GBd_regression"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Special function to read the known data structure


def read_data(folder_rx):
    data = {}
    # Read root directory
    for folder in os.listdir(folder_rx):
        # Check name consistency for subdirectories
        if folder.endswith("spacing"):
            # Extract "pretty" part of the name
            spacing = folder[:-8]
            data[spacing] = {}
            # Read each data file
            for file in os.listdir(f"{folder_rx}/{folder}"):
                # Check name consistency for data files
                if file.find("consY") != -1:
                    # Extract "pretty" part of the name
                    osnr = file.split("_")[2][5:-4]
                    # Initialize if not created yet
                    if data[spacing].get(osnr) is None:
                        data[spacing][osnr] = {}
                    # Set data
                    csv_file_data = pl.read_csv(f"{folder_rx}/{folder}/{file}")
                    data[spacing][osnr] = csv_file_data
    return data


def get_histograms():
    spacings = ["15", "15.5", "16", "16.5", "17", "17.6", "18"]

    histograms_hist = defaultdict(lambda: defaultdict(list))
    histograms_gmm = defaultdict(lambda: defaultdict(list))
    bins = 128
    limits = [-5, 5]

    for spacing in spacings:
        X_rx = data[f"{spacing}GHz"]
        for snr in X_rx:
            # Extract data
            X_ch = np.array(X_rx[snr])
            X_ch = X_ch[:, 0] + 1j * X_ch[:, 1]

            X_chs = gmm_utils.split(X_ch, 12)

            for n, x_ch in enumerate(X_chs):
                # Calculate 2D GMM
                input_data = np.vstack((x_ch.real, x_ch.imag)).T
                gm_kwargs = {
                    "means_init": np.array(list(product([-3, -1, 1, 3],
                                                        repeat=2))),
                    "n_components": 16,
                }
                gm_2d = gmm_utils.calculate_gmm(input_data, gm_kwargs)

                # Calculate 3D histogram
                hist, x_mesh, y_mesh = gmm_utils.calculate_3d_histogram(
                    x_ch, bins, limits
                )

                # Save 3D histogram
                histograms_hist[f"{spacing}GHz"][snr].append(hist)

                # Calculate I and Q histograms
                hist_x, hist_y = gmm_utils.calculate_1d_histogram(
                    x_ch.real, bins)
                input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
                gm_kwargs = {
                    "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
                    "n_components": 4,
                }
                gm_i = gmm_utils.calculate_gmm(input_data, gm_kwargs)

                # Q
                hist_x, hist_y = gmm_utils.calculate_1d_histogram(
                    x_ch.imag, bins)
                input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
                gm_kwargs = {
                    "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
                    "n_components": 4,
                }
                gm_q = gmm_utils.calculate_gmm(input_data, gm_kwargs)

                # Save gaussians
                histograms_gmm[f"{spacing}GHz"][snr].append(
                    [gm_2d, gm_i, gm_q])

    histograms_hist = dict(histograms_hist)
    histograms_gmm = dict(histograms_gmm)
    return histograms_hist, histograms_gmm


# Load data
file_tx = f"{DATABASE_DIR}/Demodulation/Processed/2x16QAM_16GBd.csv"
folder_rx = f"{DATABASE_DIR}/Demodulation/Processed"

# Transmitted data
X_tx = np.array(pl.read_csv(file_tx))
X_txs = gmm_utils.split(X_tx, 12)

# Read received data
print("Reading data...")
data = read_data(folder_rx)

# Try to load histograms
file_models_hist = f"{RESULTS_DIR}/models16_hist.pkl"
file_models_gmm = f"{RESULTS_DIR}/models16_gmm.pkl"

print("Trying to load features...")
models_hist = sofa.joblib_load(file_models_hist)
models_gmm = sofa.joblib_load(file_models_gmm)
models_tuple = (
    None if models_hist is None or models_gmm is None else (
        models_hist, models_gmm)
)

# Calculate histograms
models_tuple = gmm_utils.calc_once("models_tuple", get_histograms, {})
models_hist, models_gmm = models_tuple
sofa.joblib_save(models_hist, file_models_hist)
sofa.joblib_save(models_gmm, file_models_gmm)

print("Features calculated succesfully")
