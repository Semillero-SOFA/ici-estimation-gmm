#!/usr/bin/env python
# coding: utf-8

# Libraries
import os
from collections import defaultdict
from itertools import product

import numpy as np
from pathlib import Path
import scipy as sp

import sofa
import gmm_utils

# Globals
LOCAL_ROOT = sofa.find_root()
GLOBAL_ROOT = LOCAL_ROOT.parent
DATABASE_DIR = f"{GLOBAL_ROOT}/databases"
GLOBAL_RESULTS_DIR = f"{GLOBAL_ROOT}/results"

# Create results directory if it doesn't exist
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/gmm_32GBd_regression"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Special function to read the known data structure


def read_data(folder_rx):
    data = defaultdict(
        defaultdict(
            defaultdict(defaultdict(defaultdict(
                defaultdict().copy).copy).copy).copy
        ).copy
    )
    for dist_pow in os.listdir(folder_rx):
        if not os.path.isdir(os.path.join(folder_rx, dist_pow)):
            continue
        print(f"reading {dist_pow}")
        for spac in os.listdir(os.path.join(folder_rx, dist_pow)):
            print(f"reading {dist_pow}/{spac}")
            consts = os.listdir(os.path.join(folder_rx, dist_pow, spac))
            for const in consts:
                if const.endswith("xlsx"):
                    continue
                song, orth, osnr, spacing, distance, power = const.split("_")
                # Remove extension
                power = power.split(".")[0]
                spacing = spacing.replace("p", ".")
                osnr = osnr.replace("p", ".")
                mat = sp.io.loadmat(os.path.join(
                    folder_rx, dist_pow, spac, const))
                data[distance][power][spacing][osnr][song][orth] = mat["rconst"][0]
    return data


def get_histograms(data):
    histograms_hist = defaultdict(
        defaultdict(
            defaultdict(defaultdict(defaultdict(
                defaultdict(list).copy).copy).copy).copy
        ).copy
    )
    histograms_gmm = defaultdict(
        defaultdict(
            defaultdict(defaultdict(defaultdict(
                defaultdict(list).copy).copy).copy).copy
        ).copy
    )
    bins = 128
    limits = [-5, 5]

    for distance in data.keys():
        for power in data[distance].keys():
            for spacing in data[distance][power].keys():
                for osnr in data[distance][power][spacing].keys():
                    for song in data[distance][power][spacing][osnr].keys():
                        for orth in data[distance][power][spacing][osnr][song].keys():
                            print(f"Calculating GMM for: {distance}/{power}/{spacing}/{osnr}/{song}/{orth}")
                            X_rx = data[distance][power][spacing][osnr][song][orth]
                            X_chs = gmm_utils.split(X_rx, 3)

                            for n, x_ch in enumerate(X_chs):
                                # Calculate 2D GMM
                                input_data = np.vstack(
                                    (x_ch.real, x_ch.imag)).T
                                gm_kwargs = {
                                    "means_init": np.array(
                                        list(product([-3, -1, 1, 3], repeat=2))
                                    ),
                                    "n_components": 16,
                                }
                                gm_2d = gmm_utils.calculate_gmm(
                                    input_data, gm_kwargs)

                                # Calculate 3D histogram
                                hist, x_mesh, y_mesh = gmm_utils.calculate_3d_histogram(
                                    x_ch, bins, limits)

                                # Save 3D histogram
                                histograms_hist[distance][power][spacing][osnr][song][orth].append(
                                    hist)

                                # Calculate I and Q histograms
                                hist_x, hist_y = gmm_utils.calculate_1d_histogram(
                                    x_ch.real, bins)
                                input_data = np.repeat(
                                    hist_x, hist_y).reshape(-1, 1)
                                gm_kwargs = {
                                    "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
                                    "n_components": 4,
                                }
                                gm_i = gmm_utils.calculate_gmm(
                                    input_data, gm_kwargs)

                                # Q
                                hist_x, hist_y = gmm_utils.calculate_1d_histogram(
                                    x_ch.imag, bins)
                                input_data = np.repeat(
                                    hist_x, hist_y).reshape(-1, 1)
                                gm_kwargs = {
                                    "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
                                    "n_components": 4,
                                }
                                gm_q = gmm_utils.calculate_gmm(input_data,
                                                               gm_kwargs)

                                # Save gaussians
                                histograms_gmm[distance][power][spacing][osnr][song][orth].append([
                                                                                                  gm_2d, gm_i, gm_q])

    histograms_hist = dict(histograms_hist)
    histograms_gmm = dict(histograms_gmm)
    return histograms_hist, histograms_gmm


# Load data
folder_rx = f"{DATABASE_DIR}/Estimation/32GBd"

# Read received data
print("Reading data...")
data = read_data(folder_rx)

# Try to load histograms
file_models_hist = f"{RESULTS_DIR}/models32_hist.pkl"
file_models_gmm = f"{RESULTS_DIR}/models32_gmm.pkl"

print("Trying to load features...")
models_hist = sofa.joblib_load(file_models_hist)
models_gmm = sofa.joblib_load(file_models_gmm)
models_tuple = (
    None if models_hist is None or models_gmm is None else (
        models_hist, models_gmm)
)

# Calculate histograms
models_tuple = gmm_utils.calc_once("models_tuple", get_histograms,
                                   {"data": data})
models_hist, models_gmm = models_tuple
sofa.joblib_save(models_hist, file_models_hist)
sofa.joblib_save(models_gmm, file_models_gmm)

print("Features calculated succesfully")
