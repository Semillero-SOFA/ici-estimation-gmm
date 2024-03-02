#!/usr/bin/env python
# coding: utf-8

# Libraries
import os

import matplotlib.pyplot as plt
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


# Load data
file_tx = f"{DATABASE_DIR}/Demodulation/Processed/2x16QAM_16GBd.csv"
folder_rx = f"{DATABASE_DIR}/Demodulation/Processed"

# Transmitted data
X_tx = np.array(pl.read_csv(file_tx))
X_txs = gmm_utils.split(X_tx, 12)

# Read received data
data = read_data(folder_rx)

# Try to load histograms
file_models_hist = f"{RESULTS_DIR}/models16_hist.pkl"
file_models_gmm = f"{RESULTS_DIR}/models16_gmm.pkl"

models_hist = sofa.joblib_load(file_models_hist)
models_gmm = sofa.joblib_load(file_models_gmm)
models_tuple = (
    None if models_hist is None or models_gmm is None else (
        models_hist, models_gmm)
)


def plot_histograms(
    data, histograms_gmm, spacing: None | str = None, osnr: None | str = None
):
    def plot(data, histograms_gmm, osnr, spacing):
        # Extract data
        X_ch = np.array(data[spacing][osnr])
        X_ch = X_ch[:, 0] + 1j * X_ch[:, 1]

        plt.figure(figsize=(12, 12), layout="tight")

        # Plot constellation diagram
        ax = plt.subplot(2, 2, 1)
        gmm_utils.plot_constellation_diagram(X_ch, ax)

        gm_2d = histograms_gmm.get(spacing).get(osnr)[0][0]

        # Plot 2D GMM
        gmm_utils.plot_gmm_2d(gm_2d, limits, ax)
        ax.grid(True)

        # Calculate 3D histogram
        hist, x_mesh, y_mesh = gmm_utils.calculate_3d_histogram(
            X_ch, bins, limits
        )

        # Plot 3D histogram
        ax = plt.subplot(2, 2, 2, projection="3d")
        gmm_utils.plot_3d_histogram(x_mesh, y_mesh, hist, ax)

        # Plot I and Q histograms separately
        # I
        ax = plt.subplot(2, 2, 3)
        gmm_utils.plot_1d_histogram(X_ch.real, bins=bins, ax=ax)

        hist_x, hist_y = gmm_utils.calculate_1d_histogram(X_ch.real, bins)
        input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
        gm_kwargs = {
            "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
            "n_components": 4,
        }
        gm_i = gmm_utils.calculate_gmm(input_data, gm_kwargs)
        gmm_utils.plot_gmm_1d(gm_i, limits, ax)

        ax.set_title("I-Histogram")
        ax.set_xlabel("I")
        ax.grid(True)

        # Q
        ax = plt.subplot(2, 2, 4)
        gmm_utils.plot_1d_histogram(X_ch.imag, bins=bins, ax=ax)

        hist_x, hist_y = gmm_utils.calculate_1d_histogram(X_ch.imag, bins)
        input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
        gm_kwargs = {
            "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
            "n_components": 4,
        }
        gm_q = gmm_utils.calculate_gmm(input_data, gm_kwargs)
        gmm_utils.plot_gmm_1d(gm_q, limits, ax)
        ax.set_title("Q-Histogram")
        ax.set_xlabel("Q")
        ax.grid(True)

        plt.suptitle(f"Plots for {osnr} OSNR and {spacing} of spacing")

        file_osnr, file_spacing = osnr.replace(
            ".", "p"), spacing.replace(".", "p")
        fig_name = f"{RESULTS_DIR}/plot_feats_{file_osnr}_{file_spacing}.svg"
        plt.savefig(fig_name)

    bins = 128
    limits = [-5, 5]

    if spacing is not None and osnr is not None:
        plot(data, histograms_gmm, osnr, spacing)

    elif spacing is None and osnr is None:
        spacings = [f"{x}GHz" for x in [
            "15", "15.5", "16", "16.5", "17", "17.6", "18"]]
        for spacing in spacings:
            for osnr in data[f"{spacing}GHz"]:
                plot(data, histograms_gmm, osnr, spacing)
    else:
        raise ValueError


def plot_menu(data):
    def select_menu(data, name):
        print(f"Select {name}")
        options = {n: option for n, option in enumerate(data.keys())}
        for n in options.keys():
            print(f"[{n}]: {options[n]}")
        choice = None
        while choice is None:
            choice = options.get(int(input()))
            if choice is None:
                print("Invalid input, try again")
        print(f"Selected {choice}")
        return choice

    # Spacing
    spacing = select_menu(data, "spacing")

    # OSNR
    osnr = select_menu(data[spacing], "OSNR")

    plot_histograms(data, models_gmm, spacing, osnr)


plot_menu(data)
