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
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/gmm_16GBd_regression/features"
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
    data, histograms_gmm, spacing: str | None = None, osnr: str | None = None
):
    def new_plot(size, three_dim=False):
        # New figure
        fig = plt.figure(figsize=(10, 10), layout="tight")
        ax = fig.add_subplot(projection='3d' if three_dim else None)

        plt.rc('font', size=size)
        plt.rc('axes', labelsize=size)
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        plt.rc('legend', fontsize=size)
        plt.rc('figure', titlesize=size)

        return ax

    def save_plot(name, file_spacing, file_osnr):
        fig_name = f"{RESULTS_DIR}/feats_{name}_{file_spacing}_{file_osnr}.svg"
        plt.savefig(fig_name)
        plt.close()

    def plot(data, histograms_gmm, osnr, spacing):
        # Extract data
        X_ch = np.array(data[spacing][osnr])
        X_ch = X_ch[:, 0] + 1j * X_ch[:, 1]

        file_osnr, file_spacing = osnr.replace(
            ".", "p"), spacing.replace(".", "p")
        size = 20

        # Plot constellation diagram
        ax = new_plot(size)
        reduced_factor = 10
        reduced_X_ch = X_ch[::reduced_factor]
        gmm_utils.plot_constellation_diagram(reduced_X_ch, ax)

        # Extract the real and imaginary parts
        real_parts = [value.real for value in sofa.MOD_DICT.values()]
        imaginary_parts = [value.imag for value in sofa.MOD_DICT.values()]

        # Create the scatter plot
        plt.scatter(real_parts, imaginary_parts, marker='x', color='yellow',
                    label="TX Means", s=200, linewidths=5)

        gm_2d = histograms_gmm.get(spacing).get(osnr)[0][0]

        # Plot 2D GMM
        gmm_utils.plot_gmm_2d(gm_2d, limits, ax)
        ax.grid(True)
        ax.legend(loc="upper right")

        save_plot("const_diag", file_spacing, file_osnr)

        # Calculate 3D histogram
        hist, x_mesh, y_mesh = gmm_utils.calculate_3d_histogram(
            X_ch, bins, limits
        )

        # New figure
        ax = new_plot(size, three_dim=True)

        # Plot 3D histogram
        gmm_utils.plot_3d_histogram(x_mesh, y_mesh, hist, ax)
        save_plot("3d_hist", file_spacing, file_osnr)

        # Plot I and Q histograms separately
        # I
        # New figure
        ax = new_plot(size)
        gmm_utils.plot_1d_histogram(X_ch.real, bins=bins, ax=ax)

        hist_x, hist_y = gmm_utils.calculate_1d_histogram(X_ch.real, bins)
        input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
        gm_kwargs = {
            "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
            "n_components": 4,
        }
        gm_i = gmm_utils.calculate_gmm(input_data, gm_kwargs)
        gmm_utils.plot_gmm_1d(gm_i, limits, ax)

        ax.set_xlabel("I")
        ax.grid(True)
        save_plot("2d_hist_I", file_spacing, file_osnr)

        # Q
        # New figure
        ax = new_plot(size)
        gmm_utils.plot_1d_histogram(X_ch.imag, bins=bins, ax=ax)

        hist_x, hist_y = gmm_utils.calculate_1d_histogram(X_ch.imag, bins)
        input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
        gm_kwargs = {
            "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
            "n_components": 4,
        }
        gm_q = gmm_utils.calculate_gmm(input_data, gm_kwargs)
        gmm_utils.plot_gmm_1d(gm_q, limits, ax)
        ax.set_xlabel("Q")
        ax.grid(True)
        save_plot("2d_hist_Q", file_spacing, file_osnr)

    bins = 128
    limits = [-5, 5]

    if spacing is not None and osnr is not None:
        plot(data, histograms_gmm, osnr, spacing)

    elif spacing is None and osnr is None:
        for spacing in data.keys():
            for osnr in data[spacing]:
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


# plot_menu(data)
plot_histograms(data, models_gmm)

print("GMM plots generated succesfully")
