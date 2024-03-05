#!/usr/bin/env python
# coding: utf-8

# Libraries
import os
from collections import defaultdict

import matplotlib.pyplot as plt
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


def plot_histograms(data, histograms_gmm, distance, power, spacing, osnr, song, orth):
    def plot(data, histograms_gmm, distance, power, spacing, osnr, song, orth):
        # Extract data
        X_ch = data[distance][power][spacing][osnr][song][orth]

        plt.figure(figsize=(12, 12), layout="tight")

        # Plot constellation diagram
        ax = plt.subplot(2, 2, 1)
        gmm_utils.plot_constellation_diagram(X_ch, ax)

        gm_2d = histograms_gmm[distance][power][spacing][osnr][song][orth][0][0]

        # Plot 2D GMM
        gmm_utils.plot_gmm_2d(gm_2d, limits, ax)
        ax.grid(True)

        # Calculate 3D histogram
        hist, x_mesh, y_mesh = gmm_utils.calculate_3d_histogram(
            X_ch, bins, limits)

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

        plt.suptitle(f"Plots for constellation at {distance} with {power} launch power, {spacing} spectral spacing, {osnr} OSNR, Song {song[-1]}, {orth} component.")

        file_spacing = spacing.replace(".", "p")
        file_osnr = osnr.replace(".", "p")
        fig_name = f"{RESULTS_DIR}/plot_features_{distance}_{power}_{file_spacing}_{file_osnr}_{song}_{orth}.svg"
        plt.savefig(fig_name)

    bins = 128
    limits = [-5, 5]

    if spacing is not None and osnr is not None:
        plot(data, histograms_gmm, distance, power, spacing, osnr, song, orth)

    elif spacing is None and osnr is None:
        for spacing in data.keys():
            for osnr in data[spacing]:
                plot(data, histograms_gmm, distance, power, spacing, osnr, song, orth)
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

    # Distance
    distance = select_menu(data, "distance")

    # Power
    power = select_menu(data[distance], "launch power")

    # Spacing
    spacing = select_menu(data[distance][power], "spacing")

    # OSNR
    osnr = select_menu(data[distance][power][spacing], "OSNR")

    # Song
    song = select_menu(data[distance][power][spacing][osnr], "song")

    # Component
    orth = select_menu(data[distance][power][spacing][osnr][song], "component")

    plot_histograms(data, models_gmm, distance,
                    power, spacing, osnr, song, orth)


# plot_menu(data)
plot_histograms(data, models_gmm)

print("GMM plots generated succesfully")
