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
FILENAME = os.path.basename(__file__)[:-3]

# Create a logger for this script
logger = sofa.setup_logger(FILENAME)

# Create results directory if it doesn't exist
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/gmm_32GBd_regression/features"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def initialize_histograms():
    # Return 6 levels deep dictionary
    return defaultdict(
        defaultdict(
            defaultdict(
                defaultdict(
                    defaultdict(
                        defaultdict(list).copy
                    ).copy
                ).copy
            ).copy
        ).copy
    )


def read_data(folder_rx):
    data = initialize_histograms()

    for dist_pow in os.listdir(folder_rx):
        if not os.path.isdir(os.path.join(folder_rx, dist_pow)):
            continue
        logger.info(f"Reading {dist_pow}")
        for spac in os.listdir(os.path.join(folder_rx, dist_pow)):
            logger.info(f"Reading {dist_pow}/{spac}")
            consts = os.listdir(os.path.join(folder_rx, dist_pow, spac))
            for const in consts:
                if const.endswith("xlsx"):
                    continue
                song, orth, osnr, spacing, distance, power = const.split("_")
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
logger.info("Reading data...")
data = read_data(folder_rx)

# Try to load histograms
file_models_hist = f"{RESULTS_DIR}/models32_hist.pkl"
file_models_gmm = f"{RESULTS_DIR}/models32_gmm.pkl"

logger.info("Trying to load features...")
models_hist = sofa.joblib_load(file_models_hist)
models_gmm = sofa.joblib_load(file_models_gmm)
models_tuple = (
    None if models_hist is None or models_gmm is None else (
        models_hist, models_gmm)
)


def plot_histograms(data, histograms_gmm,
                    distance=None, power=None, spacing=None,
                    osnr=None, song=None, orth=None):
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

    def plot(data, histograms_gmm, distance, power, spacing, osnr, song, orth):
        logger.info(f"Plotting for constellation at {distance} with {power} launch power, {spacing} spectral spacing, {osnr} OSNR, Song {song[-1]}, {orth} component. ")
        # Extract data
        X_ch = data[distance][power][spacing][osnr][song][orth]
        file_osnr, file_spacing = osnr.replace(
            ".", "p"), spacing.replace(".", "p")
        size = 20

        # Preprocess data, reducing number of points
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
        plot(data, histograms_gmm, distance, power, spacing, osnr, song, orth)

    else:
        for distance in data.keys():
            for power in data[distance].keys():
                for spacing in data[distance][power].keys():
                    for osnr in data[distance][power][spacing].keys():
                        for song in data[distance][power][spacing][osnr].keys():
                            for orth in data[distance][power][spacing][osnr][song].keys():
                                plot(data, histograms_gmm, distance,
                                     power, spacing, osnr, song, orth)


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
                logger.error("Invalid input, try again")
        logger.info(f"Selected {choice}")
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

logger.info("GMM plots generated succesfully")
