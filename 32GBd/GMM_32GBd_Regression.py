#!/usr/bin/env python
# coding: utf-8

# # Inter-channel interference (ICI) estimation using constellation diagrams Gaussian Mixture Models in a 32 GBd system.

# Libraries
import os
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pathlib import Path
import scipy as sp
from tensorflow.keras import utils

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


folder_rx = f"{DATABASE_DIR}/Estimation/32GBd/ICI characterization"

# Read received data
data = read_data(folder_rx)

# Try to load histograms
file_models_hist = f"{RESULTS_DIR}/models32_hist.pkl"
file_models_gmm = f"{RESULTS_DIR}/models32_gmm.pkl"

models_hist = sofa.joblib_load(file_models_hist)
models_gmm = sofa.joblib_load(file_models_gmm)
models_tuple = (
    None if models_hist is None or models_gmm is None else (
        models_hist, models_gmm)
)

# Calculate histograms


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
                            print(f"Calculating GMM for {
                                  distance}/{power}/{spacing}/{osnr}/{song}/{orth}")
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
                                gm_q = gmm_utils.calculate_gmm(
                                    input_data, gm_kwargs)

                                # Save gaussians
                                histograms_gmm[distance][power][spacing][osnr][song][orth].append([
                                                                                                  gm_2d, gm_i, gm_q])

    histograms_hist = dict(histograms_hist)
    histograms_gmm = dict(histograms_gmm)
    return histograms_hist, histograms_gmm


models_tuple = gmm_utils.calc_once(
    "models_tuple", get_histograms, {"data": data})
models_hist, models_gmm = models_tuple
sofa.joblib_save(models_hist, file_models_hist)
sofa.joblib_save(models_gmm, file_models_gmm)


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

        plt.suptitle(f"Plots for constellation at {distance} with {power} launch power, {
                     spacing} spectral spacing, {osnr} OSNR, Song {song[-1]}, {orth} component.")

        fig_name = f"{
            RESULTS_DIR}/plot_features_{osnr.replace('.', 'p')}_{spacing.replace('.', 'p')}.svg"
        plt.savefig(fig_name)
        plt.show()

    bins = 128
    limits = [-5, 5]

    plot(data, histograms_gmm, distance, power, spacing, osnr, song, orth)


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


# Plot constellation diagrams, contour for GMM and histograms
plot_menu(data)

# Pre-process data
# Dataframe with 98 columns
# First 32 for means
# Following 64 for each value of the covariances matrixes
# (repeated values are included)
# Next to last for OSNR value in dB
# Last column for spectral spacing value in GHz

n_features = 82
df_dict = {f"col{n}": [] for n in range(n_features)}
data_list = []

# Iterate over the dictionary and populate the DataFrame
for distance in models_gmm.keys():
    for power in models_gmm[distance].keys():
        for spacing in models_gmm[distance][power].keys():
            for osnr in models_gmm[distance][power][spacing].keys():
                for song in models_gmm[distance][power][spacing][osnr].keys():
                    for orth in models_gmm[distance][power][spacing][osnr][song].keys():
                        for n in range(3):
                            gmm_2d = models_gmm[distance][power][spacing][osnr][song][orth][n][0]
                            means = gmm_2d.means_.flatten()
                            covariances_raw = gmm_2d.covariances_.flatten()
                            # Remove repeated covariances
                            covariances = []
                            for x, covariance in enumerate(covariances_raw):
                                if x % 4 != 1:
                                    covariances.append(covariance)
                            osnr_value = np.array([float(osnr[:-2])])
                            spacing_value = np.array([float(spacing[:-3])])

                            features = np.concatenate(
                                (means, covariances, osnr_value, spacing_value))
                            row_dict = {f"col{n}": feature for n,
                                        feature in enumerate(features)}
                            data_list.append(row_dict)

# Convert the list of dictionaries into a DataFrame
df = pl.DataFrame(data_list)

# Print the DataFrame
df.write_json(f"{RESULTS_DIR}/gmm32_features.json")

# Shuffle the dataframe
df_shuffled = df.sample(n=len(df), shuffle=True, seed=1036681523)

# ## Hyperparameters evaluation

# The following hyperparameters are going to be combined and evaluated:
# - Maximum number of neurons in the first layer (8, 16, 32, 64, 128, 256, 512, 1024).
# - Number of hidden layers (1, 2, 3).
# - Activation functions (ReLu, tanh, sigmoid).
# - Using or not the OSNR value as an additional feature.
#
# Results will have the following structure:
# ```
# {"xyz": {"n_neurons": {"osnr": results}}}
# ```
# Where `xyz` will be each initial of the activation functions in the model (r for ReLu, t for tanh and s for sigmoid), `n_neurons` will be the maximum number of neurons in the model (corresponding to the first layer), `osnr` will be a string telling if that model used OSNR as input or not (`"osnr"` or `wo_osnr`).
# Finally the results will store the loss history, the serialized model in JSON format in a string and MAE, RMSE and R² values for training, test and production data.
osnr_lst = ["osnr", "wo_osnr"]
max_neurons = [str(2**n) for n in range(3, 11)]
functs = ["relu", "tanh", "sigmoid"]
layers_n = [1, 2, 3]

combinations = [
    [list(subset) for subset in product(functs, repeat=n)] for n in layers_n
]

hidden_layers = [item for sublist in combinations for item in sublist]

# Training
results_file = f"{RESULTS_DIR}/gmm32_reg_results.h5"
try:
    histograms_reg_results = sofa.load_hdf5(results_file)
except:
    print("Error loading from file, creating a new dictionary")
    histograms_reg_results = defaultdict(
        defaultdict(defaultdict(defaultdict().copy).copy).copy
    )

# Evaluar
for activations in hidden_layers:
    for neurons in max_neurons:
        for osnr in osnr_lst:
            args = {
                "data": df_new,
                "data_prod": df_prod,
                "n_splits": 5,
                "max_neurons": int(neurons),
                "activations": activations,
                "use_osnr": True if osnr == "osnr" else False,
            }
            act_fn_name = "".join([s[0] for s in activations])
            if histograms_reg_results[act_fn_name][neurons][osnr] == defaultdict():
                # Get results
                results = gmm_utils.test_estimation_model(**args)
                # Serialize model
                results["model"] = [
                    utils.serialize_keras_object(model) for model in results["model"]
                ]
                # Save serialized model for serialization
                histograms_reg_results[act_fn_name][neurons][osnr] = results
                # Save results with serialized model
                print("Saving results...")
                sofa.save_hdf5(histograms_reg_results, results_file)
                print("Results saved!")
print("Training complete")


# Results
# Neurons
gmm_neurons_avg_results = [
    np.mean(
        gmm_utils.get_avg_score(
            histograms_reg_results,
            neurons,
            target="neurons",
            metric="mae",
            score="test",
        )
    )
    for neurons in max_neurons
]
neurons_filename = Path(f"{RESULTS_DIR}/gmm_neurons_avg_results.json")
sofa.save_json(gmm_neurons_avg_results,
               neurons_filename)
gmm_utils.plot_results(neurons, gmm_neurons_avg_results, neurons_filename,
                       "Maximum number of neurons", log=True)

# Layers
gmm_layers_avg_results = [
    np.mean(
        gmm_utils.get_avg_score(
            histograms_reg_results,
            layers,
            target="layers",
            metric="mae",
            score="test"
        )
    )
    for layers in range(1, 4)
]
layers_filename = Path(f"{RESULTS_DIR}/gmm_layers_avg_results.json")
sofa.save_json(gmm_layers_avg_results,
               layers_filename)

x = range(1, 4)
gmm_utils.plot_results(x, gmm_layers_avg_results,
                       "Number of layers", log=False, intx=True)

# OSNR
gmm_osnr_avg_results = [
    np.mean(
        gmm_utils.get_avg_score(
            histograms_reg_results,
            osnr,
            target="osnr",
            metric="mae",
            score="test"
        )
    )
    for osnr in ["osnr", "wo_osnr"]
]
osnr_filename = Path(f"{RESULTS_DIR}/gmm_osnr_avg_results.json")
sofa.save_json(gmm_osnr_avg_results,
               osnr_filename)

# Sort models by score
better_models_df = gmm_utils.get_better_models(
    histograms_reg_results, metric="mae", score="test"
)
better_models_df.write_json(
    f"{RESULTS_DIR}/gmm_32GBd_models_summary.json")
better_models_df.head(25).write_json(
    f"{RESULTS_DIR}/gmm_32GBd_better_models.json")
better_models_df.tail(25).write_json(
    f"{RESULTS_DIR}/gmm_32GBd_worst_models.json")
