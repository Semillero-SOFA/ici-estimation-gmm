#!/usr/bin/env python
# coding: utf-8

# Libraries
import os
import logging

from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import scipy as sp

import sofa
import gmm_utils

FILENAME = os.path.basename(__file__)[:-3]

# Create a logger for this script
logger = sofa.setup_logger(FILENAME)

# Globals
LOCAL_ROOT = sofa.find_root()
GLOBAL_ROOT = LOCAL_ROOT.parent
DATABASE_DIR = f"{GLOBAL_ROOT}/databases"
GLOBAL_RESULTS_DIR = f"{GLOBAL_ROOT}/results"

# Create results directory if it doesn't exist
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/gmm_32GBd_regression"
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


def calculate_gmm_2d(input_data):
    gm_kwargs = {
        "means_init": np.array(list(product([-3, -1, 1, 3], repeat=2))),
        "n_components": 16,
        "covariance_type": "full",
    }
    return gmm_utils.calculate_gmm(input_data, gm_kwargs)


def calculate_3d_histogram(x_ch, bins, limits):
    return gmm_utils.calculate_3d_histogram(x_ch, bins, limits)


def calculate_1d_gmms(x_ch, bins):
    gm_kwargs = {
        "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
        "n_components": 4,
        "covariance_type": "full",
    }
    hist_x, hist_y = gmm_utils.calculate_1d_histogram(x_ch.real, bins)
    input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
    gm_i = gmm_utils.calculate_gmm(input_data, gm_kwargs)

    hist_x, hist_y = gmm_utils.calculate_1d_histogram(x_ch.imag, bins)
    input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
    gm_q = gmm_utils.calculate_gmm(input_data, gm_kwargs)

    return gm_i, gm_q


def process_channel(x_ch, bins, limits):
    input_data = np.vstack((x_ch.real, x_ch.imag)).T
    gm_2d = calculate_gmm_2d(input_data)
    hist_3d = calculate_3d_histogram(x_ch, bins, limits)
    gm_i, gm_q = calculate_1d_gmms(x_ch, bins)
    return gm_2d, hist_3d, gm_i, gm_q


def process_channels(histograms_hist, histograms_gmm, distance, power, spacing, osnr, song, orth, X_chs, bins, limits):
    for x_ch in X_chs:
        gm_2d, hist_3d, gm_i, gm_q = process_channel(x_ch, bins, limits)
        histograms_hist[distance][power][spacing][osnr][song][orth].append(
            hist_3d)
        histograms_gmm[distance][power][spacing][osnr][song][orth].append([
                                                                          gm_2d, gm_i, gm_q])


def calculate_gmm_and_histograms(data):
    histograms_hist = initialize_histograms()
    histograms_gmm = initialize_histograms()
    bins = 128
    limits = [-5, 5]

    for distance, powers in data.items():
        for power, spacings in powers.items():
            for spacing, osnrs in spacings.items():
                for osnr, songs in osnrs.items():
                    for song, orths in songs.items():
                        for orth, X_rx in orths.items():
                            logger.info(f"Calculating GMM for: {distance}/{power}/{spacing}/{osnr}/{song}/{orth}")
                            X_chs = gmm_utils.split(X_rx, 3)
                            process_channels(histograms_hist, histograms_gmm, distance,
                                             power, spacing, osnr, song, orth, X_chs, bins, limits)

    return dict(histograms_hist), dict(histograms_gmm)


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
models_tuple = None if models_hist is None or models_gmm is None else (
    models_hist, models_gmm)

# Calculate histograms if not loaded
if models_tuple is None:
    logger.info("Calculating histograms and GMMs...")
    models_tuple = gmm_utils.calc_once(
        "models_tuple", calculate_gmm_and_histograms, {"data": data})
    models_hist, models_gmm = models_tuple
    sofa.joblib_save(models_hist, file_models_hist)
    sofa.joblib_save(models_gmm, file_models_gmm)
    logger.info("Features calculated successfully")
else:
    logger.info("Features loaded successfully")
