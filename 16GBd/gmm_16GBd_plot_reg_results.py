#!/usr/bin/env python
# coding: utf-8

# Libraries
import numpy as np
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

# Training
results_file = f"{RESULTS_DIR}/gmm16_reg_results.h5"
histograms_reg_results = sofa.load_hdf5(results_file)

# Results
# Neurons
gmm_neurons_avg_results = [
    np.mean(
        gmm_utils.get_avg_reg_score(
            histograms_reg_results,
            neurons,
            target="neurons",
            metric="mae",
            score="test",
        )
    )
    for neurons in gmm_utils.MAX_NEURONS_LIST
]
neurons_json_filename = Path(f"{RESULTS_DIR}/gmm_neurons_avg_results.json")
neurons_svg_filename = Path(f"{RESULTS_DIR}/gmm_neurons_avg_results.svg")
sofa.save_json(gmm_neurons_avg_results,
               neurons_json_filename, n_backups=0)
x = list(map(int, gmm_utils.MAX_NEURONS_LIST))
gmm_utils.plot_results(x,
                       gmm_neurons_avg_results,
                       neurons_svg_filename,
                       "Maximum number of neurons",
                       "MAE",
                       log=True)

# Layers
gmm_layers_avg_results = [
    np.mean(
        gmm_utils.get_avg_reg_score(
            histograms_reg_results,
            layers,
            target="layers",
            metric="mae",
            score="test"
        )
    )
    for layers in gmm_utils.LAYERS_NUMBER_LIST
]
layers_json_filename = Path(f"{RESULTS_DIR}/gmm_layers_avg_results.json")
layers_svg_filename = Path(f"{RESULTS_DIR}/gmm_layers_avg_results.svg")
sofa.save_json(gmm_layers_avg_results,
               layers_json_filename, n_backups=0)
x = list(map(int, gmm_utils.LAYERS_NUMBER_LIST))
gmm_utils.plot_results(x,
                       gmm_layers_avg_results,
                       layers_svg_filename,
                       "Number of layers",
                       "MAE",
                       log=False, intx=True)

# OSNR
gmm_osnr_avg_results = [
    np.mean(
        gmm_utils.get_avg_reg_score(
            histograms_reg_results,
            osnr,
            target="osnr",
            metric="mae",
            score="test"
        )
    )
    for osnr in gmm_utils.OSNR_LIST
]
osnr_json_filename = Path(f"{RESULTS_DIR}/gmm_osnr_avg_results.json")
sofa.save_json(gmm_osnr_avg_results,
               osnr_json_filename, n_backups=0)

# Sort models by score
better_models_df = gmm_utils.get_better_reg_models(
    histograms_reg_results, metric="mae", score="test"
)
better_models_df.write_json(
    f"{RESULTS_DIR}/gmm_16GBd_models_summary.json")
better_models_df.head(25).write_json(
    f"{RESULTS_DIR}/gmm_16GBd_better_models.json")
better_models_df.tail(25).write_json(
    f"{RESULTS_DIR}/gmm_16GBd_worst_models.json")

print("Results saved succesfully")
