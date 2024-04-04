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
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/gmm_16GBd_classification"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Training
results_file = f"{RESULTS_DIR}/gmm16_class_results.h5"
histograms_class_results = sofa.load_hdf5(results_file)

# Results
# Neurons
gmm_neurons_avg_results = [
    np.mean(
        gmm_utils.get_avg_class_score(
            histograms_class_results,
            neurons,
            target="neurons",
            metric="acc",
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
                       "Accuracy",
                       log=True)

# Layers
gmm_layers_avg_results = [
    np.mean(
        gmm_utils.get_avg_class_score(
            histograms_class_results,
            layers,
            target="layers",
            metric="acc",
            score="test"
        )
    )
    for layers in gmm_utils.LAYERS_NUMBER_LIST
]
layers_json_filename = Path(f"{RESULTS_DIR}/gmm_layers_avg_results.json")
layers_svg_filename = Path(f"{RESULTS_DIR}/gmm_layers_avg_results.svg")
sofa.save_json(gmm_layers_avg_results,
               layers_json_filename)
gmm_utils.plot_results(gmm_utils.LAYERS_NUMBER_LIST,
                       gmm_layers_avg_results,
                       layers_svg_filename,
                       "Number of layers",
                       "Accuracy",
                       log=False, intx=True)

# Classes
gmm_classes_avg_results = [
    np.mean(
        gmm_utils.get_avg_class_score(
            histograms_class_results,
            n_classes,
            target="n_classes",
            metric="acc",
            score="test"
        )
    )
    for n_classes in gmm_utils.N_CLASSES_LIST
]
classes_json_filename = Path(f"{RESULTS_DIR}/gmm_classes_avg_results.json")
classes_svg_filename = Path(f"{RESULTS_DIR}/gmm_classes_avg_results.svg")
sofa.save_json(gmm_classes_avg_results,
               classes_json_filename)
gmm_utils.plot_results(gmm_utils.N_CLASSES_LIST,
                       gmm_classes_avg_results,
                       classes_svg_filename,
                       "Number of classes",
                       "Accuracy",
                       log=False, intx=True)

# OSNR
gmm_osnr_avg_results = [
    np.mean(
        gmm_utils.get_avg_class_score(
            histograms_class_results,
            osnr,
            target="osnr",
            metric="acc",
            score="test"
        )
    )
    for osnr in gmm_utils.OSNR_LIST
]
osnr_json_filename = Path(f"{RESULTS_DIR}/gmm_osnr_avg_results.json")
sofa.save_json(gmm_osnr_avg_results,
               osnr_json_filename)

# Sort models by score
better_models_df = gmm_utils.get_better_class_models(
    histograms_class_results, metric="acc", score="test"
)
# Invert list
better_models_df = reversed(better_models_df)

better_models_df.write_json(
    f"{RESULTS_DIR}/gmm_16GBd_models_summary.json")
better_models_df.head(25).write_json(
    f"{RESULTS_DIR}/gmm_16GBd_better_models.json")
better_models_df.tail(25).write_json(
    f"{RESULTS_DIR}/gmm_16GBd_worst_models.json")

print("Results saved succesfully")
