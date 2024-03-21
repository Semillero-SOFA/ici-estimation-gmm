#!/usr/bin/env python
# coding: utf-8

# Libraries
from collections import defaultdict

import numpy as np
import polars as pl
from pathlib import Path
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

# Try to load histograms
file_models_hist = f"{RESULTS_DIR}/models32_hist.pkl"
file_models_gmm = f"{RESULTS_DIR}/models32_gmm.pkl"

models_hist = sofa.joblib_load(file_models_hist)
models_gmm = sofa.joblib_load(file_models_gmm)
models_tuple = (
    None if models_hist is None or models_gmm is None else (
        models_hist, models_gmm)
)

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

# Hyperparameters evaluation
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
# Training
results_file = f"{RESULTS_DIR}/gmm32_reg_results.h5"
try:
    histograms_reg_results = sofa.load_hdf5(results_file)
except:
    print("Error loading from file, creating a new dictionary")
    histograms_reg_results = defaultdict(
        defaultdict(defaultdict(defaultdict().copy).copy).copy
    )

for activations in gmm_utils.HIDDEN_LAYERS_LIST:
    for neurons in gmm_utils.MAX_NEURONS_LIST:
        for osnr in gmm_utils.OSNR_LIST:
            args = {
                "data": df_shuffled,
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
