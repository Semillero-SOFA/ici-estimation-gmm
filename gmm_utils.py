from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow.keras as ker
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

OSNR_LIST = ["osnr", "wo_osnr"]
MAX_NEURONS_LIST = [str(2**n) for n in range(3, 11)]
FUNCTIONS_LIST = ["relu", "tanh", "sigmoid"]
LAYERS_NUMBER_LIST = [1, 2, 3]
COMBINATIONS_LIST = [
    [list(subset) for subset in product(FUNCTIONS_LIST, repeat=n)] for n in LAYERS_NUMBER_LIST
]
HIDDEN_LAYERS_LIST = [
    item for sublist in COMBINATIONS_LIST for item in sublist]
N_CLASSES_LIST = [str(n) for n in range(2, 6)]


def split(a, n):
    k, m = divmod(len(a), n)
    return np.array(
        [a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]
    )


def calc_once(varname, fn, args):
    """Calculate a variable only once."""
    if varname not in globals() or eval(varname) is None:
        return fn(**args)
    return eval(varname)


def estimation_model(
    layers_props_lst: list, loss_fn: ker.losses.Loss, input_dim: int
) -> ker.models.Sequential:
    """Compile a sequential model for regression purposes."""
    model = ker.Sequential()
    # Hidden layers
    for i, layer_props in enumerate(layers_props_lst):
        if i == 0:
            model.add(ker.layers.Dense(input_dim=input_dim, **layer_props))
        else:
            model.add(ker.layers.Dense(**layer_props))
    # Regressor
    model.add(ker.layers.Dense(units=1, activation="linear"))

    model.compile(loss=loss_fn, optimizer="adam")

    return model


def estimation_crossvalidation(
    X, y, n_splits, layer_props, loss_fn, callbacks
):
    """Crossvalidation of an estimation network."""
    # Scores dict
    scores = {}
    scores["model"] = []
    scores["loss"] = []
    scores["mae"] = {"train": [], "test": []}
    scores["r2"] = {"train": [], "test": []}
    scores["rmse"] = {"train": [], "test": []}

    # K-fold crossvalidation
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Input variables standarizer
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test_kf = sc.transform(X_test)

        model = estimation_model(layer_props, loss_fn, X_train.shape[1])

        # Save test scalar loss
        if callbacks:
            loss = model.fit(
                X_train,
                y_train,
                epochs=5000,
                batch_size=64,
                callbacks=callbacks,
                verbose=0,
            )
        else:
            loss = model.fit(X_train, y_train, epochs=5000,
                             batch_size=64, verbose=0)
        print(f"Needed iterations: {len(loss.history['loss'])}")
        loss = loss.history["loss"]

        # Predict using train values
        predictions_train = model.predict(X_train, verbose=0).flatten()
        predictions_test = model.predict(X_test_kf, verbose=0).flatten()

        # Dataframe for better visualization
        train_data_train = pl.DataFrame(
            {"ICI": y_train, "Predicted ICI": predictions_train}
        )
        train_data_test = pl.DataFrame(
            {"ICI": y_test, "Predicted ICI": predictions_test}
        )

        # MAE
        mae_score_train = mean_absolute_error(
            train_data_train["ICI"], train_data_train["Predicted ICI"]
        )
        mae_score_test = mean_absolute_error(
            train_data_test["ICI"], train_data_test["Predicted ICI"]
        )

        # R²
        r2_score_train = r2_score(
            train_data_train["ICI"], train_data_train["Predicted ICI"]
        )
        r2_score_test = r2_score(
            train_data_test["ICI"], train_data_test["Predicted ICI"]
        )

        # RMSE
        rmse_score_train = mean_squared_error(
            train_data_train["ICI"], train_data_train["Predicted ICI"],
            squared=False
        )
        rmse_score_test = mean_squared_error(
            train_data_test["ICI"], train_data_test["Predicted ICI"],
            squared=False
        )

        # Append to lists
        scores["model"].append(model)
        scores["loss"].append(loss)
        scores["mae"]["train"].append(mae_score_train)
        scores["mae"]["test"].append(mae_score_test)
        scores["r2"]["train"].append(r2_score_train)
        scores["r2"]["test"].append(r2_score_test)
        scores["rmse"]["train"].append(rmse_score_train)
        scores["rmse"]["test"].append(rmse_score_test)

    return scores


def test_estimation_model(
    data,
    n_splits,
    max_neurons,
    activations,
    use_osnr=True,
    loss_fn="mean_absolute_error",
):
    """Test a spectral spacing estimation model with given parameters."""
    n_feat = data.shape[1]
    var_n = n_feat - 1 if use_osnr else n_feat - 2

    # Split variables
    # Variables
    X = np.array(data[:, 0:var_n])
    # Tags
    y = np.array(data[:, -1])

    # Layer properties
    layer_props = [
        {"units": max_neurons // (2**i), "activation": activation}
        for i, activation in enumerate(activations)
    ]
    print(f"{layer_props}{' + OSNR' if use_osnr else ''}")
    callbacks = [
        EarlyStopping(
            monitor="loss", patience=30, mode="min", restore_best_weights=True
        )
    ]

    return estimation_crossvalidation(
        X, y, n_splits, layer_props, loss_fn, callbacks
    )


def classificator(df, interval_lst, column_name):
    """Transforms a dataframe's column into classes"""
    array = df[column_name].to_numpy()
    new_column = pl.Series(np.digitize(array, interval_lst))

    df_classfull = df.clone()
    df_classfull = df_classfull.with_columns(new_column.alias(column_name))

    return df_classfull


def classifier_model(
    layers_props_lst: list, classes_n: int,
    loss_fn: ker.losses.Loss, input_dim: int
) -> ker.models.Sequential:
    """Compile a sequential model for classification purposes."""
    model = ker.Sequential()
    # Hidden layers
    for i, layer_props in enumerate(layers_props_lst):
        if i == 0:
            model.add(ker.layers.Dense(input_dim=input_dim, **layer_props))
        else:
            model.add(ker.layers.Dense(**layer_props))
    # Classifier
    model.add(ker.layers.Dense(units=classes_n, activation="softmax"))

    model.compile(loss=loss_fn, optimizer="adam")

    return model


def classification_crossvalidation(
    X, y, n_splits, layer_props, classes_n, loss_fn, callbacks
):
    """Crossvalidation of a classification network."""
    # Scores dict
    scores = {}
    scores["model"] = []
    scores["loss"] = []
    scores["acc"] = {"train": [], "test": []}
    scores["f1"] = {"train": [], "test": []}
    scores["cm"] = {"train": [], "test": []}

    # K-fold crossvalidation
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Input variables standarizer
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test_kf = sc.transform(X_test)

        model = classifier_model(
            layer_props, classes_n,
            loss_fn, X_train.shape[1]
        )

        # Save test scalar loss
        if callbacks:
            loss = model.fit(
                X_train,
                y_train,
                epochs=5000,
                batch_size=64,
                callbacks=callbacks,
                verbose=0,
            )
        else:
            loss = model.fit(X_train, y_train, epochs=5000,
                             batch_size=64, verbose=0)
        print(f"Needed iterations: {len(loss.history['loss'])}")
        loss = loss.history["loss"]

        # Predict using train, test and prod values
        fuzzy_predictions_train = model.predict(X_train)
        fuzzy_predictions_test = model.predict(X_test_kf)

        # Assign class based on higher probability in membership vector
        predictions_train = np.array(
            [
                np.argmax(fuzzy_prediction)
                for fuzzy_prediction in fuzzy_predictions_train
            ]
        ).flatten()
        predictions_test = np.array(
            [
                np.argmax(fuzzy_prediction)
                for fuzzy_prediction in fuzzy_predictions_test
            ]
        ).flatten()

        # Dataframe for better visualization
        train_data_train = pl.DataFrame(
            {"ICI": y_train, "Predicted ICI": predictions_train}
        )
        train_data_test = pl.DataFrame(
            {"ICI": y_test, "Predicted ICI": predictions_test}
        )

        # Accuracy
        acc_score_train = accuracy_score(
            train_data_train["ICI"], train_data_train["Predicted ICI"]
        )
        acc_score_test = accuracy_score(
            train_data_test["ICI"], train_data_test["Predicted ICI"]
        )

        # F1
        f1_score_train = f1_score(
            train_data_train["ICI"],
            train_data_train["Predicted ICI"],
            average="micro",
        )
        f1_score_test = f1_score(
            train_data_test["ICI"],
            train_data_test["Predicted ICI"],
            average="micro"
        )

        # Confusion matrix
        cm_score_train = multilabel_confusion_matrix(
            train_data_train["ICI"], train_data_train["Predicted ICI"]
        ).tolist()
        cm_score_test = multilabel_confusion_matrix(
            train_data_test["ICI"], train_data_test["Predicted ICI"]
        ).tolist()

        # Append to lists
        scores["model"].append(model)
        scores["loss"].append(loss)
        scores["acc"]["train"].append(acc_score_train)
        scores["acc"]["test"].append(acc_score_test)
        scores["f1"]["train"].append(f1_score_train)
        scores["f1"]["test"].append(f1_score_test)
        scores["cm"]["train"].append(cm_score_train)
        scores["cm"]["test"].append(cm_score_test)

    return scores


def test_classification_model(
    data,
    n_splits,
    max_neurons,
    activations,
    classes_n,
    use_osnr=True,
    loss_fn="sparse_categorical_crossentropy",
):
    """Test a spectral overlapping classification model with given parameters."""
    n_feat = data.shape[1]
    var_n = n_feat - 1 if use_osnr else n_feat - 2

    # Split variables
    # Variables
    X = np.array(data[:, 0:var_n])
    # Tags
    y = np.array(data[:, -1])

    # Layer properties
    layer_props = [
        {"units": max_neurons // (2**i), "activation": activation}
        for i, activation in enumerate(activations)
    ]
    print(f"{layer_props} {classes_n} classes {' + OSNR' if use_osnr else ''}")
    callbacks = [
        EarlyStopping(
            monitor="loss", patience=30, mode="min", restore_best_weights=True
        )
    ]

    return classification_crossvalidation(
        X, y, n_splits, layer_props, classes_n, loss_fn, callbacks
    )


def plot_constellation_diagram(X, ax):
    ax.scatter(X.real, X.imag, alpha=0.5)
    ax.set_title("Constellation diagram")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")


def calculate_gmm(data, gm_kwargs):
    return GaussianMixture(**gm_kwargs).fit(data)


def calculate_1d_histogram(X, bins):
    hist_y, hist_x = np.histogram(X.real, bins=bins)
    # Remove last bin edge
    hist_x = hist_x[:-1]

    return hist_x, hist_y


def plot_1d_histogram(X, bins=128, ax=None):
    ax.hist(X, bins=bins, density=True, alpha=0.5,
            label="Calculated histogram")


def plot_gmm_1d(gm, limits, ax):
    x = np.linspace(*limits, 1000)

    logprob = gm.score_samples(x.reshape(-1, 1))
    responsibilities = gm.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax.plot(x, pdf_individual, "--", label="Adjusted histogram")


def plot_gmm_2d(gm, limits, ax):
    x = y = np.linspace(*limits)
    X, Y = np.meshgrid(x, y)
    Z = -gm.score_samples(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    ax.contour(
        X,
        Y,
        Z,
        norm=LogNorm(vmin=1.0, vmax=1000.0),
        levels=np.logspace(0, 3, 25),
        cmap="seismic",
    )


def calculate_3d_histogram(X, bins, limits):
    hist, xedges, yedges = np.histogram2d(
        X.real, X.imag, bins=bins, range=[[*limits], [*limits]]
    )
    # Create the meshgrid for the surface plot, excluding the last edge
    x_mesh, y_mesh = np.meshgrid(xedges[:-1], yedges[:-1])
    return hist, x_mesh, y_mesh


def plot_3d_histogram(x_mesh, y_mesh, hist, ax):
    ax.plot_surface(
        x_mesh, y_mesh, hist.T, cmap="seismic",
        rstride=1, cstride=1, edgecolor="none"
    )
    ax.set_title("3D Histogram")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")


def plot_results(
        x_values, scores, path, xlabel, ylabel,
        log=False, intx=False):
    plt.figure(figsize=(8, 6), layout="constrained")
    plt.scatter(x_values, scores)
    plt.plot(x_values, scores)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log:
        plt.xscale("log", base=2)
    if intx:
        plt.xticks(x_values)
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def plot_2d_results(
        x_values, y_values, z_values, path, xlabel, ylabel,
        log_flag=(False, False), int_flag=(False, False)):
    plt.figure(figsize=(12, 8))
    cs = plt.contourf(x_values, y_values, z_values, cmap="inferno", alpha=0.9,
                      linestyles="dashed")
    plt.clabel(cs, colors="#000000", inline=False)
    if any(log_flag):
        plt.xscale("log", base=2) if log_flag[0] else plt.yscale("log", base=2)
    if any(int_flag):
        plt.xticks(x_values[0]) if int_flag[0] else plt.yticks(y_values[0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.savefig(path)
    plt.close()


def plot_cm(scores, interval_lst, path):
    CM = np.array(scores.get("cm").get("test"))
    for n, interval in enumerate(interval_lst):
        result = np.zeros(CM[0][0].shape)
        for cm in CM:
            result = np.add(result, cm[n])
        result /= np.sum(result)
        disp = ConfusionMatrixDisplay(confusion_matrix=result, display_labels=[
                                      "Positive", "Negative"])
        disp.plot(colorbar=False)
        lower_limit, upper_limit = interval
        plt.title(f"Confusion matrix for class from {lower_limit} GHz up to {upper_limit} GHz")
        plt.savefig(path)
        plt.close()


def get_avg_reg_score(results, target_value, target="neurons",
                      metric="mae", score="test"):
    score_list = []
    for activations in HIDDEN_LAYERS_LIST:
        if target == "layers" and len(activations) != target_value:
            continue
        for neurons in MAX_NEURONS_LIST:
            if target == "neurons" and neurons != target_value:
                continue
            for osnr in OSNR_LIST:
                if target == "osnr" and osnr != target_value:
                    continue
                act_fn_name = "".join([s[0] for s in activations])
                score_list.append(
                    np.mean(
                        [*results[act_fn_name][neurons]
                            [osnr][metric][score].values()]
                    )
                )
    return score_list


def get_neurons_vs_layers(results, layers_target, neurons_target, metric="mae", score="test"):
    score_list = []
    for activations in HIDDEN_LAYERS_LIST:
        if len(activations) != layers_target:
            continue
        for neurons in MAX_NEURONS_LIST:
            if neurons != neurons_target:
                continue
            for osnr in OSNR_LIST:
                act_fn_name = "".join([s[0] for s in activations])
                score_list.append(
                    np.mean(
                        [*results[act_fn_name][neurons]
                            [osnr][metric][score].values()]
                    )
                )
    return score_list


def get_avg_class_score(results, target_value, target="neurons",
                        metric="acc", score="test"):
    score_list = []
    for activations in HIDDEN_LAYERS_LIST:
        if target == "layers" and len(activations) != target_value:
            continue
        for neurons in MAX_NEURONS_LIST:
            if target == "neurons" and neurons != target_value:
                continue
            for osnr in OSNR_LIST:
                if target == "osnr" and osnr != target_value:
                    continue
                for n_classes in N_CLASSES_LIST:
                    if target == "n_classes" and n_classes != target_value:
                        continue
                    act_fn_name = "".join([s[0] for s in activations])
                    score_list.append(
                        np.mean(
                            [*results[act_fn_name][neurons]
                                [osnr][n_classes][metric][score].values()]
                        )
                    )
    return score_list


def get_better_reg_models(results, metric="mae", score="test"):
    scores = []
    for activations in HIDDEN_LAYERS_LIST:
        for neurons in MAX_NEURONS_LIST:
            for osnr in OSNR_LIST:
                act_fn_name = "".join([s[0] for s in activations])
                coll = results[act_fn_name][neurons][osnr][metric][score].values()
                if isinstance(coll, defaultdict):
                    continue
                score_value = np.mean([*coll])
                scores.append((score_value, [act_fn_name, neurons, osnr]))
    scores.sort(key=lambda x: x[0])
    return pl.dataframe.DataFrame(scores)


def get_better_class_models(results, metric="acc", score="test"):
    scores = []
    for activations in HIDDEN_LAYERS_LIST:
        for neurons in MAX_NEURONS_LIST:
            for osnr in OSNR_LIST:
                for n_classes in N_CLASSES_LIST:
                    act_fn_name = "".join([s[0] for s in activations])
                    coll = results[act_fn_name][neurons][osnr][n_classes][metric][score].values(
                    )
                    if isinstance(coll, defaultdict):
                        continue
                    score_value = np.mean([*coll])
                    scores.append(
                        (score_value, [act_fn_name, neurons, osnr, n_classes]))
    scores.sort(key=lambda x: x[0])
    return pl.dataframe.DataFrame(scores)
