import itertools

import tensorflow as tf

from lib.data_utils import load_dict_from_file
from lib.tf_utils import ResidulaFCBlock

kwargs_data = [
    "input_mode",
    "label_mode",
    "n_window_targets",
    "n_window_history",
    "downsampling_rate",
    "n_window_forecasts",
    "use_actions",
    "feature_scaling",
    "p_scaling_std",
]

kwargs_model = [
    "input_dim",
    "n_hidden_layers",
    "n_hidden",
    "model_type",
    "l1_reg",
    "l2_reg",
    "dropout_rate",
    "batch_normalization",
]


def load_model_kwargs(file_path):
    dictionary = load_dict_from_file(file_path)

    model_kwargs = {key: dictionary[key] for key in kwargs_model}
    return model_kwargs


def load_data_kwargs(file_path):
    dictionary = load_dict_from_file(file_path)

    model_kwargs = {key: dictionary[key] for key in kwargs_data}
    return model_kwargs


def regularization_kwargs(l1_reg, l2_reg):
    if l1_reg > 0:
        kwargs_reg = {
            "kernel_regularizer": tf.keras.regularizers.L1(l1_reg),
            "bias_regularizer": tf.keras.regularizers.L1(l1_reg),
        }
    elif l2_reg > 0:
        kwargs_reg = {
            "kernel_regularizer": tf.keras.regularizers.L2(l2=l2_reg),
            "bias_regularizer": tf.keras.regularizers.L2(l2=l2_reg),
        }
    else:
        kwargs_reg = {}

    return kwargs_reg


def load_dnn(
    input_dim,
    n_hidden_layers,
    n_hidden,
    model_type,
    l1_reg,
    l2_reg,
    dropout_rate,
    batch_normalization,
    initial_bias=0.0,
):
    kwargs_reg = regularization_kwargs(l1_reg, l2_reg)

    if model_type == "fc":
        hidden_layers = [
            (
                tf.keras.layers.Dense(n_hidden, activation="relu", **kwargs_reg),
                tf.keras.layers.Dropout(dropout_rate),
            )
            for _ in range(n_hidden_layers)
        ]
        hidden_layers = list(itertools.chain(*hidden_layers))

        if batch_normalization:
            hidden_layers = hidden_layers + [tf.keras.layers.BatchNormalization()]

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    n_hidden, activation="relu", input_shape=(input_dim,), **kwargs_reg
                ),
                tf.keras.layers.Dropout(dropout_rate),
                *hidden_layers,
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid",
                    bias_initializer=tf.keras.initializers.Constant(initial_bias),
                    **kwargs_reg,
                ),
            ]
        )

    elif model_type == "linear":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid",
                    bias_initializer=tf.keras.initializers.Constant(initial_bias),
                    input_shape=(input_dim,),
                    **kwargs_reg,
                ),
            ]
        )
    else:
        hidden_layers = [
            (
                ResidulaFCBlock(n_hidden, activation="relu", **kwargs_reg),
                tf.keras.layers.Dropout(dropout_rate),
            )
            for _ in range(n_hidden_layers // 2)
        ]

        hidden_layers = list(itertools.chain(*hidden_layers))

        if batch_normalization:
            hidden_layers = hidden_layers + [tf.keras.layers.BatchNormalization()]

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    n_hidden, activation="relu", input_shape=(input_dim,), **kwargs_reg
                ),
                tf.keras.layers.Dropout(dropout_rate),
                *hidden_layers,
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid",
                    bias_initializer=tf.keras.initializers.Constant(initial_bias),
                    **kwargs_reg,
                ),
            ]
        )

    return model
