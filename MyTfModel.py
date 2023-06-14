import numpy as np
import gym
from typing import Dict, Optional, Sequence

# import keras_nlp

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType, List, ModelConfigDict
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.tf_utils import one_hot

import keras_nlp

tf1, tf, tfv = try_import_tf()


class MyKerasTransformerModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasTransformerModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        x = self.inputs

        # todo

        head_size = 256
        num_heads = 4
        ff_dim = 4
        dropout = 0.25

        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        # todo

        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}



class MyKerasTransformerModel_V2(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasTransformerModel_V2, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        x = self.inputs

        # todo

        head_size = 512
        num_heads = 4
        ff_dim = 4
        dropout = 0.2
        mlp_dropout = 0.3

        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)


        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        # todo

        x = tf.keras.layers.Dense(
            512,
            activation="tanh",

        )(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)

        layer_1 = tf.keras.layers.Dense(
            512,
            name="my_layer1",
            activation="tanh",
        )(x)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            # activation=None,
            activation="softmax",

        )(layer_1)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            # activation=None,
            activation="softmax",
        )(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(res)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])



class MyKerasTransformerModel_V3(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasTransformerModel_V3, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        x = self.inputs

        positional_encoding = keras_nlp.layers.SinePositionEncoding()(x)
        x = x + positional_encoding

        # todo

        head_size = 512
        num_heads = 4
        ff_dim = 4
        dropout = 0.2
        mlp_dropout = 0.3

        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)


        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        # todo

        x = tf.keras.layers.Dense(
            512,
            activation="tanh",

        )(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)

        layer_1 = tf.keras.layers.Dense(
            512,
            name="my_layer1",
            activation="tanh",
        )(x)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            # activation=None,
            activation="softmax",

        )(layer_1)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            # activation=None,
            activation="softmax",
        )(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(res)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyKerasTransformerModel_V4(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasTransformerModel_V4, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        x = self.inputs

        # todo

        head_size = model_config["custom_model_config"]["head_size"]
        num_heads = 4
        ff_dim = obs_space.shape[-1]
        dropout = model_config["custom_model_config"]["dropout"]

        x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        if model_config["custom_model_config"]["flattening_type"] == "GlobalAveragePooling1D":
            x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        elif model_config["custom_model_config"]["flattening_type"] == "Flatten":
            x = tf.keras.layers.Flatten(data_format="channels_first")(x)
        else:
            raise ValueError("flattening_type is not set")
        # todo

        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        mha_out = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        mha_out = tf.keras.layers.Dropout(dropout)(mha_out)
        mha_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(mha_out + inputs)

        # Feed Forward Part
        ff_out = tf.keras.layers.Dense(ff_dim)(mha_out)
        ff_out = tf.keras.layers.Dropout(dropout)(ff_out)
        ff_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_out + mha_out)
        return ff_out

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}
        
class MyKerasTransformerModel_V5(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasTransformerModel_V5, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        x = self.inputs

        # todo

        head_size = 256
        num_heads = 4
        ff_dim = obs_space.shape[-1]

        x = self.transformer_encoder(x, head_size, num_heads, ff_dim)
        #x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        x = tf.keras.layers.Flatten(data_format="channels_first")(x)
        # todo

        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(x)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        mha_out = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        mha_out = mha_out + inputs

        # Feed Forward Part
        ff_out = tf.keras.layers.Dense(ff_dim)(mha_out)
        ff_out = ff_out + mha_out
        return ff_out

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}

class MyKerasModel_V1(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasModel_V1, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        x = self.inputs
        # todo

        x = tf.keras.layers.Dense(
            512,
            activation="tanh",

        )(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)

        layer_1 = tf.keras.layers.Dense(
            512,
            name="my_layer1",
            activation="tanh",
        )(x)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            # activation=None,
            activation="softmax",

        )(layer_1)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            # activation=None,
            activation="softmax",
        )(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])


    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        # Define the core model layers which will be used by the other
        # output heads of DistributionalQModel
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(self.inputs)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(layer_1)
        self.base_model = tf.keras.Model(self.inputs, layer_out)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}





ModelCatalog.register_custom_model("transformer_model", MyKerasTransformerModel)

ModelCatalog.register_custom_model("transformer_model_v2", MyKerasTransformerModel_V2)

ModelCatalog.register_custom_model("transformer_model_v3", MyKerasTransformerModel_V3)
ModelCatalog.register_custom_model("transformer_model_v4", MyKerasTransformerModel_V4)
ModelCatalog.register_custom_model("transformer_model_v5", MyKerasTransformerModel_V5)


ModelCatalog.register_custom_model("my_model_v1", MyKerasModel_V1)