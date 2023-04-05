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

tf1, tf, tfv = try_import_tf()


# Конволюция не юзается или она где-то до этого используется?


# class MyTransformerModel(TFModelV2):
#     def __init__(self,
#                  obs_space,
#                  action_space,
#                  num_outputs,
#                  model_config,
#                  name):
#         super(MyTransformerModel, self).__init__(
#             obs_space, action_space, num_outputs, model_config, name
#         )

#         # input_shape = obs_space.shape[1:]
#         input_shape = obs_space.shape
#         self.base_model = self.build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4,
#                                            mlp_units=[128], mlp_dropout=0.4, dropout=0.25, n_classes=2)

#     def build_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0,
#                     mlp_dropout=0,
#                     n_classes=2):

# #         inputs = tf.keras.Input(shape=input_shape)
#         inputs = tf.keras.layers.Input(shape=input_shape, name="observations")

#         x = inputs
#         # x = tf.keras.layers.Embedding(100, 64, input_length=32)(x)
# #         x = keras_nlp.layers.SinePositionEncoding()(x)
#         for _ in range(num_transformer_blocks):
#             x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

#         x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
#         for dim in mlp_units:
#             x = tf.keras.layers.Dense(dim, activation="relu")(x)
#             x = tf.keras.layers.Dropout(mlp_dropout)(x)
#         outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
#         return tf.keras.Model(inputs, outputs)


#     def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
#         # Attention and Normalization
#         x = tf.keras.layers.MultiHeadAttention(
#             key_dim=head_size, num_heads=num_heads, dropout=dropout
#         )(inputs, inputs)
#         x = tf.keras.layers.Dropout(dropout)(x)
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
#         res = x + inputs

#         # Feed Forward Part
#         x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
#         x = tf.keras.layers.Dropout(dropout)(x)
#         x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
#         return x + res

#     def forward(self, input_dict, state, seq_lens):
#         print(input_dict["obs"])
#         model_out, self._value_out = self.base_model(input_dict["obs"])
#         return model_out, state

#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])


# class MyTransformerModel_1(TFModelV2):
#     def __init__(self,
#                  obs_space,
#                  action_space,
#                  num_outputs,
#                  model_config,
#                  name):
#         super(MyTransformerModel_1, self).__init__(
#             obs_space, action_space, num_outputs, model_config, name
#         )

#         # input_shape = obs_space.shape[1:]
#         # input_shape = obs_space.shape[0]
#         # input_shape = tf.keras.layers.Input(
#         #     shape=obs_space.shape, name="observations")
#         input_shape = obs_space.shape
#         print(obs_space.shape)
#         print("AAAAAAAAAAAAAA", input_shape)
#         self.base_model = self.build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4,
#                                            mlp_units=[128], mlp_dropout=0.4, dropout=0.25, n_classes=2)

#     def build_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks,
#                     mlp_units,
#                     dropout=0,
#                     mlp_dropout=0,
#                     n_classes=2):

#         inputs = tf.keras.Input(shape=input_shape)
#         self.inputs = tf.keras.layers.Input(shape=input_shape, name="observations")
#         x = input_shape
#         print("11111",x)
#         x = tf.keras.layers.Dense(512, activation="tanh")(x)
#         x = tf.keras.layers.Dense(512, activation="tanh")(x)

#         # for dim in mlp_units:
#             # x = tf.keras.layers.Dense(dim, activation="relu")(x)
#             # x = tf.keras.layers.Dropout(mlp_dropout)(x)
#         outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
#         return tf.keras.Model(inputs, outputs)


#     def forward(self, input_dict, state, seq_lens):
#         print(input_dict["obs"])
#         model_out, self._value_out = self.base_model(input_dict["obs"])
#         return model_out, state

#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])

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


# ModelCatalog.register_custom_model("transformer_model", MyTransformerModel)


# ModelCatalog.register_custom_model("transformer_model", MyTransformerModel_1)

# ModelCatalog.register_custom_model("transformer_model", MyKerasModel)

ModelCatalog.register_custom_model("transformer_model", MyKerasTransformerModel)
