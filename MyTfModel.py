import numpy as np
import gym
from typing import Dict, Optional, Sequence

import keras_nlp

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
#
#         input_shape = obs_space.shape[1:]
#         self.base_model = self.build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4,
#                                            mlp_units=[128], mlp_dropout=0.4, dropout=0.25, n_classes=2)
#
#     def build_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0,
#                     mlp_dropout=0,
#                     n_classes=2):
#
#         inputs = tf.keras.Input(shape=input_shape)
#         x = inputs
#         # x = tf.keras.layers.Embedding(100, 64, input_length=32)(x)
#         x = keras_nlp.layers.SinePositionEncoding()(x)
#         for _ in range(num_transformer_blocks):
#             x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
#
#         x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
#         for dim in mlp_units:
#             x = tf.keras.layers.Dense(dim, activation="relu")(x)
#             x = tf.keras.layers.Dropout(mlp_dropout)(x)
#         outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
#         return tf.keras.Model(inputs, outputs)
#
#     # def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
#     #     # Normalization and Attention
#     #     x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
#     #     x = tf.keras.layers.MultiHeadAttention(
#     #         key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
#     #     x = tf.keras.layers.Dropout(dropout)(x)
#     #     res = x + inputs
#     #
#     #     # Feed Forward Part
#     #     x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
#     #     x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(x)
#     #     x = tf.keras.layers.Dropout(dropout)(x)
#     #     x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     #     return x + res
#
#     def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
#         # Attention and Normalization
#         x = tf.keras.layers.MultiHeadAttention(
#             key_dim=head_size, num_heads=num_heads, dropout=dropout
#         )(inputs, inputs)
#         x = tf.keras.layers.Dropout(dropout)(x)
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
#         res = x + inputs
#
#         # Feed Forward Part
#         x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
#         x = tf.keras.layers.Dropout(dropout)(x)
#         x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
#         return x + res
#
#     def forward(self, input_dict, state, seq_lens):
#         model_out, self._value_out = self.base_model(input_dict["obs"])
#         return model_out, state
#
#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])


class MyTransformerModel(TFModelV2):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        super(MyTransformerModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        input_shape = obs_space.shape[1:]
        self.base_model = self.build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4,
                                           mlp_units=[128], mlp_dropout=0.4, dropout=0.25, n_classes=2)

    def build_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0,
                    mlp_dropout=0,
                    n_classes=2):

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(mlp_dropout)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        return tf.keras.Model(inputs, outputs)



    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# ModelCatalog.register_custom_model("transformer_model", MyTransformerModel)

class MyFrameStackingModel(TFModelV2):
    """A simple FC model that takes the last n observations as input."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 num_frames=3):
        super(MyFrameStackingModel, self).__init__(
            obs_space, action_space, None, model_config, name)

        self.num_frames = num_frames
        self.num_outputs = num_outputs

        # Construct actual (very simple) FC model.
        assert len(obs_space.shape) == 1
        obs = tf.keras.layers.Input(
            shape=(self.num_frames, obs_space.shape[0]))
        obs_reshaped = tf.keras.layers.Reshape(
            [obs_space.shape[0] * self.num_frames])(obs)
        rewards = tf.keras.layers.Input(shape=(self.num_frames))
        rewards_reshaped = tf.keras.layers.Reshape([self.num_frames])(rewards)
        actions = tf.keras.layers.Input(
            shape=(self.num_frames, self.action_space.n))
        actions_reshaped = tf.keras.layers.Reshape(
            [action_space.n * self.num_frames])(actions)
        input_ = tf.keras.layers.Concatenate(axis=-1)(
            [obs_reshaped, actions_reshaped, rewards_reshaped])
        layer1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(input_)
        layer2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(layer1)
        out = tf.keras.layers.Dense(self.num_outputs)(layer2)
        values = tf.keras.layers.Dense(1)(layer1)
        self.base_model = tf.keras.models.Model([obs, actions, rewards],
                                                [out, values])
        self._last_value = None

        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs",
            shift="-{}:0".format(num_frames - 1),
            space=obs_space)
        self.view_requirements["prev_n_rewards"] = ViewRequirement(
            data_col="rewards", shift="-{}:-1".format(self.num_frames))
        self.view_requirements["prev_n_actions"] = ViewRequirement(
            data_col="actions",
            shift="-{}:-1".format(self.num_frames),
            space=self.action_space)

    def forward(self, input_dict, states, seq_lens):
        obs = tf.cast(input_dict["prev_n_obs"], tf.float32)
        rewards = tf.cast(input_dict["prev_n_rewards"], tf.float32)
        actions = one_hot(input_dict["prev_n_actions"], self.action_space)
        print(obs, obs.shape)
        out, self._last_value = self.base_model([obs, actions, rewards])
        return out, []

    def value_function(self):
        return tf.squeeze(self._last_value, -1)

ModelCatalog.register_custom_model("transformer_model", MyFrameStackingModel)
