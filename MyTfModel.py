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

class MyFCNetFrames(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super(MyFCNetFrames, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        self.num_frames = 3

        hiddens = list(model_config.get("fcnet_hiddens", [])) + \
            list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        activation = get_activation_fn(activation)
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std = model_config.get("free_log_std")

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable(
                [0.0] * num_outputs, dtype=tf.float32, name="log_std")

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(
            #shape=(self.num_frames, obs_space.shape[0], obs_space.shape[1], ), name="observations")
            #shape=(int(np.product(obs_space.shape)), ), name="observations")
            shape=(self.num_frames, int(np.product(obs_space.shape)), ), name="observations")
        obs_reshaped = tf.keras.layers.Reshape([int(np.product(obs_space.shape))*self.num_frames])(inputs)
        # Last hidden layer output (before logits outputs).
        last_layer = obs_reshaped
        # The action distribution outputs.
        logits_out = None
        i = 1

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                last_layer = tf.keras.layers.Dense(
                    hiddens[-1],
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
            if num_outputs:
                logits_out = tf.keras.layers.Dense(
                    num_outputs,
                    name="fc_out",
                    activation=None,
                    kernel_initializer=normc_initializer(0.01))(last_layer)
            # Adjust num_outputs to be the number of nodes in the last layer.
            else:
                self.num_outputs = (
                    [int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std and logits_out is not None:

            def tiled_log_std(x):
                return tf.tile(
                    tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(inputs)
            logits_out = tf.keras.layers.Concatenate(axis=1)(
                [logits_out, log_std_out])

        last_vf_layer = None
        if not vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            last_vf_layer = inputs
            i = 1
            for size in hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_vf_layer)
                i += 1

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(
                last_vf_layer if last_vf_layer is not None else last_layer)

        self.base_model = tf.keras.Model(
            inputs, [(logits_out
                      if logits_out is not None else last_layer), value_out])

        self._last_value = None
        self.view_requirements["prev_n_obs"] = ViewRequirement(
                data_col="obs",
                shift="-{}:0".format(self.num_frames-1),
                space = obs_space
            )
        #self.view_requirements["prev_n_rewards"] = ViewRequirement(
        #        data_col="rewards",
        #        shift="-{}:-1".format(self.num_frames)
        #    )
        #self.view_requirements["prev_n_actions"] = ViewRequirement(
        #        data_col="actions",
        #        shift="-{}:-1".format(self.num_frames),
        #        space=self.action_space
        #    )

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        print(input_dict.keys())
        print(input_dict["obs"].shape)
        print(input_dict["prev_n_obs"].shape)
        print(input_dict["obs_flat"].shape)
        print(self.view_requirements)
        print(self.__class__)
        obs = tf.cast(input_dict["prev_n_obs"], tf.float32)
        #model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        model_out, self._value_out = self.base_model(obs)
        return model_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model("transformer_model", MyFCNetFrames)
