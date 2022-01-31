import numpy as np
import gym
from typing import Dict, Optional, Sequence

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

# TODO: (sven) obsolete this class once we only support native keras models.
class MyFCNet(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super(MyFCNet, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
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
            shape=(int(np.product(obs_space.shape)), ), name="observations")
        # Last hidden layer output (before logits outputs).
        last_layer = inputs
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

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])

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

ModelCatalog.register_custom_model("MyFCNet", MyFrameStackingModel)
