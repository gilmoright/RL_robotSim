#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import yaml
from datetime import datetime
import random
import pyhocon
import ray
from ray.tune.config_parser import make_parser
from ray.tune.progress_reporter import CLIReporter, JupyterNotebookReporter
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.resources import resources_to_json
from ray.tune.tune import run_experiments
from ray.tune.schedulers import create_scheduler
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import MyMisc
# import MyModel
import MyTfModel
import MyNewModels


tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.")

    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    parser.add_argument(
        "--experiment",
        default="",
        type=str,
        help="experiment name from config_file")
    return parser


def run(args, parser):
    print("START", datetime.now())
    configFileContent = pyhocon.ConfigFactory.parse_file(args.config_file)
    expName = args.experiment
    experiments = {}
    CONFIG = configFileContent[expName].as_plain_ordered_dict()
    print(expName, "config")
    print(CONFIG)
    hyperparam_mutations = {}
    for param, value in CONFIG["config"].items():
        if type(value)==list:
            if value[0] == "tune":
                if value[1]=="randint":
                    print("randint value[2]", value[2], *value[2])
                    int_param_min = value[2][0]
                    int_param_max = value[2][1]
                    #hyperparam_mutations[param] = lambda: random.randint(int_param_min, int_param_max)
                    hyperparam_mutations[param] = tune.randint(int_param_min, int_param_max)
                    #CONFIG[param] = tune.randint(value[2])
                    if len(value)==4:
                        CONFIG["config"][param] = value[3]
                    else:
                        CONFIG["config"][param] = tune.randint(*value[2])
                elif value[1]=="uniform":
                    value[2] = [float(x) for x in value[2]]
                    float_param_min = value[2][0]
                    float_param_max = value[2][1]
                    print("uniform value[2]", value[2], *value[2])
                    #hyperparam_mutations[param] = lambda: random.uniform(float_param_min, float_param_max)
                    hyperparam_mutations[param] = tune.uniform(float_param_min, float_param_max)
                    #CONFIG[param] = tune.uniform(value[2])
                    if len(value)==4:
                        CONFIG["config"][param] = value[3]
                    else:
                        CONFIG["config"][param] = tune.uniform(*value[2])
                elif value[1]=="choice":
                    print("choice value[2]", value[2], *value[2])
                    list_of_variants = value[2]
                    #hyperparam_mutations[param] = lambda: random.choice(value[2])
                    hyperparam_mutations[param] = list_of_variants
                    #CONFIG[param] = tune.choice(value[2])
                    if len(value)==4:
                        CONFIG["config"][param] = value[3]
                    else:
                        CONFIG["config"][param] = tune.choice(value[2])
                else:
                    raise ValueError("Wrong sample parameter for {}:{}".format(param, str(value)))
        if value=="ray.rllib.agents.callbacks.DefaultCallbacks":
            CONFIG["config"][param] = ray.rllib.agents.callbacks.DefaultCallbacks
        if value=="ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector":
            CONFIG["config"][param] = ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector
        if param == "exploration_config" and type(value["type"]) == list and value["type"][0]=="tune":
            hyperparam_mutations["exploration_config"] = {}
            #hyperparam_mutations["exploration_config"]["type"] = lambda: random.choice([getattr(ray.rllib.utils.exploration, x) for x in value["type"][2]])
            hyperparam_mutations["exploration_config"]["type"] = value["type"][2]
            if len(value)==4:
                CONFIG["config"][param] = getattr(ray.rllib.utils.exploration, value["type"][3])
            else:
                CONFIG["config"]["exploration_config"]["type"] = tune.choice([getattr(ray.rllib.utils.exploration, x) for x in value["type"][2]])
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=50,#100,
        resample_probability=0.33,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
    )

    # Stop when we've either reached 100 training iterations or reward=300
    #stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 300}

    analysis = tune.run(
        run_or_experiment = CONFIG["run"],
        name=expName,
        scheduler = pbt,
        metric="episode_reward_mean",
        mode="max",
        num_samples=12, #8,
        #config=CONFIG,
        config=CONFIG["config"],        
        local_dir = CONFIG["local_dir"],
        stop = CONFIG["stop"],
        log_to_file = True,
        #run_config=air.RunConfig(stop=stopping_criteria),
    )

    print("best hyperparameters: ", analysis.best_config)
    print("END", datetime.now())


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)


if __name__ == "__main__":
    main()
