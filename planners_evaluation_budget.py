"""Usage: planners_evaluation.py [options]

Compare performances of several planners

Options:
  -h --help
  --generate <true or false>  Generate new data [default: True].
  --show <true_or_false>      Plot results [default: True].
  --directory <path>          Specify directory path [default: ./out/planners].
  --data_file <path>          Specify output data file name [default: data.csv].
  --budgets <start,end,N>     Computational budgets available to planners, in logspace [default: 1,3,100].
  --seeds <(s,)n>             Number of evaluations of each configuration, with an optional first seed [default: 10].
  --processes <p>             Number of processes [default: 4]
  --chunksize <c>             Size of data chunks each processor receives [default: 1]
"""
from ast import literal_eval
from pathlib import Path

import tqdm
from docopt import docopt
from collections import OrderedDict
from itertools import product
from multiprocessing.pool import Pool

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging

from matplotlib import ticker
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, SymmetricalLogLocator, LogFormatterExponent

from sklearn.linear_model import LinearRegression

sns.set(font_scale=1.5, rc={'text.usetex': True})

from rl_agents.agents.common.factory import load_environment, agent_factory
from rl_agents.trainer.evaluation import Evaluation

logger = logging.getLogger(__name__)

gamma = 0.7
SEED_MAX = 1e9


def env_configs():
    return ['env_garnet.json']


def agent_configs():
    agents = {
        "KL-OLOP": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "upper_bound": {
                "type": "kullback-leibler",
                "threshold": "1*np.log(time)"
            },
            "lazy_tree_construction": True,
            "continuation_type": "uniform",
        },
        "MDP-GapE": {
            "__class__": "<class 'rl_agents.agents.tree_search.mdp_gape.MDPGapEAgent'>",
            "gamma": gamma,
            "accuracy": 0,
            "confidence": 1,
            "upper_bound":
            {
                "type": "kullback-leibler",
                "time": "global",
                "threshold": "1*np.log(time)",
                "transition_threshold": "0.1*np.log(time)"
            },
            "max_next_states_count": 2,
            "continuation_type": "uniform",
            "step_strategy": "reset",
        },
        "BRUE": {
            "__class__": "<class 'rl_agents.agents.tree_search.brue.BRUEAgent'>",
            "gamma": gamma,
            "step_strategy": "reset",
        },
        "UCT": {
            "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
            "gamma": gamma,
            "closed_loop": True
        },
        "value_iteration": {
            "__class__": "<class 'rl_agents.agents.dynamic_programming.value_iteration.ValueIterationAgent'>",
            "gamma": gamma,
            "iterations": int(3 / (1 - gamma))
        }
    }
    return OrderedDict(agents)


def evaluate(experiment):
    # Prepare workspace
    seed, budget, agent_config, env_config, path = experiment
    gym.logger.set_level(gym.logger.DISABLED)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make environment
    env = load_environment(env_config)

    # Make agent
    agent_name, agent_config = agent_config
    agent_config["budget"] = int(budget)
    agent = agent_factory(env, agent_config)

    logger.debug("Evaluating agent {} with budget {} on seed {}".format(agent_name, budget, seed))

    # Compute true value
    env.seed(seed)
    observation = env.reset()
    vi = agent_factory(env, agent_configs()["value_iteration"])
    best_action = vi.act(observation)
    action = agent.act(observation)
    q = vi.state_action_value
    simple_regret = q[vi.mdp.state, best_action] - q[vi.mdp.state, action]
    gap = q[vi.mdp.state, best_action] - np.sort(q[vi.mdp.state, :])[-2]

    # Save results
    result = {
        "agent": agent_name,
        "budget": budget,
        "seed": seed,
        "simple_regret": simple_regret,
        "gap": gap
    }

    df = pd.DataFrame.from_records([result])
    with open(path, 'a') as f:
        df.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


def prepare_experiments(budgets, seeds, path):
    budgets = np.unique(np.logspace(*literal_eval(budgets)).astype(int))

    selected_agents = [
        "KL-OLOP",
        "MDP-GapE",
        "BRUE",
        "UCT"
    ]
    agents = {agent: config for agent, config in agent_configs().items() if agent in selected_agents}

    seeds = seeds.split(",")
    first_seed = int(seeds[0]) if len(seeds) == 2 else np.random.randint(0, SEED_MAX, dtype=int)
    seeds_count = int(seeds[-1])
    seeds = (first_seed + np.arange(seeds_count)).tolist()
    envs = env_configs()
    paths = [path]
    experiments = list(product(seeds, budgets, agents.items(), envs, paths))
    return experiments


latex_names = {
    "simple_regret": "simple regret $r_n$",
    "MDP-GapE": r"\texttt{MDP-GapE}",
    "KL-OLOP": r"\texttt{KL-OLOP}",
    "BRUE": r"\texttt{BRUE}",
    "GBOP": r"\texttt{GBOP}",
    "UCT": r"\texttt{UCT}",
    "budget": r"budget $n$",
}


def rename_df(df):
    df = df.rename(columns=latex_names)
    for key, value in latex_names.items():
        df["agent"] = df["agent"].replace(key, value)
    return df


def rename(value, latex=True):
    return latex_names.get(value, value) if latex else value


def plot_all(data_file, directory):
    print("Reading data from {}".format(directory / data_file))
    df = pd.read_csv(str(directory / data_file))
    df = df[~df.agent.isin(['agent'])].apply(pd.to_numeric, errors='ignore')
    df = df.sort_values(by="agent")

    m = df.loc[df['simple_regret'] != np.inf, 'simple_regret'].max()
    df['simple_regret'].replace(np.inf, m, inplace=True)

    df = rename_df(df)
    print("Number of seeds found: {}".format(df.seed.nunique()))

    with sns.axes_style("ticks"):
        sns.set_palette("colorblind")
        fig, ax = plt.subplots()
        field = "simple_regret"
        ax.set(xscale="log")
        ax.set_yscale("symlog", linthreshy=1e-3)

        sns.lineplot(x=rename("budget"), y=rename(field), ax=ax, hue="agent", style="agent", data=df)
        plt.legend(loc="lower left")

        field_path = directory / "{}.pdf".format(field)
        fig.savefig(field_path, bbox_inches='tight')
        field_path = directory / "{}.png".format(field)
        fig.savefig(field_path, bbox_inches='tight')
        print("Saving {} plot to {}".format(field, field_path))


def main(args):
    if args["--generate"] == "True":
        experiments = prepare_experiments(args["--budgets"], args['--seeds'],
                                          str(Path(args["--directory"]) / args["--data_file"]))
        chunksize = int(args["--chunksize"])
        with Pool(processes=int(args["--processes"])) as p:
            list(tqdm.tqdm(p.imap_unordered(evaluate, experiments, chunksize=chunksize), total=len(experiments)))
    if args["--show"] == "True":
        plot_all(args["--data_file"], Path(args["--directory"]))


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
