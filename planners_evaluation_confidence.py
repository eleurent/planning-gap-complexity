"""Usage: planners_evaluation.py [options]

Compare performances of several planners

Options:
  -h --help
  --generate <true or false>  Generate new data [default: True].
  --show <true_or_false>      Plot results [default: True].
  --directory <path>          Specify directory path [default: ./out/confidence].
  --data_file <path>          Specify output data file name [default: data.csv].
  --accuracies <start,end,N>  Accuracy epsilon required for planners [default: [1,0.5,0.2]].
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
        "MDP-GapE-conf": {
            "__class__": "<class 'rl_agents.agents.tree_search.mdp_gape.MDPGapEAgent'>",
            "gamma": gamma,
            "accuracy": 1.0,
            "confidence": 0.9,
            "upper_bound":
            {
                "type": "kullback-leibler",
                "time": "global",
                "threshold": "np.log(1/(1 - confidence)) + np.log(count)",
                "transition_threshold": "np.log(1/(1 - confidence)) + np.log(count)"
            },
            "max_next_states_count": 2,
            "continuation_type": "uniform",
            "horizon_from_accuracy": True,
            "step_strategy": "reset",
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
    seed, accuracy, agent_config, env_config, path = experiment
    gym.logger.set_level(gym.logger.DISABLED)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make environment
    env = load_environment(env_config)

    # Make agent
    agent_name, agent_config = agent_config
    agent_config["accuracy"] = float(accuracy)
    agent_config["budget"] = 10**9
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

    if hasattr(agent.planner, "budget_used"):
        budget = agent.planner.budget_used

    # Save results
    result = {
        "agent": agent_name,
        "budget": budget,
        "accuracy": agent.planner.config["accuracy"],
        "horizon": agent.planner.config["horizon"],
        "seed": seed,
        "simple_regret": simple_regret,
        "gap": gap
    }

    df = pd.DataFrame.from_records([result])
    with open(path, 'a') as f:
        df.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


def prepare_experiments(accuracies, seeds, path):
    accuracies = literal_eval(accuracies)

    selected_agents = [
        "MDP-GapE-conf",
    ]
    agents = {agent: config for agent, config in agent_configs().items() if agent in selected_agents}

    seeds = seeds.split(",")
    first_seed = int(seeds[0]) if len(seeds) == 2 else np.random.randint(0, SEED_MAX, dtype=int)
    seeds_count = int(seeds[-1])
    seeds = (first_seed + np.arange(seeds_count)).tolist()
    envs = env_configs()
    paths = [path]
    experiments = list(product(seeds, accuracies, agents.items(), envs, paths))
    return experiments


latex_names = {
    "simple_regret": "simple regret $r_n$",
    "total_reward": "total reward $R$",
    "mean_return": "mean return $E[R]$",
    "1/epsilon": r"${1}/{\epsilon}$",
    "MDP-GapE-conf": r"\texttt{MDP-GapE}",
    "MDP-GapE": r"\texttt{MDP-GapE}",
    "KL-OLOP": r"\texttt{KL-OLOP}",
    "BRUE": r"\texttt{BRUE}",
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
    df["1/epsilon"] = 1/df["accuracy"]

    m = df.loc[df['simple_regret'] != np.inf, 'simple_regret'].max()
    df['simple_regret'].replace(np.inf, m, inplace=True)

    df = rename_df(df)
    print("Number of seeds found: {}".format(df.seed.nunique()))

    with sns.axes_style("ticks"):
        sns.set_palette("colorblind")
        fig, ax = plt.subplots()
        field = "budget"
        ax.set(xscale="log", yscale="log")
        df_max = df.groupby(rename("1/epsilon"), as_index=False).max()
        sns.lineplot(x=rename("1/epsilon"), y=rename("budget"), ax=ax, data=df_max, ci=95)
        sns.scatterplot(x=rename("1/epsilon"), y=rename("budget"), ax=ax, data=df, edgecolor=None, alpha=0.05)

        field_path = directory / "{}.pdf".format(field)
        fig.savefig(field_path, bbox_inches='tight')
        field_path = directory / "{}.png".format(field)
        fig.savefig(field_path, bbox_inches='tight')
        print("Saving {} plot to {}".format(field, field_path))

    print_budget_from_epsilon(df, directory)


def print_budget_from_epsilon(df, directory):
    df = df[df["agent"] == rename("MDP-GapE-conf")]
    data = [{
                "epsilon": name,
                "max_regret": group[rename("simple_regret")].max(),
                "max_budget": float(group[rename("budget")].max()),
                "med_budget": group[rename("budget")].median()
            } for name, group in df.groupby("accuracy")]
    pd.set_option('display.float_format', '{:.2E}'.format)
    print(pd.DataFrame(data).sort_values(by=['epsilon'], ascending=False))

    a_df = df.groupby(rename("accuracy"), as_index=False).max()
    x = np.log10(a_df[rename("1/epsilon")].values.reshape(-1, 1))
    y = np.log10(a_df[rename("budget")].values.reshape(-1, 1))
    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)
    print("Regression", "a:", linear_regressor.coef_, "b:", linear_regressor.intercept_)


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
