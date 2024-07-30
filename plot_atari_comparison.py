"""retrieve runs from wandb project in a specified group,
integrate the evaluation reward and calculate the percentage advantage/disadvantage
of our method compared to the baseline for each available environment
"""
import copy
import os
import re

import wandb
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from atari_utils import get_human_normalized_score


def plot_advantage(project, comparison_variable, value_variable="eval/mean_reward", baseline_value=False, cache=True, use_cached=True):
    api = wandb.Api()  # eval/mean_reward
    print("getting runs")
    runs = api.runs(project)
    if use_cached and os.path.exists(f"envs_{project}.pkl"):
        with open(f"envs_{project}.pkl", "rb") as f:
            envs = pickle.load(f)
    else:
        envs = {}
        for run in tqdm.tqdm(runs):
            try:
                if comparison_variable not in run.config:
                    continue
                if run.config["env_id"] not in envs:
                    envs[run.config["env_id"]] = {}
               
                key = f"{comparison_variable}={run.config[comparison_variable]}"
                # Append this key with empty dict:
                if key not in envs[run.config["env_id"]]:
                    envs[run.config["env_id"]][key] = {
                        "eval_reward_auc": 0,
                        "number_of_runs": 0,
                        "max_eval_reward": 0,
                        "number_of_steps": 0,
                    }
                # calculate reward auc
                eval_reward = run.history(keys=[value_variable])
                eval_reward = np.array(eval_reward).astype(float)
                # skip if there is less than 20% of the expected data
                if run.state != "finished":# or run.config[comparison_variable] not in [0,1]:
                    continue
                envs[run.config["env_id"]][key]['eval/mean_reward'] = eval_reward
                envs[run.config["env_id"]][key]['eval_reward_auc'] += np.sum(eval_reward)
                envs[run.config["env_id"]][key]['number_of_runs'] += 1
                envs[run.config["env_id"]][key]['number_of_steps'] += len(eval_reward)
                envs[run.config["env_id"]][key]['max_eval_reward'] = max(
                    envs[run.config["env_id"]][key]['max_eval_reward'], np.max(eval_reward)
                )
            except Exception as e:
                print("error in run", run.name, e)
                continue
        if cache:
            with open(f"envs_{project}.pkl", "wb") as f:
                pickle.dump(envs, f)
    # calculate percentage advantage/disadvantage
    compared_envs = {}
    for env_id, env_stat in envs.items():
        cont=False
        for key in env_stat:
            n_runs = env_stat[key]['number_of_runs']
            if n_runs < 1:
                print(f"Not enough runs for {env_id} ({n_runs})")
                cont = True
                break
            env_stat[key]['eval_reward_auc'] /= env_stat[key]['number_of_steps']
        if cont:
            env_stat['percentage_advantage'] = 0
            continue
        rew_base = env_stat[f"{comparison_variable}={baseline_value}"]['eval_reward_auc']

        values = env_stat.keys()
        for comparison_variable_value in values:
            rew_targ = env_stat[comparison_variable_value]['eval_reward_auc']
            adv = rew_targ / rew_base
            adv = 1 / adv if rew_base<0 and rew_targ<0 else adv
            # todo: add a baseline human performance for normalization
            env_stat[comparison_variable_value]['percentage_advantage'] = adv * 100 - 100
        compared_envs[env_id[:-len("NoFrameskip-v4")]] = env_stat

    # sort the environments by percentage advantage of 
    compared_envs = dict(
        sorted(compared_envs.items(), key=lambda item: item[1]['shape_scale=1']['percentage_advantage'])
    )
    # plot the results
    sns.set_theme(style="whitegrid")
    # use poster settings:
    sns.set_context("poster")
    # Make font color black:
    plt.rcParams['text.color'] = 'black'
    # plot the results
    plt.figure(figsize=(12, 8))
    # track the best comparison_variable for each env:
    for comparison_variable_value in values:
        if comparison_variable_value == f'{comparison_variable}={baseline_value}':
            continue
        plt.bar(compared_envs.keys(), [compared_envs[env][comparison_variable_value]['percentage_advantage'] for env in compared_envs],
                label=comparison_variable_value)

    plt.title(f"Percentage advantage of {comparison_variable} compared to baseline")
    plt.xlabel("Environment")
    plt.yscale("linear")
    plt.ylabel("Percentage advantage")
    plt.xticks(rotation=90)
    plt.legend()
    # make the x-axis labels fit in the plot
    plt.tight_layout()
    plt.savefig(f"advantage_{project}_{comparison_variable}.png")
    # plt.show()

    # Now plot the median human normalized scores
    plt.figure(figsize=(12, 8))

    # get all env eval rewards:
    for variable in list(values):
        env_rewards = {}
        for env_id, env_stat in envs.items():
            try:
                rwd = env_stat[variable]['eval/mean_reward']
                # get the human normalized scores
                human_norm = get_human_normalized_score(env_id, rwd[:,1])
                # if nan, skip:
                if any(np.isnan(human_norm)):
                    continue
                env_rewards[env_id] = human_norm

            except KeyError:
                print("Skipping", env_id, variable)

        # Take the median of all envs
        median_human_normalized = np.median(np.array(list(env_rewards.values())), axis=0)
        # plot the results
        plt.plot(median_human_normalized, label=variable)
    plt.legend()
    plt.title(f"Median human normalized score of {comparison_variable}")
    plt.savefig(f"human_norm_{project}_{comparison_variable}.png")

    # Do a similar plot with baseline and the finetuned (sort by percentage advantage) comparison variable:
    plt.figure(figsize=(12, 8))
    # Plot the baseline:
        # Plot the baseline:
    baseline_rewards = {}
    for env_id, env_stat in envs.items():
        try:
            rwd = env_stat[f"{comparison_variable}={baseline_value}"]['eval/mean_reward']
            human_norm = get_human_normalized_score(env_id, rwd[:, 1])
            if any(np.isnan(human_norm)):
                continue
            baseline_rewards[env_id] = human_norm
        except KeyError:
            print("Skipping", env_id)
    
    baseline_median_human_normalized = np.median(np.array(list(baseline_rewards.values())), axis=0)
    plt.plot(baseline_median_human_normalized, label='baseline')

    # Now pick the best shape scale value for each environment before aggregating:
    finetuned_data = {}
    for env_id, env_stat in envs.items():
        best_shape_scale = 0
        best_adv = 0
        for variable in values:
            if variable == f'{comparison_variable}={baseline_value}':
                continue
            try:
                adv = env_stat[variable]['percentage_advantage']
                if adv > best_adv:
                    best_adv = adv
                    best_shape_scale = variable
                    human_norm = get_human_normalized_score(env_id, env_stat[f"{best_shape_scale}"]['eval/mean_reward'][:,1])
                    if any(np.isnan(human_norm)):
                        continue
                    finetuned_data[env_id] = human_norm
                    print(f"Best shape scale for {env_id} is {best_shape_scale}")

            except KeyError:
                print("Skipping", env_id)
                continue

    median_human_normalized = np.median(np.array(list(finetuned_data.values())), axis=0)
    # plot the results
    plt.plot(median_human_normalized, label=variable)
    plt.legend()
    plt.title(f"Median human normalized score of {comparison_variable}")
    plt.tight_layout()
    plt.savefig(f"finetuned_{project}_{comparison_variable}.png")


plot_advantage('atari10m', 'shape_scale', cache=True, use_cached=1, baseline_value=0)