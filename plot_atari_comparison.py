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

x_axis = np.arange(0,10_000_000, 50_000)

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
        try:
            rew_base = env_stat[f"{comparison_variable}={baseline_value}"]['eval_reward_auc']
            _ = env_stat[f"{comparison_variable}=0.5"]['eval/mean_reward']
        except KeyError:
            print(f"Skipping {env_id} due to missing baseline")
            continue
        values = env_stat.keys()
        for comparison_variable_value in values:
            rew_targ = env_stat[comparison_variable_value]['eval_reward_auc']
            adv = rew_targ / rew_base
            adv = 1 / adv if rew_base<0 and rew_targ<0 else adv
            # todo: add a baseline human performance for normalization
            env_stat[comparison_variable_value]['percentage_advantage'] = adv * 100 - 100
        compared_envs[env_id[:-len("NoFrameskip-v4")]] = env_stat

    # sort the environments by percentage advantage of 
    def get_percentage_advantage(item, value=None):

        try:
            if value is None:
                return sum([item[v]['percentage_advantage'] for v in item])
            return item[value]['percentage_advantage']
        except KeyError:
            return 0

    compared_envs = dict(
        sorted(compared_envs.items(), key=lambda item: get_percentage_advantage(item[1]))
    )
    # plot the results
    sns.set_theme(style="whitegrid")
    # use poster settings:
    sns.set_context("poster")
    # Make font color black:
    plt.rcParams['text.color'] = 'black'
    # plot the results
    plt.figure(figsize=(18, 10))

    num_values = len(values) - 1  # Exclude the baseline value
    bar_width = 0.7 / num_values  # Width of each bar, adjusted for spacing

    # Create an array of positions for the bars
    positions = np.arange(len(compared_envs))

    for i, comparison_variable_value in enumerate(values):
        if comparison_variable_value == f'{comparison_variable}={baseline_value}':
            continue
        
        offset = i * bar_width  # Calculate the offset for the current set of bars
        plt.bar(positions + offset, 
                [get_percentage_advantage(compared_envs[env], comparison_variable_value) for env in compared_envs],
                width=bar_width,
                label=comparison_variable_value)

    # Set the x-ticks to the middle of the group of bars
    plt.xticks(positions + bar_width * (num_values - 1) / 2, compared_envs.keys())

    plt.title(f"Percentage advantage of {comparison_variable} compared to baseline")
    plt.xlabel("Environment")
    plt.yscale("symlog")
    plt.ylabel("Percentage advantage")
    plt.xticks(rotation=90)
    # put legend above figure:
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
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
        plt.plot(x_axis, median_human_normalized, label=variable)
    plt.legend()
    plt.title(f"Aggregated Reward Curve")
    plt.xlabel('Env. Steps')
    plt.ylabel('Median Human Normalized Score')
    plt.savefig(f"human_norm_{project}_{comparison_variable}.png")

    # Do a similar plot with baseline and the finetuned (sort by percentage advantage) comparison variable:
    plt.figure(figsize=(12, 8))
    # Plot the baseline:
    baseline_rewards = {}
    envs_with_baseline = list(envs.keys())
    for env_id, env_stat in envs.items():
        try:
            rwd = env_stat[f"{comparison_variable}={baseline_value}"]['eval/mean_reward']
            human_norm = get_human_normalized_score(env_id, rwd[:, 1])
            if any(np.isnan(human_norm)):
                envs_with_baseline.remove(env_id)
                continue
            baseline_rewards[env_id] = human_norm
        except KeyError:
            envs_with_baseline.remove(env_id)
            print("Skipping", env_id)
    
    baseline_median_human_normalized = np.median(np.array(list(baseline_rewards.values())), axis=0)
    plt.plot(x_axis, baseline_median_human_normalized, label='baseline')

    # Now pick the best shape scale value for each environment before aggregating:
    finetuned_data = {}
    for env_id in envs_with_baseline:
        env_stat = envs[env_id]
        best_shape_scale = None
        best_adv = float('-inf')
        for variable in values:
            try:
                human_norm = get_human_normalized_score(env_id, env_stat[f"{variable}"]['eval/mean_reward'][:,1])
                if any(np.isnan(human_norm)):
                    continue
                adv = np.nanmean(human_norm)
                if adv > best_adv:
                    best_adv = adv
                    best_shape_scale = variable
                    finetuned_data[env_id] = human_norm
            except KeyError:
                print(f"Skipping {env_id} for {variable}")
                continue
        
        if best_shape_scale is not None:
            print(f"Best shape scale for {env_id} is {best_shape_scale}")
        else:
            print(f"No valid shape scale found for {env_id}")

    # Calculate median human normalized score
    median_human_normalized = np.nanmedian(np.array(list(finetuned_data.values())), axis=0)


    plt.plot(x_axis, median_human_normalized, label=r'Finetuned $\eta$ for each env.')
    plt.legend(loc='upper left')
    plt.title("Finetuned Aggregated Reward Curve")
    plt.xlabel('Env. Steps')
    plt.ylabel('Median Human Normalized Score')
    plt.tight_layout()
    plt.savefig(f"finetuned_{project}_{comparison_variable}.png")
    plt.close()
plot_advantage('atari10m', 'shape_scale', cache=True, use_cached=0, baseline_value=0)