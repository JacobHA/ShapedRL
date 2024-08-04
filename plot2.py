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

x_axis = np.arange(0, 10_000_000, 50_000)
def plot_advantage(project, comparison_variable, value_variable="eval/mean_reward", baseline_value=False, cache=True, use_cached=True):
    api = wandb.Api()  # eval/mean_reward
    print("Getting runs...")
    runs = api.runs(project)
    
    # Load cached data if available
    if use_cached and os.path.exists(f"envs_{project}.pkl"):
        with open(f"envs_{project}.pkl", "rb") as f:
            envs = pickle.load(f)
    else:
        envs = {}
        for run in tqdm.tqdm(runs):
            try:
                if run.state != "finished":
                    continue
                history = run.history(keys=[value_variable])
                
                # Skip runs with any NaNs in the history
                if history.isna().any().any():
                    continue
                if comparison_variable not in run.config:
                    continue
                
                env_id = run.config["env_id"]
                config_value = run.config[comparison_variable]
                key = f"{comparison_variable}={config_value}"
                
                if env_id not in envs:
                    envs[env_id] = {}
                if key not in envs[env_id]:
                    envs[env_id][key] = {
                        "eval_reward_auc": 0,
                        "number_of_runs": 0,
                        "max_eval_reward": 0,
                        "eval/mean_reward": []
                    }
                
                eval_reward = np.array(run.history(keys=[value_variable])).astype(float)[:, 1]  # Ignore the time axis
                if len(eval_reward) != 200:
                    continue

                envs[env_id][key]['eval/mean_reward'].append(eval_reward)
                # envs[env_id][key]['eval_reward_auc'] += np.sum(eval_reward)
                envs[env_id][key]['number_of_runs'] += 1
                envs[env_id][key]['max_eval_reward'] = max(
                    envs[env_id][key]['max_eval_reward'], np.max(eval_reward)
                )
            except Exception as e:
                print("Error in run", run.name, e)
                continue
        
        # Normalize the mean reward
        for env_id, env_stat in envs.items():
            for key in env_stat:
                if env_stat[key]['number_of_runs'] > 0:
                    env_stat[key]['eval/mean_reward'] = np.mean(np.array(env_stat[key]['eval/mean_reward']), axis=0)
                else:
                    env_stat[key]['eval/mean_reward'] = np.zeros_like(x_axis)  # Default to zero if no runs

        if cache:
            with open(f"envs_{project}.pkl", "wb") as f:
                pickle.dump(envs, f)

    # Calculate percentage advantage/disadvantage
    compared_envs = {}
    for env_id, env_stat in envs.items():
        values = env_stat.keys()

        cont = False
        for key in env_stat:
            n_runs = env_stat[key]['number_of_runs']
            if n_runs < 1:
                print(f"Not enough runs for {env_id} ({n_runs})")
                cont = True
                break
            env_stat[key]['eval_reward_auc'] = env_stat[key]['eval/mean_reward'].mean()
            
        
        if cont:
            env_stat['percentage_advantage'] = 0
            continue

        try:
            rew_base = env_stat[f"{comparison_variable}={baseline_value}"]['eval_reward_auc']
        except KeyError:
            print(f"Skipping {env_id} due to missing baseline")
            continue

        for comparison_variable_value in values:
            if comparison_variable_value == f"{comparison_variable}={baseline_value}":
                continue
            rew_targ = env_stat.get(comparison_variable_value, {}).get('eval_reward_auc', None)
            if rew_targ is None:
                continue
            
            adv = rew_targ / rew_base
            if rew_base < 0 and rew_targ < 0:
                adv = 1 / adv
            env_stat[comparison_variable_value]['percentage_advantage'] = adv * 100 - 100

        compared_envs[env_id[:-len("NoFrameskip-v4")]] = env_stat

    def get_percentage_advantage(item, value=None):
        try:
            if value is None:
                return sum([item[v]['percentage_advantage'] for v in item if 'percentage_advantage' in item[v]])
            return item[value]['percentage_advantage']
        except KeyError:
            return 0

    compared_envs = dict(
        sorted(compared_envs.items(), key=lambda item: get_percentage_advantage(item[1]))
    )

    sns.set_theme(style="whitegrid")
    sns.set_context("poster")
    plt.rcParams['text.color'] = 'black'
    plt.figure(figsize=(18, 10))

    num_values = len(values) - 1  # Exclude the baseline value
    bar_width = 0.7 / num_values  # Width of each bar, adjusted for spacing
    positions = np.arange(len(compared_envs))

    # Prepare data for plotting
    for i, comparison_variable_value in enumerate(values):
        if comparison_variable_value == f'{comparison_variable}={baseline_value}':
            continue
        
        offset = i * bar_width  # Calculate the offset for the current set of bars
        heights = [get_percentage_advantage(compared_envs[env], comparison_variable_value) for env in compared_envs]
        plt.bar(positions + offset, heights, width=bar_width, label=comparison_variable_value)

    plt.xticks(positions + bar_width * (num_values - 1) / 2, compared_envs.keys())
    plt.title(f"Percentage advantage of {comparison_variable} compared to baseline")
    plt.xlabel("Environment")
    plt.yscale("symlog")
    plt.ylabel("Percentage advantage")
    plt.xticks(rotation=90)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
    plt.tight_layout()
    plt.savefig(f"advantage_{project}_{comparison_variable}.png")

    plt.figure(figsize=(12, 8))

    for variable in list(values):
        env_rewards = {}
        for env_id, env_stat in envs.items():
            try:
                rwd = env_stat[variable]['eval/mean_reward']
                human_norm = get_human_normalized_score(env_id, rwd)
                if any(np.isnan(human_norm)):
                    continue
                env_rewards[env_id] = human_norm
            except KeyError:
                print("Skipping", env_id, variable)

        median_human_normalized = np.median(np.array(list(env_rewards.values())), axis=0)
        plt.plot(x_axis, median_human_normalized, label=variable)
    plt.legend()
    plt.title(f"Aggregated Reward Curve")
    plt.xlabel('Env. Steps')
    plt.ylabel('Median Human Normalized Score')
    plt.savefig(f"human_norm_{project}_{comparison_variable}.png")

    plt.figure(figsize=(12, 8))
    baseline_rewards = {}
    envs_with_baseline = list(envs.keys())
    for env_id, env_stat in envs.items():
        try:
            rwd = env_stat[f"{comparison_variable}={baseline_value}"]['eval/mean_reward']
            human_norm = get_human_normalized_score(env_id, rwd)
            if any(np.isnan(human_norm)):
                envs_with_baseline.remove(env_id)
                continue
            baseline_rewards[env_id] = human_norm
        except KeyError:
            envs_with_baseline.remove(env_id)
            print("Skipping", env_id)
    
    baseline_median_human_normalized = np.median(np.array(list(baseline_rewards.values())), axis=0)
    plt.plot(x_axis, baseline_median_human_normalized, label='baseline')

    finetuned_data = {}
    for env_id in envs_with_baseline:
        env_stat = envs[env_id]
        best_shape_scale = None
        best_adv = float('-inf')
        for variable in values:
            try:
                human_norm = get_human_normalized_score(env_id, env_stat[variable]['eval/mean_reward'])
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

    median_human_normalized = np.nanmedian(np.array(list(finetuned_data.values())), axis=0)
    plt.plot(x_axis, median_human_normalized, label=r'Finetuned $\eta$ for each env.')
    plt.legend(loc='upper left')
    plt.title("Finetuned Aggregated Reward Curve")
    plt.xlabel('Env. Steps')
    plt.ylabel('Median Human Normalized Score')
    plt.tight_layout()
    plt.savefig(f"finetuned_{project}_{comparison_variable}.png")
    plt.close()

    num_envs = len(envs_with_baseline)
    num_cols = 4
    num_rows = num_envs // num_cols
    if num_envs % num_cols != 0:
        num_rows += 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(100, 60))
    envs_with_baseline = sorted(envs_with_baseline)
    for i, env_id in enumerate(envs_with_baseline):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        ax.plot(x_axis, baseline_rewards[env_id], label='baseline')
        ax.plot(x_axis, finetuned_data[env_id], label='finetuned')
        ax.set_title(env_id.split('NoFrameskip-v4')[0])
        ax.set_xlabel('Env. Steps')
        ax.set_ylabel('Human Normalized Score')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"each_env_finetuned_{project}_{comparison_variable}_envs.png")
    plt.close()

# Example call
plot_advantage('atari10m', 'shape_scale', cache=True, use_cached=0, baseline_value=0)
