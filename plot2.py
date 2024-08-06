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
    # Load cached data if available
    if use_cached and os.path.exists(f"envs_{project}.pkl"):
        with open(f"envs_{project}.pkl", "rb") as f:
            envs = pickle.load(f)
    else:
        envs = download_data(project, value_variable, comparison_variable, cache)
    all_comparison_values = []
    # Calculate percentage advantage/disadvantage
    compared_envs = {}
    for env_id, env_stat in envs.items():
        values = env_stat.keys()
        # if len(values) != 4:#len(all_comparison_values):
        # if sorted(values) != sorted([f'{comparison_variable}={x}' for x in [0, 0.5, 1, 2]]):
        # if len(values) = 5:
            # continue
        cont = False
        for key in env_stat:
            n_runs = env_stat[key]['number_of_runs']
            if n_runs < 1:
                print(f"Not enough runs for {env_id} ({n_runs})")
                cont = True
                break
            env_stat[key]['eval_reward_auc'] = env_stat[key]['eval/mean_reward'].mean()
            all_comparison_values.append(key)
            
        
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

        compared_envs[env_id] = env_stat
    all_comparison_values = set(all_comparison_values)

    def get_percentage_advantage(item, sorting=False):
        try:
            if sorting:
                return max([item[v]['percentage_advantage'] for v in item if 'percentage_advantage' in item[v]])
            return item['percentage_advantage']
        except KeyError:
            return 0

    compared_envs = dict(
        sorted(compared_envs.items(), key=lambda item: get_percentage_advantage(item[1], True))
    )

    sns.set_theme(style="whitegrid")
    sns.set_context("poster")
    plt.rcParams['text.color'] = 'black'
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    plt.figure(figsize=(18, 10))

    # Exclude the baseline value from the count
    num_values = len(all_comparison_values) - 1  
    bar_width = 1.0 / num_values  # Width of each bar, adjusted for spacing
    positions = []
    plotting_envs = []
    value_to_color = dict(zip(all_comparison_values, sns.color_palette("deep", len(all_comparison_values))))
    # Prepare data for plotting

    gap_width = 0.15  # Define a small gap width
    env_to_finetuned_value = {}

    # Initialize lists to store the positions and labels for xticks
    xtick_positions = []
    xtick_labels = []

    for env_idx, env in enumerate(compared_envs):
        plotting_envs.append(env)
        base_position = len(plotting_envs) - 1  # This represents the base position for the current environment
        positions.append(base_position)
        idx = 0
        ft_adv = 0
        for i, comparison_variable_value in enumerate(sorted(all_comparison_values)):
            if comparison_variable_value == f"{comparison_variable}={baseline_value}":
                continue
            idx += 1

            env_data = compared_envs[env].get(comparison_variable_value, None)
            if env_data is None:
                continue

            height = get_percentage_advantage(env_data)
            if height > ft_adv:
                ft_adv = height
                env_to_finetuned_value[env] = comparison_variable_value


            # Calculate the offset for the current bar
            offset = idx * bar_width + base_position * (len(all_comparison_values) - 1) * bar_width + env_idx * gap_width

            # Plot the bar for the current environment and comparison variable value
            plt.bar(offset, height, width=bar_width, 
                    label=f"{comparison_variable_value}",
                    color=value_to_color[comparison_variable_value])

            # Add the base position to the xtick positions if it is the first bar of the clump
            if i == 1:
                xtick_positions.append(offset)
                xtick_labels.append(env)

    # Set the xticks and their labels
    plt.xticks(xtick_positions, xtick_labels)

    plt.title(f"Percentage advantage of {comparison_variable} compared to baseline")
    plt.xlabel("Environment")
    plt.yscale("symlog")
    # plt.ylim(-150,250)
    plt.ylabel("Percentage advantage")
    plt.xticks(rotation=90)
    # Get unique legend labels only:
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
    plt.tight_layout()
    plt.savefig(f"advantage_{project}_{comparison_variable}.png")

    plt.figure(figsize=(10, 6))

    baselines = []
    finetuneds = []
    for env_id, env_stat in compared_envs.items():
        # Plot the baseline and finetuned:
        baseline = env_stat.get(f"{comparison_variable}={baseline_value}", None)
        if baseline is None:
            continue
        finetuned = env_stat.get(env_to_finetuned_value.get(env_id.split('NoFrameskip-v4')[0], None), None)
        if finetuned is None:
            continue
        base_human = get_human_normalized_score(env_id, baseline['eval/mean_reward'])
        baselines.append(base_human)
        finetuned_human = get_human_normalized_score(env_id, finetuned['eval/mean_reward'])
        finetuneds.append(finetuned_human)

    baseline_median_human_normalized = np.median(np.array(baselines), axis=0)
    # do a rolling average
    window = 5
    # baseline_median_human_normalized = np.convolve(baseline_median_human_normalized, np.ones(window)/window, mode='full')
    plt.plot(x_axis, baseline_median_human_normalized, label='baseline')

    finetuned_median_human_normalized = np.median(np.array(finetuneds), axis=0)
    # finetuned_median_human_normalized = np.convolve(finetuned_median_human_normalized, np.ones(window)/window, mode='full')
    plt.plot(x_axis, finetuned_median_human_normalized, label='finetuned')


    # Add plot formatting options
    plt.xlabel('Environment Steps (Millions)', fontsize=16)
    plt.ylabel('Human Normalized Score', fontsize=16)
    plt.title('Comparison of Baseline and Finetuned Models', fontsize=18)
    plt.grid(True, linestyle='--', linewidth=0.5)
    # Set x-axis major locator and formatter
    import matplotlib.ticker as ticker
    plt.xlim(left=0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x / 1e6)}'))

    # Adjust tick positions to match the new formatting

    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(tick / 1e6)}' for tick in ticks])

    plt.tight_layout()

    # Save the plot with high resolution
    plt.savefig('baseline_vs_finetuned_icml.png')#, dpi=300)

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
    # smooth the curve with rolling avg but make sure it is same length:
    baseline_median_human_normalized = np.convolve(baseline_median_human_normalized, np.ones(window)/window, mode='full')
    baseline_median_human_normalized = baseline_median_human_normalized[:len(x_axis)]

    plt.plot(x_axis, baseline_median_human_normalized, 'k-',label='baseline',  linewidth=4)
    env_to_best_variable = {}

    for variable in all_comparison_values:
        if variable == f"{comparison_variable}={baseline_value}":
            continue
        # best_shape_scale = None
        # best_adv = float('-inf')
        all_data = {}
        for env_id in envs_with_baseline:
            env_stat = envs[env_id]

            try:
                human_norm = get_human_normalized_score(env_id, env_stat[variable]['eval/mean_reward'])
                if any(np.isnan(human_norm)):
                    continue
                # adv = np.nanmean(human_norm)
                # if adv > best_adv:
                #     best_adv = adv
                #     best_shape_scale = variable
                #     env_to_best_variable[env_id] = human_norm

            except KeyError:
                print(f"Skipping {env_id} for {variable}")
                continue
        
            all_data[env_id] = human_norm
        # if best_shape_scale is not None:
        #     print(f"Best shape scale for {env_id} is {best_shape_scale}")
        # else:
        #     print(f"No valid shape scale found for {env_id}")

        median_human_normalized = np.median(np.array(list(all_data.values())), axis=0)
        # smooth the curve with rolling avg but make sure it is same length:
        median_human_normalized = np.convolve(median_human_normalized, np.ones(window)/window, mode='full')
        median_human_normalized = median_human_normalized[:len(x_axis)]
        plt.plot(x_axis, median_human_normalized, label=fr'$\eta=${variable}, {len(all_data)} envs', linewidth=4)
    plt.legend(loc='upper left')
    plt.title("Finetuned Aggregated Reward Curve")
    plt.xlabel('Env. Steps')
    plt.ylabel('Median Human Normalized Score')
    plt.tight_layout()
    plt.savefig(f"finetuned_{project}_{comparison_variable}.png")
    plt.close()

    # num_envs = len(envs_with_baseline)
    # num_cols = 4
    # num_rows = num_envs // num_cols
    # if num_envs % num_cols != 0:
    #     num_rows += 1
    # fig, axs = plt.subplots(num_rows, num_cols, figsize=(100, 60))
    # envs_with_baseline = sorted(envs_with_baseline)
    # for i, env_id in enumerate(envs_with_baseline):
    #     # env_id = env_id.split('NoFrameskip-v4')[0]
    #     row = i // num_cols
    #     col = i % num_cols
    #     ax = axs[row, col]
    #     var = env_to_best_variable[env_id]

    #     ax.plot(x_axis, baseline_rewards[env_id], label='baseline')
    #     ax.plot(x_axis, env_to_best_variable[env_id], label='finetuned')
    #     ax.set_title(env_id.split('NoFrameskip-v4')[0])
    #     ax.set_xlabel('Env. Steps')
    #     ax.set_ylabel('Human Normalized Score')
    #     ax.legend()

    # plt.tight_layout()
    # plt.savefig(f"each_env_finetuned_{project}_{comparison_variable}_envs.png")
    # plt.close()

def download_data(project, value_variable="eval/mean_reward", comparison_variable="shape_scale", cache=True):
    api = wandb.Api()  # eval/mean_reward
    print("Getting runs...")
    runs = api.runs(project)

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
            
            env_id = run.config["env_id"].split('NoFrameskip-v4')[0]
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
            envs[env_id][key]['eval_reward_auc'] += np.sum(eval_reward)
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
                continue
                # env_stat[key]['eval/mean_reward'] = np.zeros_like(x_axis)  # Default to zero if no runs

    if cache:
        with open(f"envs_{project}.pkl", "wb") as f:
            pickle.dump(envs, f)

    return envs


# Example call
plot_advantage('atari10m', 'shape_scale', cache=True, use_cached=0, baseline_value=0)
