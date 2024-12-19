import copy
import os
import re

import wandb
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from atari_utils import get_human_normalized_score

x_axis = np.arange(0, 10_000_000, 50_000)


def plot_advantage(project, comparison_variable, value_variable="eval/mean_reward", baseline_value=False, cache=True,
                   use_cached=True):
    # Load cached data if available
    baseline_label = f"{comparison_variable}={baseline_value}"
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
            rew_base = env_stat[baseline_label]['eval_reward_auc']
        except KeyError:
            print(f"Skipping {env_id} due to missing baseline")
            continue

        for comparison_variable_value in values:
            if comparison_variable_value == baseline_label:
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
    env_scores = []
    for idx, (env_str, env_stat) in enumerate(compared_envs.items()):
        # for env_idx, comparison_variable_value in enumerate(all_comparison_values):
        scores = [x['percentage_advantage'] for k, x in env_stat.items() if k != baseline_label]
        max_score = max(scores)
        idx_max = scores.index(max_score)
        env_scores.append(max_score)
        xtick_labels.append(env_str)
        xtick_positions.append(idx)

    cmap = cm.get_cmap('jet')
    norm = mcolors.Normalize(vmin=min(env_scores), vmax=max(env_scores))
    # colors = [cmap(norm(x*1000)) for x in env_scores]
    # Use bright blue for positive adv and bright red for negative adv:
    colors = ['blue' if x > 0 else 'red' for x in env_scores]
    plt.bar([i for i in range(len(env_scores))], env_scores, color=colors)#[value_to_color[x] for x in compared_envs[env_str].keys()])
    # Set the xticks and their labels
    plt.xticks(xtick_positions, xtick_labels)

    plt.title(fr"Relative advantage (%) of finetuned $\eta$ compared to baseline")
    # draw a threshold line at 0
    plt.axhline(0, color='black', linewidth=1.5)
    # plt.xlabel("Environment")
    plt.yscale("symlog")
    plt.ylim(-400,400)
    plt.ylabel("Percentage advantage")
    plt.xticks(rotation=90)
    # Get unique legend labels only:
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
    plt.tight_layout()
    plt.savefig(f"finetuned_advantage_{project}_{comparison_variable}.png")

    plt.figure(figsize=(10, 6))

    # Plot the finetuned reward curve, taking the median human normalized score:
    ft_rwd_curve = {}
    for env_str, env_stat in compared_envs.items():
        # for comparison_variable_value in all_comparison_values:
        #     if comparison_variable_value == baseline_label:
        #         continue
        #     try:
        #         percentage_advantage = env_stat[comparison_variable_value]['percentage_advantage']
        #     except KeyError:
        #         continue
        #     if percentage_advantage < 0:
        #         continue
            scores = [x['eval/mean_reward'] for k, x in env_stat.items()]# if k != baseline_label]
            # Get the max array based on the sum of the various entries:
            scores = max(scores, key=lambda x: np.sum(x))


            # # convert to human normalized score
            # scores = get_human_normalized_score(env_str, scores,
            # if nan, skip
            if np.isnan(scores).any():
                continue
            ft_rwd_curve[env_str] = scores

    # ft_rwd_curve = np.array(ft_rwd_curve)
    # median_rwd_curve = np.median(ft_rwd_curve, axis=0)
    # take median over the dicts items:
    median_rwd_curve = np.median(np.array(list(ft_rwd_curve.values())), axis=0)
    # plot:
    plt.plot(x_axis, median_rwd_curve, label="Finetuned reward")
    # plt.plot(x_axis, get_human_normalized_score(x_axis), label="Human normalized score")
    plt.title(f"Median finetuned reward curve for {project}")
    plt.xlabel("Number of frames")
    plt.ylabel("Mean reward")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"finetuned_reward_curve_{project}_{comparison_variable}.png")


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
plot_advantage('atari10m', 'shape_scale', cache=True, use_cached=False, baseline_value=0)