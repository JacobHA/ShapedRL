import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from tbparse import SummaryReader

metrics_to_ylabel = {
    'eval/mean_reward': 'Average Evaluation Reward',
    'rollout/ep_rew_mean': 'Average Rollout Reward',
}
all_metrics = [
    'rollout/ep_rew_mean', 'eval/mean_reward'
]
sns.set_theme(style="whitegrid")
# use poster settings:
sns.set_context("poster")
# Make font color black:
plt.rcParams['text.color'] = 'black'


def plotter(env, folder, x_axis='step', metrics=all_metrics, exclude_algos=[],
            xlim=None, ylim=None, title=None):
    algo_data = pd.DataFrame()
    subfolders = glob(os.path.join(folder, '*'))
    print("Found subfolders:", subfolders)
    # Sort the subfolders for consistent plotting colors (later can make a dict):
    subfolders = sorted(subfolders)
    # Collect all the data into one dataframe for parsing into figures:
    for subfolder in subfolders:
        if not os.path.isdir(subfolder) or subfolder.endswith('.png'):
            continue
        sub_name = os.path.basename(subfolder)
        algo_name = sub_name.split('alpha=')[1].split('_')[0]

        if 'NO' in sub_name:
            algo_name = 'NO' + algo_name
        # alias 0 and 0.0:
        if algo_name == '0':
            algo_name = '0.0'
        if algo_name in exclude_algos:
            print(f"Skipping {algo_name}, in exclude_algos.")
            continue

        log_files = glob(os.path.join(subfolder, '*.tfevents.*'))
        if not log_files:
            print(f"No log files found in {subfolder}")
            continue

        # Require only one log file per folder:
        # assert len(log_files) == 1
        log_file = log_files[0]
        print("Processing", os.path.basename(subfolder))

        try:
            reader = SummaryReader(log_file)
            df = reader.scalars
            df = df[df['tag'].isin(metrics + [x_axis])]
            # Add a new column with the algo name:
            df['algo'] = algo_name
            # Add run number:
            df['run'] = os.path.basename(subfolder).split('_')[1]
            algo_data = pd.concat([algo_data, df])
        except Exception as e:
            print("Error processing", log_file)
            continue

    # Now, loop over all the metrics and plot them individually:
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        plt.title(title)
        # Filter the data to only include this metric:
        metric_data = algo_data[algo_data['tag'] == metric]
        if not metric_data.empty:

            # Plot the average reward through training for each algo:
            plt.figure(figsize=(12, 8))
            plt.title(title)
            no_shaping_data = []
            shaping_data = []
            for algo, data in metric_data.groupby('algo'):
                # Average over runs:
                mean_rwd = data.groupby('step')['value'].mean()
                std_rwd = data.groupby('step')['value'].std() / np.sqrt(data.groupby('step')['value'].count())
                # convert algo name to float:
                algo = algo.split('eta=')[-1].split('(')[0]
                if 'NO' in algo:
                    eta = float(algo[2:])
                    algo = 'NO shaping' 
                    eta = 1/(1+eta)
                    no_shaping_data.append((eta, mean_rwd, std_rwd))
                else:
                    eta = float(algo)
                    algo = r'Shaping ($\eta=0.5$)'
                    eta=1/(1+eta)
                    shaping_data.append((eta, mean_rwd, std_rwd))
                # color = 'b' if algo == 'Shaping' else 'r'
                # plt.errorbar(eta, mean_rwd.mean(), yerr=std_rwd.mean(), color=color, fmt='o-', label=algo)
            # sort the data by eta:
            no_shaping_data = sorted(no_shaping_data, key=lambda x: x[0])
            shaping_data = sorted(shaping_data, key=lambda x: x[0])
            plt.plot([x[0] for x in no_shaping_data], [x[1].mean() for x in no_shaping_data], 'o-', color='r', label='No shaping')
            plt.fill_between([x[0] for x in no_shaping_data], [x[1].mean() - x[2].mean() for x in no_shaping_data],
                             [x[1].mean() + x[2].mean() for x in no_shaping_data], color='r', alpha=0.2)
            plt.plot([x[0] for x in shaping_data], [x[1].mean() for x in shaping_data], 'o-', color='b', label='Shaping eta=0.5')
            plt.fill_between([x[0] for x in shaping_data], [x[1].mean() - x[2].mean() for x in shaping_data],
                                [x[1].mean() + x[2].mean() for x in shaping_data], color='b', alpha=0.2)
            # plt.legend()
            plt.xlabel('Discount factor')
            plt.ylabel('Avg Reward')
            # plt.legend()
            # remove duplicates from legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.ylim(0, 1.0)
            plt.xlim(0.9,1.0)
            plt.tight_layout()
            # plt.xscale('log')
            plt.savefig(os.path.join(folder, f"{metric.split('/')[-1]}-alpha-{env}.png"), dpi=300)

        else:
            print("No data to plot.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='FrozenLake-v0')
    args = parser.parse_args()
    env = args.env

    folder = f'./gamma-runs'

    plotter(env=env, folder=folder, metrics=['eval/mean_reward'], title=env)
    # plotter(env=env, folder=folder, metrics=['rollout/ep_rew_mean'], title=env)
