import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from tbparse import SummaryReader
import matplotlib.ticker as ticker
# use a nicer color palette:

metrics_to_ylabel = {
    'eval/mean_reward': 'Average Evaluation Reward',
}
all_metrics = [
    'rollout/ep_rew_mean', 'eval/mean_reward'
]
sns.set_theme(style="whitegrid")
# use poster settings:
sns.set_context("poster")
# Make font color black:
plt.rcParams['text.color'] = 'black'
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})


def plotter(env, folder, x_axis='step', metrics=all_metrics, exclude_algos=[],
            xlim=None, ylim=None, title=None, lr_starts=0, max_step=None):
    algo_data = pd.DataFrame()
    subfolders = glob(os.path.join(folder, '*'))
    print("Found subfolders:", subfolders)
    # Sort the subfolders for consistent plotting colors (later can make a dict):
    subfolders = sorted(subfolders)
    # Collect all the data into one dataframe for parsing into figures:
    for subfolder in subfolders:
        if not os.path.isdir(subfolder) or subfolder.endswith('.png'):
            continue

        algo_name = os.path.basename(subfolder).split('_')[0]
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
            # exclude the steps before learning starts
            df = df[df['step'] > lr_starts]
            if max_step is not None:
                assert df['step'].max() >= max_step
                df = df[df['step'] <= max_step]
            # Add run number:
            df['run'] = os.path.basename(subfolder).split('_')[-1]
            algo_name_float = algo_name.split('eta')[-1]
            # replace "eta with $\eta$":
            algo_name_float = algo_name_float
            print(algo_name_float)
            algo_data = pd.concat([algo_data, df])
        except Exception as e:
            print("Error processing", log_file)
            continue

    # Now, loop over all the metrics and plot them individually:
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        # plt.title(title)

        # Filter the data to only include this metric:
        metric_data = algo_data[algo_data['tag'] == metric]
        # filter out any runs that have 'Humanoid' in the name:
        # metric_data = metric_data[~metric_data['algo'].str.contains('Humanoid')]
        
        if not metric_data.empty:
            print(f"Plotting {metric}...")
            # Append the number of runs to the legend for each algo:
            algo_runs = metric_data.groupby('algo')['run'].nunique()
            for algo, runs in algo_runs.items():
                # replace "eta" with "$\eta$":
                algo = algo.replace('eta', r'$\eta$')
                metric_data.loc[metric_data['algo'] == algo, 'algo'] = f"{algo} ({runs} runs)"
            # make a color palette that starts with black then uses the default color cycle:
            colors = sns.color_palette(n_colors=len(algo_runs)-1)
            colors = colors[0:1] + ['black'] + colors[1:]
            # sort by algo name:
            metric_data = metric_data.sort_values('algo')

            # sns.lineplot(data=metric_data, x='step', y='value', hue='algo', palette=colors, lw=5)
            # collapse the metric_data by taking mean over training time and plotting as one point:
            avg_values = [
                metric_data[metric_data['algo']==key]['value'].mean()
                for key in algo_runs.keys()]
            std_values = [
                metric_data[metric_data['algo']==key]['value'].std() / np.sqrt(algo_runs[key])
                for key in algo_runs.keys()]
            
            keys = [float(key.split('=')[-1]) for key in algo_runs.keys()]
            # sort both lists by keys:
            keys, avg_values, std_values = zip(*sorted(zip(keys, avg_values, std_values)))
            plt.plot(keys, avg_values, 'o-', color='orange', markersize=20, lw=5)
            plt.fill_between(keys, np.array(avg_values)-np.array(std_values), np.array(avg_values)+np.array(std_values), color='orange', alpha=0.3)
            plt.xscale('symlog')

            # strip the title from the values in legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            labels = []
            for handle in handles:
                label = handle.get_label()
                try:
                    # labels.append(label.split(env + '-')[-1])
                    labels.append(label.split('shape_')[-1])
                    print(labels)
                except TypeError:
                    labels.append(label)
                # swap U for EVAL:
                labels = [label.replace('U', 'EVAL') for label in labels]
                # Swap Rawlik for PPI:
                labels = [label.replace('rawlik', 'PPI') for label in labels]
                # Remove the number of runs:
                # labels = [label.split(' (')[0] for label in labels]
            # labels = [label.split(title+'-')[-1] for label in labels]
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            plt.xlim(xlim)
            plt.ylim(ylim)
            # increase x tick fontsize
            plt.xticks(fontsize=32)
            plt.yticks(fontsize=32)

            #plt.yticks([-1000, -600, -200])
            plt.xlabel(r'Shape scale, $\eta$', fontsize=38)
            plt.ylabel('Mean Training Reward', fontsize=38)
            # turn off the grid:
            plt.grid(False)

            plt.tight_layout()
            plt.savefig(os.path.join(folder, f"eta-sens-{env}.png"), dpi=500)
            plt.close()
        else:
            print("No data to plot.")

from run import env_to_steps

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='Humanoid-v4')
    args = parser.parse_args()
    env = args.env

    # folder = f'./reacher_runs_lrs10k'
    folder = f'./humanoid-dec18'
    # env_to_settings = {
    exclude_algos = ['td3Reacher-v4eta=10.0']
    # exclude_algos = ['td3eta=10.0', 'td3eta=3.0', 'td3eta=0.5']
    max_steps = env_to_steps.get(env, 1_000_000)
    # max_steps = 1_500_000
    # plotter(folder=folder, metrics=['eval/avg_reward'], ylim=(0, 510), exclude_algos=['CartPole-v1-U','CartPole-v1-Umin',  'CartPole-v1-Ured', 'CartPole-v1-Umean', 'CartPole-v1-Umse-b02', ])
    # plotter(folder=folder, metrics=['rollout/ep_reward'], ylim=(0, 510), exclude_algos=['CartPole-v1-U','CartPole-v1-Umin', 'CartPole-v1-Ured', 'CartPole-v1-Umean', 'CartPole-v1-Umse-b02', ])

    plotter(env=env, folder=folder, metrics=['eval/mean_reward'], exclude_algos=exclude_algos, title=env, lr_starts=10_000, max_step=max_steps)#, xlim=(0, 10_000))
    # plotter(env=env, folder=folder, metrics=['rollout/ep_reward'])
    # plotter(env=env, folder=folder, metrics=['rollout/avg_entropy'], exclude_algos=['Acrobot-v1-U', 'Acrobot-v1-SQL'], title=r'Relative Entropy $\mathrm{KL}\left(\pi|\pi_0\right)$')

    # plotter(folder=folder, metrics=['step', 'train/theta', 'theta'])
    # plotter(folder=folder, metrics=['step', 'train/avg logu', 'avg logu'])