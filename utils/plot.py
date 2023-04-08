import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

filter_keywords = ["do_shape"]

def plot_data(data, xaxis='global_step', value="eval/mean_reward", condition=None,
              title=None, normalize=False, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):

    if normalize:
        for datum in data:
            datum[value] /= np.max(np.abs(datum[value]))

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    plt.figure(constrained_layout=True)
    sns.set(style="whitegrid")
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition,
                 linewidth=3, **kwargs)

    plt.legend(loc='best').set_draggable(True)
    # Change fontsize of legend
    for text in plt.gca().get_legend().get_texts():
        text.set_fontsize(24)

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # plt.tight_layout(pad=2.5)
    plt.title(title, fontsize=32)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # Make size of plot 2 inches by 4 inches
    plt.gcf().set_size_inches(12, 6)


def load_data(path, learning_starts=1000):
    df = pd.read_csv(path)
    df = df[df["global_step"] > learning_starts]
    # absolute all numeric values
    for col in df.columns:
        # if col != "clip_method":
        #     df[col] = df[col].abs()

        if col == "global_step":
            df[col] = df[col] / 1e3

    return df


def plot(value="eval/mean_reward", title=None, xlabel=None, ylabel=None,
         xlim=None, ylim=None, **kwargs):
    data_list = []
    files = os.listdir("export")
    cond_list = []
    for file_name in files:
        try:
            data = load_data("export/" + file_name)
            data_list.append(data)
            cond_list.append(file_name.split("-")[1])
        except Exception as e:
            print("Error loading file: {}, error {}".format(file_name, e))
    plot_data(
        data_list,
        xaxis="global_step",
        value=value,
        condition='do_shape',
        errorbar=("ci", 95),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        **kwargs
    )
    plt.savefig(f'plot_{title}.png', dpi=800)


if __name__ == "__main__":
    plot(title='Evaluation Rewards During Training',
         xlabel=r'Environment Steps ($\times 1,000$)', ylabel='Rewards',
         xlim=(0, 50), ylim=(-500,-50))
