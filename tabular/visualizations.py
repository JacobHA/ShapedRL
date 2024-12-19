import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
# import imageio


def make_animation(desc, sequence, filename, fps=5, **kvargs):
    if os.path.exists(f'{filename}.mp4'):
        print(
            f'animation video already available for {filename}... continuing')
        return
    tmpdir = tempfile.gettempdir()
    files_list = []
    print('writing frames ...')
    for i, (paths_list, meta) in enumerate(sequence):
        ith_filename = f"{filename}_{i:05d}.png"
        filepath = os.path.join(tmpdir, ith_filename)
        kvargs.update(meta)
        plot_dist(desc, *paths_list, filename=filepath,
                  show_plot=False, **kvargs)
        plt.close()
        files_list.append(filepath)

    print('retrieving frames ...')
    frames = []
    for ith_filepath in files_list:
        image = imageio.imread(ith_filepath)
        frames.append(image)

    for _ in range(fps*3):
        frames.append(image)

    print('creating animation ...')
    # imageio.mimsave(f'{filename}.gif', frames, 'GIF', fps=fps)
    imageio.mimsave(f'{filename}.mp4', frames, 'MP4', fps=fps)

    print('removing temporary files ...')
    for ith_filepath in files_list:
        os.remove(ith_filepath)


def plot_dist(desc, *paths_list, ncols=4, filename=None, titles=None, main_title=None, figsize=None, show_values=False, show_plot=True, symbols_in_color=True, symbol_size=180, dpi=300):
    desc = np.asarray(desc, dtype='c')
    plt.figure()
    # use viridis as default colormap
    if len(paths_list) == 0:
        paths_list = [desc]
        axes = [plt.gca()]
    elif len(paths_list) == 1:
        fig = plt.figure(figsize=figsize)
        axes = [plt.gca()]
    elif len(paths_list) > 1:
        n_axes = len(paths_list)

        ncols = min(ncols, n_axes)
        nrows = (n_axes-1)//ncols+1

        figsize = (5*ncols, 5*nrows) if figsize is None else figsize
        fig, axes = plt.subplots(nrows, ncols, sharey=False, figsize=figsize)
        axes = axes.ravel()
    else:
        raise ValueError("Missing required parameter: path")

    if titles is not None:
        assert type(titles) == list
        assert len(titles) == len(paths_list)
    else:
        titles = [None] * len(paths_list)

    for axi, paths, title in zip(axes, paths_list, titles):
        if paths is None:
            # fig.delaxes(axi)
            continue
        if type(paths) == dict:
            data = paths['data']
            set_kvargs = paths['set']
            axi.plot(*data)
            axi.set_title(title)
            axi.set(**set_kvargs)
        else:
            draw_paths(desc, axi, paths, title, show_values,
                       symbols_in_color, symbol_size)

    if main_title is not None:
        plt.suptitle(main_title)
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        return plt.gcf()
    elif show_plot:
        plt.show()
    else:
        return plt.gcf(), axes


def draw_paths(desc, axi, paths, title=None, show_values=False, symbols_in_color=True, symbol_size=120):
    if paths is None:
        return
    nrow, ncol = desc.shape
    nsta = nrow * ncol
    out = np.ones(desc.shape + (3,), dtype=float)

    show_whole_maze = (desc.shape == paths.shape) and (desc == paths).all()
    if paths.shape in [desc.shape, (nsta,)] and not show_whole_maze:
        paths = paths - paths.min()
        if paths.max() > 0:
            paths = paths / paths.max()
        paths = paths.reshape(desc.shape)

        # Path: blue
        out[:, :, 0] = out[:, :, 1] = 1 - paths

    out = add_layout(desc, out)
    # use a cmap:
    print(out)
    # axi.imshow(out*0.15, cmap='viridis')
    axi.imshow(out)

    # show symbols for some special states
    axi.scatter(*np.argwhere(desc.T == b'S').T,
                color='#00CD00' if symbols_in_color else 'k', s=symbol_size, marker='o')
    axi.scatter(*np.argwhere(desc.T == b'G').T,
                color='#E6CD00' if symbols_in_color else 'k', s=symbol_size, marker='*')
    axi.scatter(*np.argwhere(desc.T == b'H').T,
                color='#E60000' if symbols_in_color else 'k', s=symbol_size, marker='X')
    axi.scatter(*np.argwhere(desc.T == b'C').T,
                color='#FF8000' if symbols_in_color else 'k', s=symbol_size, marker='D')
    axi.scatter(*np.argwhere(desc.T == b'N').T,
                color='#808080' if symbols_in_color else 'k', s=symbol_size, marker=6)

    if len(paths.shape) == 2 and paths.shape[0] == nsta:
        # looks like a policy, lets try to illustrate it with arrows
        # axi.scatter(*np.argwhere(desc.T == b'F').T, color='#FFFFFF', s=10)

        nact = paths.shape[1]

        if nact in [2, 3]:
            direction = ['left', 'right', 'stay']
        elif nact in [4, 5]:
            direction = ['left', 'down', 'right', 'up', 'stay']
        elif nact in [8, 9]:
            direction = ['left', 'down', 'right', 'up', 'stay',
                         'leftdown', 'downright', 'rightup', 'upleft']
        else:
            raise NotImplementedError

        for state, row in enumerate(paths):
            for action, prob in enumerate(row):
                action_str = direction[action]
                if action_str == 'stay':
                    continue
                if action_str == 'left':
                    d_x, d_y = -prob, 0
                if action_str == 'down':
                    d_x, d_y = 0, prob
                if action_str == 'right':
                    d_x, d_y = prob, 0
                if action_str == 'up':
                    d_x, d_y = 0, -prob
                if action_str == 'leftdown':
                    d_x, d_y = -prob / np.sqrt(2), prob / np.sqrt(2)
                if action_str == 'downright':
                    d_x, d_y = prob / np.sqrt(2), prob / np.sqrt(2)
                if action_str == 'rightup':
                    d_x, d_y = prob / np.sqrt(2), -prob / np.sqrt(2)
                if action_str == 'upleft':
                    d_x, d_y = -prob / np.sqrt(2), -prob / np.sqrt(2)
                if desc[state // ncol, state % ncol] not in [b'W', b'G', b'H']:
                    axi.arrow(state % ncol, state // ncol, d_x*0.4, d_y*0.4,
                              width=0.035, head_width=0.32*prob, head_length=0.2*prob,
                              fc='k', ec='k')

    elif paths.shape == desc.shape and show_values:
        for i, row in enumerate(paths):
            for j, value in enumerate(row):
                # if desc[state // ncol, state % ncol] not in [b'W', b'G']:
                if value != 0:
                    axi.text(j-0.4, i-0.15,
                             f"{value:.2f}", c='k', fontsize=10.)

    elif paths.shape == (2, nrow, ncol):
        # this is the signature for a force field. Let's plot this with arrows
        dx = np.cos(paths[0]) * paths[1] * 0.4
        dy = np.sin(paths[0]) * paths[1] * 0.4

        for row in range(nrow):
            for col in range(ncol):
                size = paths[1, row, col]
                axi.arrow(col, row, dx[row, col], dy[row, col], width=0.001,
                          head_width=0.15*size, head_length=0.15*size, fc='k', ec='k')

    if title is not None:
        axi.set_title(title)

    axi.set_xlim(-0.5, ncol - 0.5)
    axi.set_ylim(nrow - 0.5, -0.5)
    axi.get_xaxis().set_visible(False)
    axi.get_yaxis().set_visible(False)

    return out 

def add_layout(desc, out):

    walls = (desc == b'W')

    # Walls: black
    out[walls] = [0, 0, 0]

    return out


def save_err_plot(err, name=''):
    plt.figure()
    plt.title("Policy Error")
    plt.plot(err)
    plt.savefig(f'figures/policy_error_{name}.png')
    plt.close()


def save_policy_plot(desc, learned_policy, optimal_policy, name=''):
    plot_dist(desc, learned_policy, optimal_policy,
              titles=["Learned policy", "True policy"],
              filename=f'figures/policy_{name}.png')
