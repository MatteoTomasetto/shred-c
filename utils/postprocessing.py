import numpy as np
import matplotlib.pyplot as plt
import imageio
from IPython.display import clear_output as clc
from IPython.display import display
from matplotlib import colors
import matplotlib.cm as cm

# define error metrics
mae = lambda datatrue, datapred: (datatrue - datapred).abs().mean()
mse = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis = -1).mean()
mre = lambda datatrue, datapred: ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() / (datatrue).pow(2).sum(axis = -1).sqrt()).mean()
mse_numpy = lambda datatrue, datapred: (np.linalg.norm(datatrue - datapred, axis=-1)**2).mean()
mre_numpy = lambda datatrue, datapred: (np.linalg.norm(datatrue - datapred, axis=-1) / np.linalg.norm(datatrue, axis=-1)).mean()
num2p = lambda prob : ("%.2f" % (100*prob)) + "%"

# define colormaps
cmap = cm.get_cmap('terrain_r')
col = [cmap(i) for i in np.linspace(0, 1, 1000)]
col_negative = []
for i in range(1, len(col)):
    col_negative.append([1 - col[i][0], 1 - col[i][1], 1 - col[i][2]])
cmap = colors.LinearSegmentedColormap.from_list("", col_negative)
black = colors.LinearSegmentedColormap.from_list("", ["black", "black"])
white = colors.LinearSegmentedColormap.from_list("", ["white", "white"])

# plotting utilities
def multiplot(yts, plot, titles = None, fontsize = None, figsize = None, vertical = False, axis = False, save = False, name = "multiplot"):
    """
    Multi plot of different snapshots
    Input: list of snapshots, related plot function, plot options, save option and save path
    """
    
    plt.figure(figsize = figsize)
    for i in range(len(yts)):
        if vertical:
            plt.subplot(len(yts), 1, i+1)
        else:
            plt.subplot(1, len(yts), i+1)
        plot(yts[i])
        plt.title(titles[i], fontsize = fontsize)
        if not axis:
            plt.axis('off')
    
    if save:
    	plt.savefig(name.replace(".png", "") + ".png", transparent = True, bbox_inches='tight')


def trajectory(yt, plot, ut = None, title = None, times = None, fontsize = None, figsize = None, axis = False, save = False, name = 'gif'):
    """
    Trajectory gif
    Input: trajectory with dimension (sequence length, data shape), related plot function for a snapshot, plot inputs and options, save option and save path
    """

    arrays = []
        
    for i in range(yt.shape[0]):
        plt.figure(figsize = figsize)
        if ut is None:
            plot(yt[i])
        else:
            plot(yt[i], ut[i])
        plt.title(title, fontsize = fontsize)
        if times is not None:
            plt.xlabel("Time = " + str(round(times[i], 1)) + " sec")
        if not axis:
            plt.xticks([])
            plt.yticks([])
        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)


def trajectories(yts, plot, uts = None, titles = None, times = None, fontsize = None, figsize = None, vertical = False, axis = False, save = False, name = 'gif'):
    """
    Gif of different trajectories
    Input: list of trajectories with dimensions (sequence length, data shape), plot function for a snapshot, plot options, save option and save path
    """

    arrays = []

    for i in range(yts[0].shape[0]):

        plt.figure(figsize = figsize)
        for j in range(len(yts)):
            if vertical:
                plt.subplot(len(yts), 1, j+1)
            else:
                plt.subplot(1, len(yts), j+1)
            if uts is None:
                plot(yts[j][i])
            else:
                plot(yts[j][i], uts[j][i])
            if titles is not None:
                plt.title(titles[j], fontsize = fontsize)
            if times is not None:
                plt.xlabel("Time = " + str(round(times[i], 1)) + " sec")         
            if not axis:
                plt.xticks([])
                plt.yticks([])
        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)
