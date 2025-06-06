import matplotlib.pyplot as plt
import time

def plot_heatmap(u, v, t, ax=None, colors_volt='RdYlBu_r'):
    if ax is None:
        fig, ax = plt.subplots()  # create new axis if not provided
        created_fig = True
    else:
        created_fig = False

    heatmap = ax.imshow(u, cmap=colors_volt, origin='lower', vmin=-0.4, vmax=1.2)
    plt.colorbar(heatmap, location='left', ax=ax, fraction=0.046, pad=0.04, label="Normalized Voltage")

    ax.set_title(f'Voltage Map at t = {round(t,2)} ms', fontsize=24)
    ax.set_xlabel('x-space', fontsize=16)
    ax.set_ylabel('y-space', fontsize=16)
    ax.axis('off')

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig("VoltMap" + timestr + ".png")

    # Clear or close figure
    if created_fig:
        plt.close()  # if we created the figure, just close it
    else:
        ax.clear()   # if user passed in an ax, just clear the axis
