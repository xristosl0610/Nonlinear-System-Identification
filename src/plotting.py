import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from typing import Tuple


def plot_jonswap_excitation(fs, Sigmaf, ts, eta) -> None:
    """
    Plot JONSWAP wave excitation based on the provided parameters.

    Args:
        fs: Frequencies for plotting.
        Sigmaf: Spectral density values for plotting.
        ts: Time values for plotting.
        eta: Displacement values for plotting.

    Returns:
        None
    """
    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
    ax1.plot(fs, Sigmaf)

    ax1.set_ylabel(r"Spectral Density [m$^2$/Hz]", fontsize=24)
    ax1.set_xlabel(r"Frequency, $f$ [Hz]", fontsize=24)
    ax2.plot(ts, eta)
    ax2.set_xlabel(r"$t$ [s]", fontsize=24)
    ax2.set_ylabel(r"Displacement [m]", fontsize=24)

    ax1.set_xticks(np.arange(0, 8.1, 1), np.arange(0, 8.1, 1), fontsize=20)
    ax1.set_yticks(np.arange(0, 10.1, 2), np.arange(0, 10.1, 2), fontsize=20)
    ax2.set_xticks(np.arange(0, 5.1, 1), np.arange(0, 5.1, 1), fontsize=20)
    ax2.set_yticks(np.arange(-4, 4.1, 2), np.arange(-4, 4.1, 2), fontsize=20)

    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    plt.show()


def plot_results(times, exact_sol, measurements, learnt_sol, child_dir, type_flag='d', save_flag=True, figsize=(16, 8)):
    """"
    A function that plots and saves the comparison between the learnt solution, the exact solution and the used measurements

    Parameters
    ----------
    times : numpy.ndarray
        an 1D matrix containing the discrete time values
    exact_sol : numpy.ndarray
        a 2D array containing displacement and velocity values for each time step according to the exact equation
    measurements : numpy.ndarray
        a 2D array containing noisy displacement and velocity measurement for each time step
    learnt_sol : str
        a 2D array containing displacement and velocity values for each time step according to the learnt equation
    child_dir : str
        the path of the directory that will contain the results of the current run
    type_flag : bool
        if set to 'd', the displacement plot will be generated/saved, if set to 'v', the velocity one
    """
    plt.figure(figsize=figsize)

    plt.plot(times[::20], measurements[::20], "ro", markersize=5, label="Measurements")
    plt.plot(times, learnt_sol, "k--", linewidth=2, label="RK4-SINDy")
    plt.plot(times, exact_sol, alpha=0.3, linewidth=4, label="Ground truth")

    plt.legend(fontsize=22)  # , bbox_to_anchor=(1., 0.6))#, loc='upper right')

    plt.xlabel(r"$t \, \, \mathrm{[s]}$", fontsize=24)
    if type_flag == 'd':
        plt.ylabel(r"$x(t) \, \, \mathrm{[m]}$", fontsize=24)
    elif type_flag == 'v':
        plt.ylabel(r"$\dot{x}(t) \, \, \mathrm{[m/s]}$", fontsize=24)
    else:
        print("A wrong type_flag was given\nIt should be either 'd' for displacements, or 'v' for velocities")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if save_flag:
        plt.savefig(os.path.join(child_dir, f"{type_flag}.png"), bbox_inches='tight')

    plt.show()


def plot_error(times: np.ndarray,
               exact_sol: np.ndarray,
               learnt_sol: np.ndarray,
               nrmse: np.ndarray,
               child_dir: str,
               type_flag: str = 'd',
               save_flag: bool = True,
               figsize: Tuple[int] = (16, 12)) -> None:
    """

    Args:
        times (np.ndarray): an 1D matrix containing the discrete time values
        exact_sol (np.ndarray): a 2D array containing displacement and velocity values for each time step according to the exact equation
        learnt_sol (np.ndarray): a 2D array containing displacement and velocity values for each time step according to the learnt equation
        nrmse (np.ndarray): a 2D array containing the normalized root mean squared error between the exact solution and the learnt one,
                            for both displacements and velocities
        child_dir (str): the path of the directory that will contain the results of the current run
        type_flag (str): if set to 'd', the displacement plot will be generated/saved, if set to 'v', the velocity one
        save_flag (bool):
        figsize (Tuple[int]):

    Returns:
        None
    """
    if type_flag == 'd':
        ind = 0
    elif type_flag == 'v':
        ind = 1
    else:
        print("A wrong type_flag was given\nIt should be either 'd' for displacements, or 'v' for velocities")

    squaredError = np.abs(exact_sol - learnt_sol)

    _, ax = plt.subplots(figsize=figsize)

    ax.plot(times, squaredError[:, ind], label="Squared Error")

    textstr = f"NRMSE : {nrmse[ind]:.4f}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=24,
            verticalalignment='top', bbox=props)

    ax.set_xlabel(r"$t \, \, \mathrm{[s]}$", fontsize=24)
    if ind == 0:
        ax.set_ylabel(r"$\mid x_{\mathrm{true}} - x_{\mathrm{learned}} \mid \, \, \mathrm{[m]}$", fontsize=24)
    elif ind == 1:
        ax.set_ylabel(r"$\mid \dot{x}_{\mathrm{true}} - \dot{x}_{\mathrm{learned}}\mid \, \, \mathrm{[m/s]}$", fontsize=24)

    ax.tick_params(axis='both', which='major', labelsize=20)

    if save_flag:
        plt.savefig(os.path.join(child_dir, "err.png"), bbox_inches='tight')

    plt.show()
