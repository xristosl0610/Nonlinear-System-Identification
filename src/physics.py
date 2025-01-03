import numpy as np
import torch as T
from scipy.integrate import solve_ivp

from src.config_data_class import Config
from src.plotting import plot_jonswap_excitation


def apply_forcing(times: np.ndarray, params: Config) -> np.ndarray:
    """
    Returns the forcing term for the given time instances based on the specified parameters.

    Args:
        times (np.ndarray): A 1D array containing the discrete time values.
        params (Parameters): The parameters of the run, including the type of forcing and related parameters.

    Returns:
        np.ndarray: A 1D array containing the forcing term for the input discrete time instances.

    Raises:
        ValueError: If an invalid `force_flag` is provided in the `params`.
    """

    match params.physics.force_flag:
        case 'Jonswap':
            return np.expand_dims((np.expand_dims(params.physics.aps, 1)
                                   * np.cos((np.expand_dims(params.physics.omegaps, 1) * np.expand_dims(times, 0)
                                             + np.expand_dims(params.physics.eps, 1)))).sum(axis=0), axis=1)
        case 'mono':
            return params.physics.F0 * np.cos(params.physics.forcing_freq * times)
        case _:
            raise ValueError("Invalid excitation flag provided. Valid options are 'Jonswap' or 'mono'.")


def apply_known_physics(x, params) -> (np.ndarray | T.Tensor):
    """
    A function that returns the known part (terms) of the governing equation

    Args:
        x (numpy.ndarray | torch.Tensor): A 2D array/tensor containing the displacement in the first column
                                          and the velocity in the second one
        params (Config): The parameters of the run
    Returns:
        numpy.ndarray | torch.Tensor: An 1D array/tensor that contains known part of the governing equation
    """
    return - 2 * params.physics.zeta * params.physics.omega * x[:, 1] - params.physics.omega ** 2 * x[:, 0]


def build_true_model(x: np.ndarray, t: np.ndarray, params: Config) -> np.ndarray:
    """
    A function that gets the displacement, velocity and time as an input, and returns the true vector field output (velocity and acceleration)

    Args:
        x (numpy.ndarray): A 2D array containing the displacement in the first column and the velocity in the second one
        t (numpy.ndarray): An 1D array containing the discrete time values
        params (Config): The parameters of the run

    Returns:
        numpy.ndarray: A 2D array with the two vector field values, velocity as first column and acceleration as second,
                       for the given input x and t
    """

    if params.physics.friction_model == "C":
        friction_force = params.physics.friction_force_ratio
    elif params.physics.friction_model == "DR":
        friction_force = params.physics.friction_force_ratio \
                         + params.physics.DR["a"] * np.log((np.abs(x[1]) + params.physics.DR["eps"]) / params.physics.DR["V_star"]) \
                         + params.physics.DR["b"] * np.log(params.physics.DR["c"] + params.physics.DR["V_star"] / (np.abs(x[1]) + params.physics.DR["eps"]))
    elif params.physics.friction_model is None:
        friction_force = 0

    forcing = apply_forcing(t, params)
    # Check if there is sticking - Zero velocity and friction force bigger than the opposing forces
    if (np.abs(x[1]) < 1e-5) and (np.abs(forcing - params.physics.stiffness * x[0]) <= np.abs(friction_force) * params.physics.F0):
        return np.array([0., 0.])

    return np.array([x[1],
                     - 2 * params.physics.zeta * params.physics.omega * x[1]
                     - params.physics.omega ** 2 * x[0]
                     - friction_force * params.physics.F0 / params.physics.mass * np.sign(x[1])
                     + forcing / params.physics.mass], dtype=object)


def generate_data(
    params: Config
) -> (np.ndarray, np.ndarray):
    """
    Solves the system's equation and generates the ground truth data based on the defined run parameters.

    Args:
        params (Parameters): The parameters of the run, which define the system and its equations.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - A 1D numpy array with the discrete time instances.
            - A 2D numpy array with the ground truth data, where the first column represents displacements and the second column represents velocities.
    """
    ts = np.arange(0, params.physics.timefinal, params.hyperparams.timestep)

    sol = solve_ivp(
        lambda t, x: build_true_model(x, t, params),
        t_span=[ts[0], ts[-1]], y0=params.physics.x0, t_eval=ts
        )

    return ts, np.transpose(sol.y)


def calculate_jonswap_excitation(params: Config, plot_jonswap: bool = False) -> None:
    """
    Calculate JONSWAP wave excitation based on provided parameters.

    Args:
        params: The Config object containing parameters needed for the calculation.
        plot_jonswap: A boolean to plot the excitation. Defaults to False.

    Returns:
        None
    """
    Hs = 10
    Tp = 0.5
    fp = 1 / Tp
    omegap = 2 * np.pi / Tp
    omegas = np.arange(0.1, 50.01, 0.1)

    sigmap = np.where((omegas < omegap), 0.07, 0.09)

    gamma = 3.3
    fs = omegas / (2 * np.pi)
    beta = np.exp(-0.5 * ((fs / fp - 1) / sigmap) ** 2)
    Sigmaf = 0.3125 * Hs ** 2 * Tp * (fs / fp) ** (-5) * np.exp(-1.25 * (fs / fp) ** (-4)) * (1 - 0.287 * np.log(gamma)) * gamma ** beta

    inds = np.arange(omegas.shape[0])
    noOfHarmonics = omegas.shape[0]
    deltaf = fs[1] - fs[0]
    aps = np.sqrt(2 * Sigmaf[inds] * deltaf)
    eps = 2 * np.pi * np.random.rand(noOfHarmonics)

    ts = np.arange(0, params.physics.timefinal, params.hyperparams.timestep)

    eta = (np.expand_dims(aps, 1) * np.cos(
        (2 * np.pi * np.expand_dims(fs[inds], 1) * np.expand_dims(ts, 0) + np.expand_dims(eps, 1)))).sum(axis=0)

    params.physics.F0 = np.abs(eta).max()
    params.physics.aps = aps
    params.physics.eps = eps
    params.physics.omegaps = omegas[inds]

    if plot_jonswap:
        plot_jonswap_excitation(fs, Sigmaf, ts, eta)


def contaminate_measurements(params: Config, x_denoised) -> np.ndarray:
    """
    Contaminate measurements with noise based on the provided parameters.

    Args:
        params: The Config object containing parameters for noise contamination.
        x_denoised: The denoised measurements to be contaminated.

    Returns:
        np.ndarray: Contaminated measurements.
    """
    x = np.random.normal(loc=x_denoised, scale=params.physics.noise_level * np.abs(x_denoised), size=x_denoised.shape)

    if params.physics.noisy_input_flag:
        params.physics.omega = np.clip(np.random.normal(loc=params.physics.true_omega, scale=params.physics.omega_noise * params.physics.true_omega), a_max=None, a_min=0.1)
        params.physics.zeta = np.clip(np.random.normal(loc=params.physics.true_zeta, scale=params.physics.zeta_noise * params.physics.true_zeta), a_max=0.99, a_min=0.01)
    else:
        params.physics.omega = params.physics.true_omega
        params.physics.zeta = params.physics.true_zeta

    return x
