from dataclasses import dataclass, field, is_dataclass
from typing import Any
import toml
import numpy as np


@dataclass
class HyperparamsConfig:
    """
    A data class representing hyperparameters configuration settings.

    Args:
        num_epochs: Number of training epochs.
        num_iter: Number of iterations.
        lr: Learning rate.
        weightdecay: Weight decay coefficient.
        tol_coeffs: Tolerance for coefficients.
        scaling: Flag for feature scaling.
        timestep: Timestep for simulation.

    Returns:
        HyperparamsConfig: A HyperparamsConfig object with specified hyperparameters configuration settings.
    """
    num_epochs: int = 1000
    num_iter: int = 3
    lr: float = 1e-1
    weightdecay: float = 0.0
    tol_coeffs: float = 1e-1
    scaling: bool = True
    timestep: float = 1e-2


@dataclass
class FeaturesConfig:
    """
    A data class representing features configuration settings.

    Args:
        poly_order: Polynomial order for feature expansion.
        cos_phases: Array of cosine phases.
        sin_phases: Array of sine phases.
        x_sgn_flag: Flag for including sign of x in features.
        y_sgn_flag: Flag for including sign of y in features.
        log_1: Flag for applying logarithm to feature 1.
        log_2: Flag for applying logarithm to feature 2.

    Returns:
        FeaturesConfig: A FeaturesConfig object with specified features configuration settings.
    """
    poly_order: int = 1
    cos_phases: np.ndarray | list | tuple = ()
    sin_phases: np.ndarray | list | tuple = ()
    x_sgn_flag: bool = False
    y_sgn_flag: bool = False
    log_1: bool = False
    log_2: bool = False


@dataclass
class DieterichRuinaFriction:
    """
    A data class representing the parameters for the Dieterich-Ruina friction model.

    This class encapsulates the parameters used in the Dieterich-Ruina friction model, which describes the relationship between normal stress and frictional resistance. Each parameter can be adjusted to fit specific frictional behavior in simulations or calculations.

    Args:
        a (float): Parameter a, default is 0.07.
        b (float): Parameter b, default is 0.09.
        c (float): Parameter c, default is 0.022.
        V_star (float): Characteristic velocity, default is 0.003.
        eps (float): Small positive constant to avoid division by zero, default is 1e-1.
    """
    a: float = 0.07
    b: float = 0.09
    c: float = 0.022
    V_star: float = 0.003
    eps: float = 1e-1


@dataclass
class PhysicsConfig:
    """
    A data class representing physics configuration settings.

    Args:
        timefinal: Final time for simulation.
        mass: Mass of the system.
        stiffness: Stiffness of the system.
        damping: Damping coefficient.
        omega: Calculated natural frequency (init-only).
        zeta: Calculated damping ratio (init-only).
        forcing_freq: Frequency of external forcing.
        F0: Amplitude of external forcing.
        force_flag: Flag indicating the type of forcing.
        friction_model_ind: Index representing the friction model.
        friction_model: Friction model used (init-only).
        friction_force_ratio: Ratio of friction force.
        DR: Dictionary containing parameters for the DR friction model.
        x0: Initial position and velocity tuple.
        true_omega: True natural frequency for comparison.
        true_zeta: True damping ratio for comparison.
        noisy_measure_flag: Flag for noisy measurements.
        noise_level: Level of measurement noise.
        noisy_input_flag: Flag for noisy inputs.
        omega_noise: Noise level for natural frequency.
        zeta_noise: Noise level for damping ratio.
        stick_tol: Tolerance for stick-slip transitions.

    Returns:
        PhysicsConfig: A PhysicsConfig object with specified physics configuration settings.
    """
    timefinal: float = 50.0
    mass: float = 1.0
    stiffness: float = 1.0
    damping: float = 0.1
    omega: float = field(init=False)
    zeta: float = field(init=False)
    forcing_freq: float = 0.175
    F0: float = 1.0
    force_flag: str = 'mono'
    aps: (np.ndarray | None) = None
    eps: (np.ndarray | None) = None
    omegaps: (np.ndarray | None) = None
    friction_model_ind: int = 1
    friction_model: str = field(init=False)
    friction_force_ratio: float = 0.5
    DR: DieterichRuinaFriction | None = DieterichRuinaFriction
    x0: tuple[float, float] = (0.1, 0.1)
    true_omega: float = 1.0
    true_zeta: float = 0.05
    noisy_measure_flag: bool = True
    noise_level: float = 1e-1
    noisy_input_flag: bool = False
    omega_noise: float = 5e-2
    zeta_noise: float = 2e-1
    stick_tol: float = 1e-3
    mus: (np.ndarray | None) = None
    stds: (np.ndarray | None) = None

    @staticmethod
    def __post_init__():
        PhysicsConfig.omega = np.sqrt(PhysicsConfig.stiffness / PhysicsConfig.mass)
        PhysicsConfig.zeta = PhysicsConfig.damping / (2 * np.sqrt(PhysicsConfig.stiffness * PhysicsConfig.mass))
        PhysicsConfig.friction_model = [None, "C", "DR"][PhysicsConfig.friction_model_ind]


@dataclass
class Config:
    """
    A data class representing configuration settings.

    Args:
        hyperparams: Hyperparameters configuration object or None.
        features: Features configuration object or None.
        physics: Physics configuration object or None.

    Returns:
        Config: A Config object with hyperparams, features, and physics configurations.
    """
    hyperparams: (HyperparamsConfig | None) = HyperparamsConfig
    features: (FeaturesConfig | None) = FeaturesConfig
    physics: (PhysicsConfig | None) = PhysicsConfig


def update_params(obj: Any, updates: dict) -> None:
    """
    Recursively update attributes of an object based on a dictionary.

    Args:
        obj (Any): The object whose attributes will be updated.
        updates (dict): A dictionary containing the updates to be applied to the object's attributes.

    Returns:
        None
    """
    for key, value in updates.items():
        if hasattr(obj, key):
            attr = getattr(obj, key)
            if is_dataclass(attr) and isinstance(value, dict):
                update_params(attr, value)
            else:
                setattr(obj, key, value)


def update_params_from_toml(params: Config, toml_file: str) -> Config:
    """
    Update the parameters of a Config object from a TOML file.

    Args:
        params (Config): The Config object whose parameters will be updated.
        toml_file (str): The path to the TOML file containing parameter updates.

    Returns:
        Config: The Config object with updated parameters.
    """
    config_dict = toml.load(toml_file)
    update_params(params, config_dict)
    params.physics.__post_init__()
    return params


if __name__ == "__main__":
    config = Config()
    print(config.physics)
