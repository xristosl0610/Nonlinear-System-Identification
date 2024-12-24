import sys
import torch as T
import numpy as np
import torch_optimizer as optim_all
from src.config_data_class import Config
from src.physics import apply_known_physics
from typing import Tuple


class CoeffsDictionary(T.nn.Module):
    """
    A class for initializing, storing, and updating the ksi coefficients
    These coefficients are the linear weights of a neural network
    The class inherits from the torch.nn.Module
    """

    def __init__(self, n_combinations):
        super(CoeffsDictionary, self).__init__()
        self.linear = T.nn.Linear(n_combinations, 1, bias=False)
        # Setting the weights to zeros
        self.linear.weight = T.nn.Parameter(0 * self.linear.weight.clone().detach())

    def forward(self, x):
        return self.linear(x)


def apply_features(
    x: np.ndarray | T.Tensor,
    t: np.ndarray | T.Tensor,
    params: Config,
    torch_flag: bool = True
) -> np.ndarray | T.Tensor:
    """
    Applies the feature candidates to the given data.

    Args:
        x (np.ndarray | T.Tensor): A 2D array/tensor containing the displacement in the first column and the velocity in the second one.
        t (np.ndarray | T.Tensor): A 1D array/tensor containing the discrete time values.
        params (Config): The parameters of the run.
        torch_flag (bool, optional): If True, the type of `x`, `t`, and the return object is `torch.Tensor`.
                                     Otherwise, it is `numpy.ndarray`. Defaults to True.

    Returns:
        (np.ndarray | T.Tensor): A 2D array/tensor with the feature values applied to the input `x` and `t`. The first column represents velocity, and the second column represents acceleration.
    """
    module = T if torch_flag else np

    def compute_features(module, x, t, params):
        return module.column_stack(
            (
                *[module.cos(ph * t) for ph in params.features.cos_phases],
                *[module.sin(ph * t) for ph in params.features.sin_phases],
                *[module.sign(x[:, 0]), ] * params.features.x_sgn_flag,
                *[module.sign(x[:, 1]), ] * params.features.y_sgn_flag,
                *[module.log((module.abs(x[:, 1]) + params.physics.DR.eps) / params.physics.DR.V_star), ] * params.features.log_1,
                *[module.log(params.physics.DR.c + params.physics.DR.V_star / (module.abs(x[:, 1]) + params.physics.DR.eps)), ] * params.features.log_2,
                *[module.ones(size=(x.shape[0],))] * (params.features.poly_order >= 0),
                *[x[:, 0], x[:, 1]] * (params.features.poly_order >= 1),
                *[x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 1] ** 2] * (params.features.poly_order >= 2),
                *[x[:, 0] ** 3, x[:, 0] ** 2 * x[:, 1], x[:, 0] * x[:, 1] ** 2, x[:, 1] ** 3, ] * (params.features.poly_order >= 3),
                *[x[:, 0] ** 4, x[:, 0] ** 3 * x[:, 1], x[:, 0] ** 2 * x[:, 1] ** 2, x[:, 0] * x[:, 1] ** 3, x[:, 1] ** 4] * (params.features.poly_order >= 4),
            )
        )

    return compute_features(module, x, t, params)


def get_feature_names(params: Config):
    """
    A function that stores the assumed features as strings.

    Args:
        params (Config): The parameters of the run.

    Returns:
        list: The list containing the assumed features as strings.
    """
    return [*[f"cos({ph:.1f} t)" for ph in params.features.cos_phases],
            *[f"sin({ph:.1f} t)" for ph in params.features.sin_phases],
            *["sgn(x)",] * params.features.x_sgn_flag,
            *["sgn(y)",] * params.features.y_sgn_flag,
            *[f"ln((|y| + {params.physics.DR.eps:.1e}) / {params.physics.DR.V_star:.3f})", ] * params.features.log_1,
            *[f"ln({params.physics.DR.c:.3f} + {params.physics.DR.V_star:.3f} / (|y| + {params.physics.DR.eps:.1e}))",] * params.features.log_2,
            *["1",] * (params.features.poly_order >= 0),
            *["x", "y"] * (params.features.poly_order >= 1),
            *["x^2", "xy", "y^2"] * (params.features.poly_order >= 2),
            *["x^3", "x^2y", "xy^2", "y^3"] * (params.features.poly_order >= 3),
            *["x^4", "x^3y", "x^2y^2", "xy^3", "y^4"] * (params.features.poly_order >= 4),]


def print_learnt_equation(learnt_coeffs: np.ndarray[float], params: Config):
    """
    A function that combines the learnt coefficients and the assumed features
    to form the governing equation.

    Args:
        learnt_coeffs (numpy.ndarray): An array containing the learnt coefficients.
        params (Config): The parameters of the run.

    Returns:
        str: The governing equation.
    """
    feature_names = get_feature_names(params)

    string_list = [f'{"+" if coeff > 0 else ""}{coeff:.3f} {feat}' for coeff, feat in zip(np.squeeze(learnt_coeffs, axis=1), feature_names)
                   if np.abs(coeff) > 1e-5]

    equation = " ".join(string_list)

    if equation[0] == "+":
        equation = equation[1:]

    return equation


def apply_rk4_SparseId_known_forcing(x: T.tensor,
                                     coeffs: CoeffsDictionary,
                                     times: T.tensor,
                                     f_1: T.tensor,
                                     f_m: T.tensor,
                                     f_2: T.tensor,
                                     timesteps: T.tensor,
                                     params: Config) -> T.tensor:
    """
    A function that applies the fourth order Runge-Kutta scheme to the given data in order to derive the values for the following timestep. During this process, the approximate derivatives are used.

    Args:
        x (torch.Tensor): A 2D tensor containing the displacement in the first column and the velocity in the second one.
        coeffs (CoeffsDictionary): The neural network with the sought coefficients as its weights.
        times (torch.Tensor): A 1D tensor containing the discrete time values.
        f_1 (torch.Tensor): A 1D tensor containing the forcing at the current timestep.
        f_m (torch.Tensor): A 1D tensor containing the average of the current and next timestep forcing.
        f_2 (torch.Tensor): A 1D tensor containing the forcing at the next timestep.
        timesteps (torch.Tensor): A 1D tensor containing the discrete time intervals.
        params (Config): The parameters of the run.

    Returns:
        torch.Tensor: Predictions of both displacement and velocity for the next timesteps.
    """
    d1 = apply_features(x, times, params)
    k1 = T.column_stack((x[:, 1].unsqueeze(1),
                         apply_known_physics(x, params).unsqueeze(1)
                         + f_1 / params.physics.mass
                         - coeffs(d1) * params.physics.F0 / params.physics.mass * T.sign(x[:, 1]).unsqueeze(1)))

    k1 = T.where((T.abs(x[:, 1]).unsqueeze(1) <= params.physics.stick_tol), 0., k1)

    xtemp = x + 0.5 * timesteps * k1
    d2 = apply_features(xtemp, times + 0.5 * timesteps, params)
    k2 = T.column_stack((xtemp[:, 1].unsqueeze(1),
                         apply_known_physics(xtemp, params).unsqueeze(1)
                         + f_m / params.physics.mass
                         - coeffs(d2) * params.physics.F0 / params.physics.mass * T.sign(xtemp[:, 1]).unsqueeze(1)))

    k2 = T.where((T.abs(xtemp[:, 1]).unsqueeze(1) <= params.physics.stick_tol), 0., k2)

    xtemp = x + 0.5 * timesteps * k2
    d3 = apply_features(xtemp, times + 0.5 * timesteps, params)
    k3 = T.column_stack((xtemp[:, 1].unsqueeze(1),
                         apply_known_physics(xtemp, params).unsqueeze(1)
                         + f_m / params.physics.mass
                         - coeffs(d3) * params.physics.F0 / params.physics.mass * T.sign(xtemp[:, 1]).unsqueeze(1)))

    k3 = T.where((T.abs(xtemp[:, 1]).unsqueeze(1) <= params.physics.stick_tol), 0., k3)

    xtemp = x + timesteps * k3
    d4 = apply_features(xtemp, times + timesteps, params)
    k4 = T.column_stack((xtemp[:, 1].unsqueeze(1),
                         apply_known_physics(xtemp, params).unsqueeze(1)
                         + f_2 / params.physics.mass
                         - coeffs(d4) * params.physics.F0 / params.physics.mass * T.sign(xtemp[:, 1]).unsqueeze(1)))

    k4 = T.where((T.abs(xtemp[:, 1]).unsqueeze(1) <= params.physics.stick_tol), 0., k4)

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timesteps


def scale_torch(
    unscaled_tensor: T.Tensor,
    params: Config
) -> T.Tensor:
    """
    Applies standard scaling to a torch tensor.

    Args:
        unscaled_tensor (torch.Tensor): A 2D tensor to be scaled.
        params (Parameters): The parameters containing means and standard deviations used for scaling.

    Returns:
        torch.Tensor: The scaled tensor.
    """
    return (unscaled_tensor - params.physics.mus) / params.physics.stds


def learn_sparse_model(
    coeffs: CoeffsDictionary,
    train_set: T.Tensor,
    times: T.Tensor,
    f: T.Tensor,
    f_m: T.Tensor,
    params: Config,
    lr_reduction: int = 10
) -> (CoeffsDictionary, np.ndarray):
    """
    Calculates which ksi coefficients lead to optimal prediction. The updating of the coefficients
    is performed in a deep learning fashion.

    Args:
        coeffs (CoeffsDictionary): The neural network with the sought coefficients as its weights.
        train_set (torch.Tensor): A 2D tensor containing the displacement in the first column and the velocity in the second one.
        times (torch.Tensor): An 1D tensor containing the discrete time values.
        f (torch.Tensor): An 1D tensor containing the forcing at the discrete time values.
        f_m (torch.Tensor): An 1D tensor containing the average of the forcing i.e., forcing at in-between time instances.
        params (Parameters): The parameters of the run.
        lr_reduction (int, optional): The value that the learning rate is divided by in each training batch. Defaults to 10.

    Returns:
        Tuple[CoeffsDictionary, np.ndarray]:
            - coeffs (CoeffsDictionary): The neural network with the updated/learnt coefficients as its weights.
            - loss_track (np.ndarray): A 2D array containing the loss for each training batch (row), for each epoch (column).
    """
    # Define optimizer
    opt_func = optim_all.RAdam(
        coeffs.parameters(), lr=params.hyperparams.lr, weight_decay=params.hyperparams.weightdecay
    )
    # Define loss function
    criteria = T.nn.MSELoss()
    # pre-allocate memory for loss_fuction
    loss_track = np.zeros((params.hyperparams.num_iter, params.hyperparams.num_epochs))

    # Training
    for p in range(params.hyperparams.num_iter):
        for g in range(params.hyperparams.num_epochs):
            coeffs.train()

            opt_func.zero_grad()

            loss_new = T.autograd.Variable(T.tensor([0.0], requires_grad=True))
            weights = 2 ** (-0.5 * T.linspace(0, 0, 1))

            timesteps_i = T.tensor(np.diff(times, axis=0)).float()

            # One forward step predictions
            y_pred = apply_rk4_SparseId_known_forcing(train_set[:-1], coeffs, times[:-1], f[:-1], f_m, f[1:], timesteps=timesteps_i,
                                                      params=params)

            # One backward step predictions
            y_pred_back = apply_rk4_SparseId_known_forcing(train_set[1:], coeffs, times[1:], f[1:], f_m, f[:-1], timesteps=-timesteps_i,
                                                           params=params)

            if params.hyperparams.scaling:
                y_pred_scaled = scale_torch(y_pred, params)
                y_pred_back_scaled = scale_torch(y_pred_back, params)
                train_set_scaled = scale_torch(train_set, params)

                loss_new += criteria(y_pred_scaled, train_set_scaled[1:]) + weights[0] * criteria(
                    y_pred_back_scaled, train_set_scaled[:-1]
                )
            else:
                loss_new += criteria(y_pred, train_set[1:]) + weights[0] * criteria(
                    y_pred_back, train_set[:-1]
                )

            loss_track[p, g] += loss_new.item()
            loss_new.backward()
            opt_func.step()

            sys.stdout.write("\r [Iter %d/%d] [Epoch %d/%d] [Training loss: %.2e] [Learning rate: %.2e]"
                             % (p + 1, params.hyperparams.num_iter, g + 1,
                                params.hyperparams.num_epochs, loss_track[p, g], opt_func.param_groups[0]["lr"],))

        # Removing the coefficients smaller than tol and set gradients w.r.t. them to zero
        # so that they will not be updated in the iterations
        Ws = coeffs.linear.weight.detach().clone()
        Mask_Ws = (Ws.abs() > params.hyperparams.tol_coeffs).type(T.float)
        coeffs.linear.weight = T.nn.Parameter(Ws * Mask_Ws)

        coeffs.linear.weight.register_hook(lambda grad: grad.mul_(Mask_Ws))
        new_lr = opt_func.param_groups[0]["lr"] / lr_reduction
        opt_func = optim_all.RAdam(coeffs.parameters(), lr=new_lr, weight_decay=params.hyperparams.weightdecay)

    return coeffs, loss_track

