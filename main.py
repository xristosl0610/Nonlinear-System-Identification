import numpy as np
import torch as T

from src import DATADIR
from src.config_data_class import Config, update_params_from_toml
from src.phi_sindy import apply_features, CoeffsDictionary, learn_sparse_model, print_learnt_equation
from src.physics import calculate_jonswap_excitation, apply_forcing, generate_data, contaminate_measurements

if __name__ == "__main__":

    CONFIG_OVERWRITE = DATADIR / 'config_overwrite.toml'

    params = Config()
    params = update_params_from_toml(params, CONFIG_OVERWRITE)

    calculate_jonswap_excitation(params)

    ts, x_denoised = generate_data(params)

    forcing = apply_forcing(ts, params)
    forcing_m = apply_forcing(0.5 * (ts[:-1] + ts[1:]), params)
    if len(forcing.shape) == 1:
        forcing = np.expand_dims(forcing, axis=1)
        forcing_m = np.expand_dims(forcing_m, axis=1)

    x = contaminate_measurements(params, x_denoised) if params.physics.noisy_measure_flag else x_denoised

    if params.hyperparams.scaling:
        params.physics.mus = T.tensor(np.mean(x, axis=0)).float().unsqueeze(0)
        params.physics.stds = T.tensor(np.std(x, axis=0)).float().unsqueeze(0)

    train_dset = T.tensor(x).float()
    times = T.tensor(ts).unsqueeze(1).float()

    no_of_terms = apply_features(train_dset[:2], times[:2], params=params).shape[1]

    coeffs = CoeffsDictionary(no_of_terms)

    coeffs, loss_track = learn_sparse_model(coeffs, train_dset, times, T.tensor(forcing).float(),
                                            T.tensor(forcing_m).float(), params, lr_reduction=10)
    learnt_coeffs = coeffs.linear.weight.detach().clone().t().numpy().astype(np.float64)

    params.equation = print_learnt_equation(learnt_coeffs, params)

    print(f"\n\n{'-' * len(params.equation)}\n{'The learnt equation is:'.center(len(params.equation))}\n{params.equation}\n{'-' * len(params.equation)}")

