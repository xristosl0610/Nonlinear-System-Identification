import os
import numpy as np
from csv import DictWriter
from typing import Any
from src.config_data_class import Config


def update_params_from_toml(params: Config, toml_file: str) -> Config:
    """
    Update the parameters of a Config object from a TOML file.

    Args:
        params (Config): Config object to update.
        toml_file (str): Path to the TOML file containing parameter updates.

    Returns:
        Config: Updated Config object with parameters from the TOML file.
    """

    config_dict = toml.load(toml_file)
    for key, value in config_dict.items():
        if hasattr(params, key):
            for kkey, vvalue in value.items():
                if hasattr(getattr(params, key), kkey):
                    setattr(getattr(params, key), kkey, vvalue)
    params.physics.__post_init__()
    return params


def setup_directories(child_dir: str) -> None:
    """
    Checks if the output directory exists.
    In case it does, stored data will be overwritten if the user confirms,
    otherwise, this new directory will be created.

    Args:
        child_dir (str): the path of the output directory

    Returns:
        None
    """
    if os.path.isdir(child_dir):
        print("""
        ------------------------------------------------------------------------------
        WARNING, the directory already exists, you are about to overwrite some data!!!
        ------------------------------------------------------------------------------
        """, sep=os.linesep)

        confirm = input("Do you want to proceed with overwriting the directory? (y/n): ").strip().lower()
        if confirm not in ('y', 'yes'):
            print("Operation cancelled.")
            return

    os.makedirs(child_dir)
    print(f"Directory {child_dir} created successfully.")


def store_results(params, loss, coeffs, parent_dir, child_dir):
    """"
    A function that appends the hyperparameters and the derived solution to the .csv with all the stored results.
    It also saves the losses and the derived ksi coefficients

    Parameters
    ----------
    params : parameters dataclass
        the parameters of the run
    loss : numpy.ndarray
        the losses stored during every epoch of every training batch
    coeffs : numpy.ndarray
        the ksi coefficients derived after applying RK4SINDy
    parent_dir : str
        the path of the directory that contains info for all runs
    child_dir : str
        the path of the directory that will contain the results of the current run
    """
    param_dict = params.__dict__
    csv_file = 'hyperparameters.csv'

    with open(os.path.join(parent_dir, csv_file), "a", newline='') as file_object:
        dict_writer_object = DictWriter(file_object, fieldnames=list(param_dict.keys()))

        if params.id == "1":
            dict_writer_object.writeheader()

        dict_writer_object.writerow(param_dict)

    np.save(os.path.join(child_dir, "losses.npy"), loss)
    np.save(os.path.join(child_dir, "coeffs.npy"), coeffs)


def union_dicts(default, overwrite):
    """Make a union of overwrite dict and default dict recursively.
    Unable to do it with | because of nested dicts"""
    for key in overwrite:
        if key in default:
            if isinstance(default[key], dict) and isinstance(overwrite[key], dict):
                union_dicts(default[key], overwrite[key])
            else:
                default[key] = overwrite[key]
        else:
            default[key] = overwrite[key]
    return default
