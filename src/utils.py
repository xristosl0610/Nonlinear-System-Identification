from pathlib import Path
import toml
import numpy as np
from typing import Any
from csv import DictWriter
from src import RESULTSDIR
from src.config_data_class import Config


def to_column_vector(array: np.ndarray | list) -> np.ndarray:
    """
    Ensures the input array is a 2D column vector of shape (n, 1).

    If the input is a 1D array, it is reshaped into a column vector.
    If the input is already a 2D array, it is returned as-is unless it is a row vector,
    in which case it is transposed into a column vector.

    Args:
        array (np.ndarray | list): The input array, which can be a 1D or 2D array, or a list.

    Returns:
        np.ndarray: A 2D column vector of shape (n, 1).

    Example:
        >>> arr = [1, 2, 3]
        >>> to_column_vector(arr)
        array([[1],
               [2],
               [3]])

        >>> arr = np.array([[1, 2, 3]])
        >>> to_column_vector(arr)
        array([[1],
               [2],
               [3]])

        >>> arr = np.array([[1], [2], [3]])
        >>> to_column_vector(arr)
        array([[1],
               [2],
               [3]])
    """
    array = np.atleast_2d(array)
    return array.T if array.shape[0] == 1 else array


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


def setup_directories(child_path: Path) -> None:
    """
    Checks if the output directory exists.
    In case it does, stored data will be overwritten if the user confirms,
    otherwise, this new directory will be created.

    Args:
        child_path (Path): the path object of the output directory

    Returns:
        None
    """
    if child_path.is_dir():
        print("""
            ------------------------------------------------------------------------------ 
            WARNING: The directory already exists. You are about to overwrite some data! 
            ------------------------------------------------------------------------------
            """)

        confirm = input("Do you want to proceed with overwriting the directory? (y/n): ").strip().lower()
        if confirm not in ('y', 'yes'):
            print("Operation cancelled.")
            return

    child_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory {child_path} created successfully.")


def store_results(
    params: Config,
    loss: np.ndarray,
    coeffs: np.ndarray,
    run_name: str
) -> None:
    """
    Appends the hyperparameters and the derived solution to a CSV file with all stored results.
    Saves the loss values and derived coefficients to the specified directory.

    Args:
        params (Config): A dataclass or object containing the parameters of the run.
        loss (np.ndarray): Loss values recorded during training, stored for each epoch and batch.
        coeffs (np.ndarray): Derived coefficients after applying RK4SINDy.
        run_name (str): The directory for saving results of the current run.

    Returns:
        None
    """
    # TODO refine the function to follow the new Config parameter class, i.e. params.id is obsolete
    param_dict = params.__dict__
    run_dir = RESULTSDIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_file = run_dir / 'hyperparameters.csv'

    # with open(os.path.join(parent_dir, csv_file), "a", newline='') as file_object:
    with csv_file.open("a", newline='') as file_object:
        dict_writer_object = DictWriter(file_object, fieldnames=list(param_dict.keys()))

        if params.id == "1":
            dict_writer_object.writeheader()

        dict_writer_object.writerow(param_dict)

    np.save(run_dir / "losses.npy", loss)
    np.save(run_dir / "coeffs.npy", coeffs)


def union_dicts(default: dict[str, Any], overwrite: dict[str, Any]) -> dict[str, Any]:
    """
    Merges two dictionaries recursively. The `overwrite` dictionary updates the `default` dictionary.

    Args:
        default (Dict[str, Any]): The base dictionary to be updated.
        overwrite (Dict[str, Any]): The dictionary with values to overwrite or add to the base.

    Returns:
        Dict[str, Any]: The merged dictionary with updated values.

    Example:
        >>> default = {"a": 1, "b": {"c": 2}}
        >>> overwrite = {"b": {"c": 3, "d": 4}, "e": 5}
        >>> union_dicts(default, overwrite)
        {'a': 1, 'b': {'c': 3, 'd': 4}, 'e': 5}
    """
    for key in overwrite:
        if key in default:
            if isinstance(default[key], dict) and isinstance(overwrite[key], dict):
                union_dicts(default[key], overwrite[key])
            else:
                default[key] = overwrite[key]
        else:
            default[key] = overwrite[key]
    return default
