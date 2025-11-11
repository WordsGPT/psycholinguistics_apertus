import pandas as pd
import yaml


def read_yaml(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def read_txt(file_path: str) -> str:
    """"
    Read text files with UTF-8 only. If the file is not UTF-8 encoded,
    a UnicodeDecodeError will be raised for the caller to handle.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_config(config_type: str, name: str) -> dict:
    """
    Load configuration for a given experiment or fine-tuning model.

    :param config_type: Type of configuration to load ('experiments' or 'finetuning').
    :param name: Name of the experiment or fine-tuning model.
    :return: Configuration dictionary for the specified name.
    """
    config = read_yaml(file_path=f"config.yaml")

    if config_type not in config:
        print(f"Configuration type {config_type} not found in config.yaml.")
        exit()

    if name not in config[config_type]:
        print(
            f"{config_type.capitalize()} {name} not found in config.yaml. "
            f"Available {config_type}: {config[config_type].keys()}"
        )
        exit()

    config_args = config[config_type][name]
    return config_args


def read_column_as_list(file_path: str, column_name: str) -> list[int]:
    if file_path.endswith(".csv"):
        """"
        Read CSV assuming UTF-8 only. If the file is not UTF-8 encoded,
        pandas will raise an error and the caller can handle it.
        """
        df = pd.read_csv(file_path, encoding="utf-8")
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return df[column_name].tolist()