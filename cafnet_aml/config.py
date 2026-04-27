# 加载、整合和生成模型训练所需的配置参数 支撑模型训练和超参数调优的流程
import json
import os
from pathlib import Path
import random
from easydict import EasyDict as edict


def _find_key_case_insensitive(container: dict, name: str):
    """Return the actual key in container matching name (case-insensitive)."""
    lname = name.lower()
    for k in container.keys():
        if k.lower() == lname:
            return k
    return None


def get_config_regression(
    model_name: str, dataset_name: str, config_file: str = ""
) -> dict:
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.

    Returns:
        config (dict): config of the given dataset and model
    """
    # 统一小写并容错大小写/空格导致的 KeyError
    dataset_name_norm = dataset_name.strip().lower()
    model_name_norm = model_name.strip().lower()
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)

    model_key = _find_key_case_insensitive(config_all, model_name_norm)
    if model_key is None:
        raise KeyError(f"Model '{model_name}' not found. Available: {list(config_all.keys())}")

    ds_params = config_all[model_key].get('datasetParams', {})
    ds_key_model = _find_key_case_insensitive(ds_params, dataset_name_norm)
    if ds_key_model is None:
        raise KeyError(f"Dataset '{dataset_name}' not found for model '{model_key}'. Available: {list(ds_params.keys())}")

    ds_common = config_all['datasetCommonParams']
    ds_key_common = _find_key_case_insensitive(ds_common, dataset_name_norm)
    if ds_key_common is None:
        raise KeyError(f"Dataset '{dataset_name}' not found in datasetCommonParams. Available: {list(ds_common.keys())}")

    model_common_args = config_all[model_key]['commonParams']
    model_dataset_args = ds_params[ds_key_model]
    dataset_args = ds_common[ds_key_common]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    if model_common_args['need_data_aligned'] and 'aligned' in dataset_args:
        dataset_args = dataset_args['aligned']
    else:
        dataset_args = dataset_args['unaligned']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_tune(
    model_name: str, dataset_name: str, config_file: str = "",
    random_choice: bool = True
) -> dict:
    """
    Get the tuning config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.
        random_choice: If True, will randomly choose a config from the list of configs.

    Returns:
        config (dict): config of the given dataset and model
    """
    dataset_name = dataset_name.lower()
    model_name = model_name.lower()
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name] if 'datasetParams' in config_all[model_name] else {}
    model_debug_args = config_all[model_name]['debugParams']
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    # random choice of args
    if random_choice:
        for item in model_debug_args['d_paras']:
            if type(model_debug_args[item]) == list:
                model_debug_args[item] = random.choice(model_debug_args[item])
            elif type(model_debug_args[item]) == dict: # nested params, 2 levels max
                for k, v in model_debug_args[item].items():
                    model_debug_args[item][k] = random.choice(v)

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    

    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_all(config_file: str) -> dict:
    """
    Get all default configs. This function is used to export default config file. 
    If you want to get config for a specific model, use "get_config_regression" or "get_config_tune" instead.

    Parameters:
        config_file: "regression" or "tune"
    
    Returns:
        config: all default configs
    """
    if config_file == "regression":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    elif config_file == "tune":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    else:
        raise ValueError("config_file should be 'regression' or 'tune'")
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    return edict(config_all)

def get_citations() -> dict:
    """
    Get paper titles and citations for models and datasets.

    Returns:
        cites (dict): {
            models: {
                tfn: {
                    title: "xxx",
                    paper_url: "xxx",
                    citation: "xxx",
                    description: "xxx"
                },
                ...
            },
            datasets: {
                ...
            },
        }
    """
    # TODO: add citations
    config_file = Path(__file__).parent / "config" / "citations.json"
    with open(config_file, 'r') as f:
        cites = json.load(f)
    return cites
