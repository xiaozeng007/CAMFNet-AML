import argparse
import errno
import gc
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import numbers
from typing import Union
import torch
from easydict import EasyDict as edict

# 导入 CAFNet-AML 框架核心模块（从包内导入，便于仓库结构清晰）
from cafnet_aml.config import get_config_regression, get_config_tune
from cafnet_aml.data_loader import MMDataLoader
from cafnet_aml.transformers_compat import apply_transformers_hub_compat

apply_transformers_hub_compat()

from cafnet_aml.models import AMIO
from cafnet_aml.trains import ATIO
from cafnet_aml.utils import assign_gpu, count_parameters, setup_seed

# 设置CUDA环境变量，确保GPU设备顺序一致性和结果可重现性
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2" # 这对结果可重现性至关重要

# 定义支持的模型列表
SUPPORTED_MODELS = ['tetfn', 'camfn', 'camfnet_aml']
# 定义支持的数据集列表
SUPPORTED_DATASETS = ['mosi', 'mosei', 'sims', 'mustard', 'mustardpp']
MODEL_ALIASES = {
    'camfnet-aml': 'camfn',
    'camfnet_aml': 'camfn',
}

# 创建日志记录器
logger = logging.getLogger('MMSA')

CLASSIFICATION_DATASETS = {'mustard', 'mustardpp'}
REGRESSION_ONLY_MODELS = {'tetfn'}


def _resolve_train_mode(dataset_name: str, model_name: str = "") -> str:
    """
    Return train mode ('classification' or 'regression') taking both dataset
    characteristics and model constraints into account.
    """
    model = (model_name or '').lower()
    if model in REGRESSION_ONLY_MODELS:
        return 'regression'

    name = (dataset_name or '').lower()
    if name in CLASSIFICATION_DATASETS:
        return 'classification'
    return 'regression'


def _canonicalize_model_name(model_name: str) -> str:
    if not model_name:
        return model_name
    key = model_name.lower()
    return MODEL_ALIASES.get(key, key)


def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    """
    设置日志记录器，同时支持文件输出和控制台输出
    
    Args:
        log_dir: 日志文件保存目录
        model_name: 模型名称
        dataset_name: 数据集名称  
        verbose_level: 详细程度级别 (0=ERROR, 1=INFO, 2=DEBUG)
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志文件路径
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA') 
    logger.setLevel(logging.DEBUG)

    # 文件处理器：将所有日志写入文件
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # 控制台处理器：根据 verbose_level 控制控制台输出级别
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def MMSA_run(
    model_name: str, dataset_name: str, config_file: str = None,
    config: dict = None, seeds: list = [], is_tune: bool = False,
    tune_times: int = 50, custom_feature: str = None, feature_T: str = None, 
    feature_A: str = None, feature_V: str = None, gpu_ids: list = [0],
    num_workers: int = 4, verbose_level: int = 1,
    model_save_dir: str = Path.cwd() / "saved_models",  # 当前目录下的 saved_models
    res_save_dir: str = Path.cwd() / "results",          # 当前目录下的 results
    log_dir: str = Path.cwd() / "logs",                  # 当前目录下的 logs
    disable_text: bool = False,
    disable_audio: bool = False,
    disable_video: bool = False,
):
   
    # 统一大小写 保证模型名和数据集名匹配时不会出错
    model_name = _canonicalize_model_name(model_name)
    dataset_name = dataset_name.lower()
    if os.name == 'nt' and num_workers > 0:
        num_workers = 0
    
    # 如果命令行传入了 -c 参数（配置文件路径） 就使用自定义配置
    if config_file is not None and config_file != '':
        config_file = Path(config_file)
    else:
        # 使用当前工作目录作为基础路径
        base_path = Path.cwd()
        if is_tune:
            config_file = base_path / "config" / "config_tune.json"
        else:
            config_file = base_path / "config" / "config_regression.json"
            
    # 确认配置文件存在 否则报错
    if not config_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
        
    # 如果未指定保存路径，则在当前文件夹下自动创建
    if model_save_dir is None:  # 使用默认模型保存目录
        model_save_dir = Path.cwd() / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if res_save_dir is None:  # 使用默认结果保存目录
        res_save_dir = Path.cwd() / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)

    if log_dir is None:  # 使用默认日志保存目录
        log_dir = Path.cwd() / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 默认使用 5 个随机种子重复实验 提高结果稳定性
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    def _apply_modality_flags(target_args):
        if disable_text:
            target_args['use_text'] = False
        elif 'use_text' not in target_args:
            target_args['use_text'] = True
        if disable_audio:
            target_args['use_audio'] = False
        elif 'use_audio' not in target_args:
            target_args['use_audio'] = True
        if disable_video:
            target_args['use_video'] = False
        elif 'use_video' not in target_args:
            target_args['use_video'] = True
	
	# 输出程序启动信息
    logger.info("======================================== Program Start ========================================")
    
    # 超参数调优模式 is_tune=True
    if is_tune:
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        _apply_modality_flags(initial_args)
        # 指定模型保存路径
        initial_args['model_save_path'] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}.pth"
        # 自动选择 GPU
        initial_args['device'] = assign_gpu(gpu_ids)
        initial_args['train_mode'] = _resolve_train_mode(dataset_name, model_name) # backward compatibility. TODO: remove all train_mode in code
        initial_args['custom_feature'] = custom_feature
        initial_args['feature_T'] = feature_T
        initial_args['feature_A'] = feature_A
        initial_args['feature_V'] = feature_V

        # 解决 PyTorch 在多 GPU 情况下 RNN 强制使用 GPU0 的问题（仅在使用 CUDA 时设置）
        try:
            dev_str = str(initial_args['device'])
            if torch.cuda.is_available() and dev_str.startswith('cuda'):
                torch.cuda.set_device(initial_args['device'])
        except Exception:
            pass
		
		# 所有调参结果保存为 CSV
        res_save_dir = Path(res_save_dir) / "tune"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        has_debuged = [] 	# 保存已尝试的参数组合
        csv_file = res_save_dir / f"{dataset_name}-{model_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in initial_args['d_paras']])
		
		# 每次循环随机生成一组新参数 合并成当前实验配置
        for i in range(tune_times):
            args = edict(**initial_args)
            random.seed(time.time())
            new_args = get_config_tune(model_name, dataset_name, config_file)
            args.update(new_args)
            
            if config:
                if config.get('model_name'):
                    assert(config['model_name'] == args['model_name'])
                args.update(config)
            _apply_modality_flags(args)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Tuning [{i + 1}/{tune_times}] {'-'*30}")
            logger.info(f"Args: {args}")
            # 检查此参数是否已运行
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info(f"This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
                
            # 实际训练函数
            setup_seed(seeds[0])
            # _run() 是真正执行训练和测试的函数 它会完成数据加载、模型构建、训练、验证、测试
            # 并返回性能指标结果（如 F1、Acc、MAE 等）
            result = _run(args, num_workers, is_tune)
            has_debuged.append(cur_param)
            # 保存结果为 csv
            if Path(csv_file).is_file():
                df2 = pd.read_csv(csv_file)
            else:
                df2 = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in result.keys()])
            res = [args[c] for c in args.d_paras]
            for col in result.keys():
                value = result[col]
                res.append(value)
            df2.loc[len(df2)] = res
            df2.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")
            
    # 启用正常训练模式        
    else:
        # 加载普通任务配置（回归/分类）
        args = get_config_regression(model_name, dataset_name, config_file)
        
        # 指定模型、GPU、特征路径
        base_model_path = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
        args['model_save_path'] = base_model_path
        args['device'] = assign_gpu(gpu_ids)
        args['train_mode'] = _resolve_train_mode(dataset_name, model_name) # backward compatibility. TODO: remove all train_mode in code
        args['custom_feature'] = custom_feature
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        if config: # 覆盖某些参数
            if config.get('model_name'):
                assert(config['model_name'] == args['model_name'])
            args.update(config)
        _apply_modality_flags(args)

        # 解决 PyTorch 在多 GPU 情况下 RNN 强制使用 GPU0 的问题（仅在使用 CUDA 时设置）
        try:
            dev_str = str(args['device'])
            if torch.cuda.is_available() and dev_str.startswith('cuda'):
                torch.cuda.set_device(args['device'])
        except Exception:
            pass
		
        # 输出当前配置
        logger.info("Running with args:")
        logger.info(args)
        logger.info(f"Seeds: {seeds}")
        # 创建结果保存路径
        res_save_dir = Path(res_save_dir) / "normal"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        model_results = []
        saved_model_paths = []
        base_name = f"{args['model_name']}-{args['dataset_name']}"
        # ���ѵ������ͬ���ӣ�
        for i, seed in enumerate(seeds):
            setup_seed(seed)
            args['cur_seed'] = i + 1
            seed_model_path = Path(model_save_dir) / f"{base_name}-seed{seed}.pth"
            args['model_save_path'] = seed_model_path
            logger.info(f"{'-'*30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-'*30}")
            # ʵ������ ����ʵ�ʽ���ѵ���Ͳ��Եĺ��� _run() 
            result = _run(args, num_workers, is_tune)
            logger.info(f"Result for seed {seed}: {result}")
            model_results.append(result)
            saved_model_paths.append(seed_model_path)
        key_metric = args.get('KeyEval', 'Accuracy')
        metric_values = [res.get(key_metric, 0.0) for res in model_results]
        if metric_values:
            best_idx = int(np.argmax(metric_values))
            best_model_path = saved_model_paths[best_idx]
            if best_model_path.exists():
                shutil.copyfile(best_model_path, base_model_path)
                args['model_save_path'] = base_model_path
                args['model_save_path'] = base_model_path
        criterions = list(model_results[0].keys())
        # 保存为 csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        # 保存结果
        res_row = {"Model": model_name}
        for c in criterions:
            values = [r[c] for r in model_results]
            if all(isinstance(v, numbers.Number) for v in values):
                mean = round(np.mean(values) * 100, 2)
                std = round(np.std(values) * 100, 2)
                res_row[c] = (mean, std)
            else:
                serialized = json.dumps(values if len(values) > 1 else values[0], ensure_ascii=False)
                res_row[c] = serialized
        df = pd.concat([df, pd.DataFrame([res_row])], ignore_index=True)
        ensemble_metrics = _ensemble_test(args, saved_model_paths, num_workers)
        if ensemble_metrics:
            logger.info(f"Ensemble ({len(saved_model_paths)} seeds) >> {ensemble_metrics}")
            ensemble_row = {"Model": f"{model_name}_ensemble"}
            for c in criterions:
                value = ensemble_metrics.get(c)
                if isinstance(value, numbers.Number):
                    ensemble_row[c] = round(value * 100, 2)
            df = pd.concat([df, pd.DataFrame([ensemble_row])], ignore_index=True)
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False):
    # load data and models
    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args).to(args['device'])

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # TODO: use multiple gpus
    # if using_cuda and len(args.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model,
    #                                   device_ids=args.gpu_ids,
    #                                   output_device=args.gpu_ids[0])
    trainer = ATIO().getTrain(args)
    # do train
    # epoch_results = trainer.do_train(model, dataloader)
    epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
    # load trained model & do test
    assert Path(args['model_save_path']).exists()
    model.load_state_dict(torch.load(args['model_save_path']))
    model.to(args['device'])
    if from_sena:
        final_results = {}
        final_results['train'] = trainer.do_test(model, dataloader['train'], mode="TRAIN", return_sample_results=True)
        final_results['valid'] = trainer.do_test(model, dataloader['valid'], mode="VALID", return_sample_results=True)
        final_results['test'] = trainer.do_test(model, dataloader['test'], mode="TEST", return_sample_results=True)
    elif is_tune:
        # use valid set to tune hyper parameters
        # results = trainer.do_test(model, dataloader['valid'], mode="VALID")
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        # delete saved model (Python 3.7 compatible)
        try:
            Path(args['model_save_path']).unlink()
        except FileNotFoundError:
            pass
    else:
        results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    return {"epoch_results": epoch_results, 'final_results': final_results} if from_sena else results


def _ensemble_test(base_args, model_paths, num_workers):
    model_paths = [Path(p) for p in model_paths if Path(p).exists()]
    if len(model_paths) < 2:
        return None
    topk = base_args.get('ensemble_topk', 0)
    key_metric = base_args.get('KeyEval', 'Accuracy')
    if topk and topk > 0:
        filtered = []
        for path in model_paths:
            log_file = Path(str(path).replace('.pth', '.log'))
            score = 0.0
            if log_file.exists():
                try:
                    with log_file.open('r', encoding='utf-8') as f:
                        for line in f.readlines()[::-1]:
                            if 'VAL' in line and key_metric in line:
                                parts = line.split()
                                for idx, token in enumerate(parts):
                                    if key_metric in token:
                                        value = parts[idx + 1].strip(',')
                                        score = float(value)
                                        break
                                break
                except Exception:
                    pass
            filtered.append((score, path))
        filtered.sort(reverse=True, key=lambda x: x[0])
        model_paths = [p for _, p in filtered[:topk]] or model_paths[:topk]
    eval_args = edict(dict(base_args))
    dataloader = MMDataLoader(eval_args, num_workers)
    trainer = ATIO().getTrain(eval_args)
    if hasattr(trainer, '_maybe_init_sarcasm_criterion'):
        try:
            trainer._maybe_init_sarcasm_criterion(dataloader)
        except Exception:
            pass
    device = eval_args['device']
    preds_list = []
    labels = None
    for path in model_paths:
        # 防御性处理: seq_lens 可能是单个整数，AlignSubNet 需要三个长度
        if 'seq_lens' in eval_args and isinstance(eval_args['seq_lens'], (int, float)):
            eval_args['seq_lens'] = [int(eval_args['seq_lens'])] * 3
        elif 'seq_lens' in eval_args and hasattr(eval_args['seq_lens'], '__len__') and len(eval_args['seq_lens']) == 1:
            eval_args['seq_lens'] = list(eval_args['seq_lens']) * 3
        model = AMIO(eval_args).to(device)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        eval_results = trainer.do_test(model, dataloader['test'], mode="TEST", return_sample_results=True)
        preds_raw = eval_results.get('Preds')
        if preds_raw is None:
            del model
            torch.cuda.empty_cache()
            continue
        preds_arr = np.array(preds_raw)
        # Skip if still contains non-numeric values
        if preds_arr.dtype == object:
            try:
                preds_arr = preds_arr.astype(float)
            except Exception:
                del model
                torch.cuda.empty_cache()
                continue
        preds_list.append(preds_arr)
        if labels is None:
            lbl_raw = eval_results.get('Labels')
            if lbl_raw is not None:
                labels = np.array(lbl_raw)
        del model
        torch.cuda.empty_cache()
    if not preds_list or labels is None:
        return None
    # 过滤 None，保证堆叠有效
    preds_list = [p for p in preds_list if p is not None]
    if not preds_list:
        return None
    stacked = np.stack(preds_list, axis=0)
    avg_preds = stacked.mean(axis=0)
    # Handle regression (pred dim=1) vs classification (pred dim>=2)
    if avg_preds.ndim == 1 or (avg_preds.ndim == 2 and avg_preds.shape[1] <= 1):
        pred_tensor = torch.from_numpy(avg_preds).float().view(-1, 1)
        label_tensor = torch.from_numpy(labels).float().view(-1, 1)
        metrics = trainer.metrics(pred_tensor, label_tensor)
        try:
            loss = torch.nn.functional.l1_loss(pred_tensor, label_tensor).item()
        except Exception:
            loss = 0.0
        metrics['Loss'] = float(loss)
        return metrics

    label_tensor = torch.from_numpy(labels).long()
    grid = base_args.get('calibration_grid', 21)
    thresholds = np.linspace(0.3, 0.7, grid)
    best_metrics = None
    best_thresh = 0.5
    for thresh in thresholds:
        pred_tensor = torch.from_numpy(avg_preds).float().clone()
        prob = torch.sigmoid(pred_tensor[:, 1] - pred_tensor[:, 0])
        two_class = torch.stack([1 - prob, prob], dim=-1)
        metrics = trainer.metrics(two_class, label_tensor)
        if best_metrics is None or metrics[base_args['KeyEval']] > best_metrics[base_args['KeyEval']]:
            best_metrics = metrics
            best_thresh = thresh
    prob = avg_preds[:, 1]
    prob = np.clip(prob, 1e-8, 1 - 1e-8)
    bce = -(label_tensor.numpy() * np.log(prob) + (1 - label_tensor.numpy()) * np.log(1 - prob))
    best_metrics['Loss'] = float(np.mean(bce))
    best_metrics['BestThresh'] = best_thresh
    return best_metrics


def MMSA_test(
    config: Union[dict, str],
    weights_path: str,
    feature_path: str, 
    # seeds: list = [], 
    gpu_id: int = 0, 
):
    """Test MSA models on a single sample.

    Load weights and configs of a saved model, input pre-extracted
    features of a video, then get sentiment prediction results.

    Args:
        model_name: Name of MSA model.
        config: Config dict or path to config file. 
        weights_path: Pkl file path of saved model weights.
        feature_path: Pkl file path of pre-extracted features.
        gpu_id: Specify which gpu to use. Use cpu if value < 0.
    """
    if type(config) == str or type(config) == Path:
        config = Path(config)
        with open(config, 'r') as f:
            args = json.load(f)
    elif type(config) == dict or type(config) == edict:
        args = config
    else:
        raise ValueError(f"'config' should be string or dict, not {type(config)}")
    dataset_name = args.get('dataset_name', '')
    args['model_name'] = _canonicalize_model_name(args.get('model_name', ''))
    args['train_mode'] = _resolve_train_mode(dataset_name, args.get('model_name', ''))

    if gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')
    args['device'] = device
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)
    args['feature_dims'] = [feature['text'].shape[1], feature['audio'].shape[1], feature['vision'].shape[1]]
    args['seq_lens'] = [feature['text'].shape[0], feature['audio'].shape[0], feature['vision'].shape[0]]
    model = AMIO(args)
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        if args.get('use_bert', None):
            text = feature['text_bert']
            if type(text) == np.ndarray:
                text = torch.from_numpy(text).float()
        else:
            text = feature['text']
            if type(text) == np.ndarray:
                text = torch.from_numpy(text).float()
        audio = feature['audio']
        if type(audio) == np.ndarray:
            audio = torch.from_numpy(audio).float()
        vision = feature['vision']
        if type(vision) == np.ndarray:
            vision = torch.from_numpy(vision).float()
        text = text.unsqueeze(0).to(device)
        audio = audio.unsqueeze(0).to(device)
        vision = vision.unsqueeze(0).to(device)
        if args.get('need_normalized', None):
            audio = torch.mean(audio, dim=1, keepdims=True)
            vision = torch.mean(vision, dim=1, keepdims=True)
        # TODO: write a do_single_test function for each model in trains
        if args['model_name'] == 'self_mm' or args['model_name'] == 'mmim':
            output = model(text, (audio, torch.tensor(audio.shape[1]).unsqueeze(0)), (vision, torch.tensor(vision.shape[1]).unsqueeze(0)))
        elif args['model_name'] == 'tfr_net':
            input_mask = torch.tensor(feature['text_bert'][1]).unsqueeze(0).to(device)
            output, _ = model((text, text, None), (audio, audio, input_mask, None), (vision, vision, input_mask, None))
        else:
            output = model(text, audio, vision)
        if type(output) == dict:
            output = output['M']
    return output.cpu().detach().numpy()[0][0]
        

SENA_ENABLED = True
try:
    from datetime import datetime
    from multiprocessing import Queue

    import mysql.connector
    from sklearn.decomposition import PCA
except ImportError:
    logger.debug("SENA_run is not loaded due to missing dependencies. Ignore this if you are not using M-SENA Platform.")
    SENA_ENABLED = False

if SENA_ENABLED:
    def SENA_run(
        task_id: int, progress_q: Queue, db_url: str,
        parameters: str, model_name: str, dataset_name: str,
        is_tune: bool, tune_times: int,
        feature_T: str, feature_A: str, feature_V: str,
        model_save_dir: str, res_save_dir: str, log_dir: str,
        gpu_ids: list, num_workers: int, seed: int, desc: str
    ) -> None:
        """
        Run M-SENA tasks. Should only be called from M-SENA Platform.
        Run only one seed at a time.

        Parameters:
            task_id (int): Task id.
            progress_q (multiprocessing.Queue): Used to communicate with M-SENA Platform.
            db_url (str): Database url.
            parameters (str): Training parameters in JSON.
            model_name (str): Model name.
            dataset_name (str): Dataset name.
            is_tune (bool): Whether to tune hyper parameters.
            tune_times (int): Number of times to tune hyper parameters.
            feature_T (str): Path to text feature file.
            feature_A (str): Path to audio feature file.
            feature_V (str): Path to video feature file.
            model_save_dir (str): Path to save trained model.
            res_save_dir (str): Path to save training results.
            log_dir (str): Path to save training logs.
            gpu_ids (list): GPU ids.
            num_workers (int): Number of workers.
            seed (int): Only one seed.
            desc (str): Description.
        """
        # TODO: add progress report
        cursor = None
        try:
            logger = logging.getLogger('app')
            logger.info(f"M-SENA Task {task_id} started.")
            time.sleep(1) # make sure task status is committed by the parent process
            # get db parameters
            db_params = db_url.split('//')[1].split('@')[0].split(':')
            db_user = db_params[0]
            db_pass = db_params[1]
            db_params = db_url.split('//')[1].split('@')[1].split('/')
            db_host = db_params[0]
            db_name = db_params[1]
            # connect to db
            db = mysql.connector.connect(
                user=db_user, password=db_pass, host=db_host, database=db_name
            )
            cursor = db.cursor()
            # load training parameters
            if parameters == "": # use default config file
                if is_tune: # TODO
                    config_file = Path(__file__).parent / "config" / "config_tune.json"
                    args = get_config_tune(model_name, dataset_name, config_file)
                else:
                    config_file = Path(__file__).parent / "config" / "config_regression.json"
                    args = get_config_regression(model_name, dataset_name, config_file)
            else: # load from JSON
                args = json.loads(parameters)
                args['model_name'] = model_name
                args['dataset_name'] = dataset_name
            args['feature_T'] = feature_T
            args['feature_A'] = feature_A
            args['feature_V'] = feature_V
            # determine feature_dims
            if args['feature_T']:
                with open(args['feature_T'], 'rb') as f:
                    data_T = pickle.load(f)
                if 'use_bert' in args and args['use_bert']:
                    args['feature_dims'][0] = 768
                else:
                    args['feature_dims'][0] = data_T['valid']['text'].shape[2]
            if args['feature_A']:
                with open(args['feature_A'], 'rb') as f:
                    data_A = pickle.load(f)
                args['feature_dims'][1] = data_A['valid']['audio'].shape[2]
            if args['feature_V']:
                with open(args['feature_V'], 'rb') as f:
                    data_V = pickle.load(f)
                args['feature_dims'][2] = data_V['valid']['vision'].shape[2]
            args['device'] = assign_gpu(gpu_ids)
            args['cur_seed'] = 1 # the _run function need this to print log
            args['train_mode'] = _resolve_train_mode(dataset_name, model_name) # backward compatibility. TODO: remove all train_mode in code
            args = edict(args)
            # create folders
            Path(model_save_dir).mkdir(parents=True, exist_ok=True)
            Path(res_save_dir).mkdir(parents=True, exist_ok=True)
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            # create db record
            args_dump = args.copy()
            args_dump['device'] = str(args_dump['device'])
            custom_feature = False if (feature_A == "" and feature_V == "" and feature_T == "") else True
            cursor.execute(
                """
                    INSERT INTO Result (dataset_name, model_name, is_tune, custom_feature, created_at,
                     args, save_model_path, loss_value, accuracy, f1, mae, corr, description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (dataset_name, model_name, is_tune, custom_feature, datetime.now(), json.dumps(args_dump), '', 0, 0, 0, 0, 0, desc)
            )
            result_id = cursor.lastrowid
            # result_id is allocated now, so model_save_path can be determined
            args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-{result_id}.pth"
            cursor.execute(
                "UPDATE Result SET save_model_path = %s WHERE result_id = %s", (str(args['model_save_path']), result_id)
            )
            # start training
            try:
                dev_str = str(args['device'])
                if torch.cuda.is_available() and dev_str.startswith('cuda'):
                    torch.cuda.set_device(args['device'])
                logger.info(f"Running with seed {seed}:")
                logger.info(f"Arg<PATH_REDACTED>")
                setup_seed(seed)
                # actual training
                results_dict = _run(args, num_workers, is_tune, from_sena=True)
                # db operations
                sample_dict = {}
                cursor2 = db.cursor(named_tuple=True)
                cursor2.execute("SELECT * FROM Dsample WHERE dataset_name=%s", (dataset_name,))
                samples = cursor2.fetchall()
                for sample in samples:
                    key = sample.video_id + '$_$' + sample.clip_id
                    sample_dict[key] = (sample.sample_id, sample.annotation)
                # update final results of test set
                cursor.execute(
                    """UPDATE Result SET loss_value = %s, accuracy = %s, f1 = %s,
                    mae = %s, corr = %s WHERE result_id = %s""",
                    (
                        results_dict['final_results']['test']['Loss'],
                        results_dict['final_results']['test']['Non0_acc_2'],
                        results_dict['final_results']['test']['Non0_F1_score'],
                        results_dict['final_results']['test']['MAE'],
                        results_dict['final_results']['test']['Corr'],
                        result_id
                    )
                )
                # save features
                if not is_tune:
                    logger.info("Running feature PCA ...")
                    features = {
                        k: results_dict['final_results'][k]['Features']
                        for k in ['train', 'valid', 'test']
                    }
                    all_features = {}
                    for select_modes in [['train', 'valid', 'test'], ['train', 'valid'], ['train', 'test'], \
                                        ['valid', 'test'], ['train'], ['valid'], ['test']]:
                        # create label index dict
                        # {"Negative": [1,2,5,...], "Positive": [...], ...}
                        cur_labels = []
                        for mode in select_modes:
                            cur_labels.extend(results_dict['final_results'][mode]['Labels'])
                        cur_labels = np.array(cur_labels)
                        label_index_dict = {}
                        label_index_dict['Negative'] = np.where(cur_labels < 0)[0].tolist()
                        label_index_dict['Neutral'] = np.where(cur_labels == 0)[0].tolist()
                        label_index_dict['Positive'] = np.where(cur_labels > 0)[0].tolist()
                        # handle features
                        cur_mode_features_2d = {}
                        cur_mode_features_3d = {}
                        for name in ['Feature_t', 'Feature_a', 'Feature_v', 'Feature_f']: 
                            cur_features = []
                            for mode in select_modes:
                                if name in features[mode]:
                                    cur_features.append(features[mode][name])
                            if cur_features != []:
                                cur_features = np.concatenate(cur_features, axis=0)
                            # PCA analysis
                            if len(cur_features) != 0:
                                pca=PCA(n_components=3, whiten=True)
                                features_3d = pca.fit_transform(cur_features)
                                # split by labels
                                cur_mode_features_3d[name] = {}
                                for k, v in label_index_dict.items():
                                    cur_mode_features_3d[name][k] = features_3d[v].tolist()
                                # PCA analysis
                                pca=PCA(n_components=2, whiten=True)
                                features_2d = pca.fit_transform(cur_features)
                                # split by labels
                                cur_mode_features_2d[name] = {}
                                for k, v in label_index_dict.items():
                                    cur_mode_features_2d[name][k] = features_2d[v].tolist()
                        all_features['-'.join(select_modes)] = {'2D': cur_mode_features_2d, '3D': cur_mode_features_3d}
                    # save features
                    save_path = args.model_save_path.parent / (args.model_save_path.stem + '.pkl')
                    with open(save_path, 'wb') as fp:
                        pickle.dump(all_features, fp, protocol = 4)
                    logger.info(f'Feature saved at {save_path}.')
                # update sample results
                for mode in ['train', 'valid', 'test']:
                    final_results = results_dict['final_results'][mode]
                    for i, cur_id in enumerate(final_results["Ids"]):
                        cursor.execute(
                            """ INSERT INTO SResults (result_id, sample_id, label_value, predict_value, predict_value_r)
                            VALUES (%s, %s, %s, %s, %s)""",
                            (result_id, sample_dict[cur_id][0], sample_dict[cur_id][1],
                            'Negative' if final_results["SResults"][i] < 0 else 'Positive',
                            float(final_results["SResults"][i]))
                        )
                # update epoch results
                cur_results = {}
                for mode in ['train', 'valid', 'test']:
                    cur_epoch_results = results_dict['final_results'][mode]
                    cur_results[mode] = {
                        "loss_value":cur_epoch_results["Loss"],
                        "accuracy":cur_epoch_results["Non0_acc_2"],
                        "f1":cur_epoch_results["Non0_F1_score"],
                        "mae":cur_epoch_results["MAE"],
                        "corr":cur_epoch_results["Corr"]
                    }
                cursor.execute(
                    "INSERT INTO EResult (result_id, epoch_num, results) VALUES (%s, %s, %s)",
                    (result_id, -1, json.dumps(cur_results))
                )

                epoch_num = len(results_dict['epoch_results']['train'])
                for i in range(0, epoch_num):
                    cur_results = {}
                    for mode in ['train', 'valid', 'test']:
                        cur_epoch_results = results_dict['epoch_results'][mode][i]
                        cur_results[mode] = {
                            "loss_value":cur_epoch_results["Loss"],
                            "accuracy":cur_epoch_results["Non0_acc_2"],
                            "f1":cur_epoch_results["Non0_F1_score"],
                            "mae":cur_epoch_results["MAE"],
                            "corr":cur_epoch_results["Corr"]
                        }
                    cursor.execute(
                        "INSERT INTO EResult (result_id, epoch_num, results) VALUES (%s, %s, %s)",
                        (result_id, i+1, json.dumps(cur_results))
                    )
                db.commit()
                logger.info(f"Task {task_id} Finished.")
            except Exception as e:
                logger.exception(e)
                db.rollback()
                # TODO: remove saved features
                raise e
            cursor.execute("UPDATE Task SET state = %s WHERE task_id = %s", (1, task_id))
        except Exception as e:
            logger.exception(e)
            logger.error(f"Task {task_id} Error.")
            if cursor:
                cursor.execute("UPDATE Task SET state = %s WHERE task_id = %s", (2, task_id))
        finally:
            if cursor:
                cursor.execute("UPDATE Task SET end_time = %s WHERE task_id = %s", (datetime.now(), task_id))
                db.commit()


    def DEMO_run(db_url, feature_file, model_name, dataset_name, result_id, seed):

        db_params = db_url.split('//')[1].split('@')[0].split(':')
        db_user = db_params[0]
        db_pass = db_params[1]
        db_params = db_url.split('//')[1].split('@')[1].split('/')
        db_host = db_params[0]
        db_name = db_params[1]
        # connect to db
        db = mysql.connector.connect(
            user=db_user, password=db_pass, host=db_host, database=db_name
        )
        cursor2 = db.cursor(named_tuple=True)
        cursor2.execute(
            "SELECT * FROM Result WHERE result_id = %s", (result_id,)
        )
        result = cursor2.fetchone()
        save_model_path = result.save_model_path
        assert Path(save_model_path).exists(), f"pkl file {save_model_path} not found."
        result_args = json.loads(result.args)
        args = get_config_regression(model_name, dataset_name)
        args['train_mode'] = _resolve_train_mode(dataset_name, model_name) # backward compatibility. TODO: remove all train_mode in code
        args['cur_seed'] = 1
        args.update(result_args)
        args['feature_T'] = feature_file
        args['feature_A'] = feature_file
        args['feature_V'] = feature_file
        # args['device'] = assign_gpu([])
        args['device'] = 'cpu'
        setup_seed(seed)
        model = AMIO(args).to(args['device'])
        model.load_state_dict(torch.load(save_model_path))
        model.to(args['device'])
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
        feature_a = torch.Tensor(features['audio']).unsqueeze(0)
        feature_t = torch.Tensor(features['text']).unsqueeze(0)
        feature_v = torch.Tensor(features['video']).unsqueeze(0)
        if 'need_normalized' in args and args['need_normalized']:
            feature_a = torch.mean(feature_a, dim=1, keepdims=True)
            feature_v = torch.mean(feature_v, dim=1, keepdims=True)
        model.eval()
        with torch.no_grad():
            outputs = model(feature_t, feature_a, feature_v)
        predict = round(float(outputs['M'].cpu().detach().squeeze()), 3)
        return predict


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Train/evaluate CAMFNet-AML and baseline models."
    )
    parser.add_argument('-m', '--model', required=True, help='Model name, e.g. camfn or camfnet_aml.')
    parser.add_argument('-d', '--dataset', required=True, help='Dataset name.')
    parser.add_argument('-c', '--config', default='', help='Path to config JSON file.')
    parser.add_argument('-s', '--seed', dest='seeds', action='append', type=int, default=[],
                        help='Seed value. Can be specified multiple times.')
    parser.add_argument('--model-save-dir', default=str(Path.cwd() / "saved_models"), help='Model output directory.')
    parser.add_argument('--res-save-dir', default=str(Path.cwd() / "results"), help='Result output directory.')
    parser.add_argument('--log-dir', default=str(Path.cwd() / "logs"), help='Log output directory.')
    parser.add_argument('-g', '--gpu-id', dest='gpu_ids', action='append', type=int, default=[],
                        help='GPU id. Can be specified multiple times.')
    parser.add_argument('-n', '--num-workers', type=int, default=4, help='Data loader workers.')
    parser.add_argument('-v', '--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level.')
    parser.add_argument('--tune', action='store_true', help='Enable hyper-parameter tuning mode.')
    parser.add_argument('--tune-times', type=int, default=50, help='Number of tuning trials.')
    parser.add_argument('--custom-feature', default=None, help='Path to unified custom feature file.')
    parser.add_argument('--feature-T', default=None, help='Path to custom text feature file.')
    parser.add_argument('--feature-A', default=None, help='Path to custom audio feature file.')
    parser.add_argument('--feature-V', default=None, help='Path to custom video feature file.')
    parser.add_argument('--disable-text', action='store_true', help='Disable text modality.')
    parser.add_argument('--disable-audio', action='store_true', help='Disable audio modality.')
    parser.add_argument('--disable-video', action='store_true', help='Disable video modality.')
    return parser


def main():
    parser = _build_cli_parser()
    args = parser.parse_args()
    model_name = _canonicalize_model_name(args.model)
    if model_name not in {'tetfn', 'camfn'}:
        raise ValueError(f"Unsupported model '{args.model}'. Supported models: {SUPPORTED_MODELS}")
    dataset_name = args.dataset.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported datasets: {SUPPORTED_DATASETS}")
    MMSA_run(
        model_name=model_name,
        dataset_name=dataset_name,
        config_file=args.config if args.config else None,
        seeds=args.seeds,
        is_tune=args.tune,
        tune_times=args.tune_times,
        custom_feature=args.custom_feature,
        feature_T=args.feature_T,
        feature_A=args.feature_A,
        feature_V=args.feature_V,
        gpu_ids=args.gpu_ids if args.gpu_ids else [0],
        num_workers=args.num_workers,
        verbose_level=args.verbose,
        model_save_dir=args.model_save_dir,
        res_save_dir=args.res_save_dir,
        log_dir=args.log_dir,
        disable_text=args.disable_text,
        disable_audio=args.disable_audio,
        disable_video=args.disable_video,
    )


if __name__ == "__main__":
    main()
