"""
基线模型统一运行脚本
用于在气象PKL数据集上运行基线模型，与HyperGKAN进行对比实验。

用法:
    # 运行baseline_config.yaml中指定的所有模型和数据集
    python run_baseline.py

    # 指定配置文件
    python run_baseline.py --config baseline_config.yaml

    # 命令行覆盖：只跑单个模型
    python run_baseline.py --model STGCN --dataset Temperature

    # 命令行覆盖：指定GPU (服务器多卡并发)
    python run_baseline.py --gpu 0
    python run_baseline.py --model STGCN --dataset Temperature --gpu 0
    python run_baseline.py --model GWNET --dataset Cloud --gpu 1
"""
# ========================================
# ⚠️ GPU 环境变量设置（必须在 import torch 之前）
# ========================================
import os
import sys

# 解析 --gpu 参数，设置 CUDA_VISIBLE_DEVICES 环境变量
gpu_id = None
if '--gpu' in sys.argv:
    idx = sys.argv.index('--gpu')
    if idx + 1 < len(sys.argv):
        gpu_id = sys.argv[idx + 1]
# 也支持 --gpu_id (向后兼容)
elif '--gpu_id' in sys.argv:
    idx = sys.argv.index('--gpu_id')
    if idx + 1 < len(sys.argv):
        gpu_id = sys.argv[idx + 1]

if gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES = {gpu_id}")

# 现在可以安全导入 torch
import argparse
import logging
import datetime
import json
import random
import numpy as np
import torch
import yaml
import importlib
from copy import deepcopy

# 确保当前目录在path中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libcity.data.dataset.pkl_dataset import PklDataset
from libcity.executor.meteo_executor import MeteoExecutor
from libcity.utils import ensure_dir


# ==================== 气象要素数据映射 ====================
ELEMENT_MAP = {
    "Temperature": {"data_dir": "temperature", "output_dim": 1, "kelvin_to_celsius": True},
    "Cloud": {"data_dir": "cloud_cover", "output_dim": 1, "kelvin_to_celsius": False},
    "Humidity": {"data_dir": "humidity", "output_dim": 1, "kelvin_to_celsius": False},
    "Wind": {"data_dir": "component_of_wind", "output_dim": 2, "kelvin_to_celsius": False},
}

# ==================== 模型默认配置 ====================
# 各模型的特定参数（从原始JSON配置提取）
MODEL_DEFAULTS = {
    "STGCN": {
        "Ks": 3, "Kt": 3,
        "blocks": [[1, 32, 64], [64, 32, 128]],
        "dropout": 0,
        "graph_conv_type": "chebconv",
        "stgcn_train_mode": "quick",
        "bidir_adj_mx": True,
        "scaler": "standard",
        "learner": "rmsprop",
        "learning_rate": 0.001,
        "lr_decay": True,
        "lr_scheduler": "steplr",
        "lr_decay_ratio": 0.7,
        "step_size": 5,
        "clip_grad_norm": False,
        "train_loss": "none",
    },
    "GWNET": {
        "dropout": 0.3, "blocks": 4, "layers": 2,
        "apt_layer": True, "gcn_bool": True, "addaptadj": True,
        "adjtype": "doubletransition", "bidir_adj_mx": False,
        "randomadj": True, "aptonly": True, "kernel_size": 2,
        "nhid": 32, "residual_channels": 32, "dilation_channels": 32,
        "skip_channels": 256, "end_channels": 512,
        "scaler": "standard",
        "learner": "adam",
        "learning_rate": 0.001,
        "lr_decay": False,
        "clip_grad_norm": True,
        "max_grad_norm": 5,
        "train_loss": "none",
    },
    "AGCRN": {
        "embed_dim": 10, "rnn_units": 64, "num_layers": 2, "cheb_order": 2,
        "bidir_adj_mx": True,
        "scaler": "standard",
        "learner": "adam",
        "learning_rate": 0.003,
        "lr_decay": False,
        "clip_grad_norm": False,
        "use_early_stop": True,
        "patience": 50,
        "train_loss": "none",
    },
    "ASTGCN": {
        "nb_block": 2, "K": 3, "nb_chev_filter": 64, "nb_time_filter": 64,
        "bidir_adj_mx": True,
        "scaler": "standard",
        "learner": "adam",
        "learning_rate": 0.0001,
        "lr_decay": False,
        "clip_grad_norm": False,
        "train_loss": "none",
    },
    "GRU": {
        "rnn_type": "GRU", "hidden_size": 64, "num_layers": 1,
        "dropout": 0, "bidirectional": False, "teacher_forcing_ratio": 0,
        "scaler": "standard",
        "learner": "adam",
        "learning_rate": 0.01,
        "lr_decay": True,
        "lr_scheduler": "multisteplr",
        "lr_decay_ratio": 0.1,
        "steps": [5, 20, 40, 70],
        "clip_grad_norm": True,
        "max_grad_norm": 5,
        "use_early_stop": True,
        "patience": 50,
        "train_loss": "none",
    },
    "LSTM": {
        "rnn_type": "LSTM", "hidden_size": 64, "num_layers": 1,
        "dropout": 0, "bidirectional": False, "teacher_forcing_ratio": 0,
        "scaler": "standard",
        "learner": "adam",
        "learning_rate": 0.01,
        "lr_decay": True,
        "lr_scheduler": "multisteplr",
        "lr_decay_ratio": 0.1,
        "steps": [5, 20, 40, 70],
        "clip_grad_norm": True,
        "max_grad_norm": 5,
        "use_early_stop": True,
        "patience": 50,
        "train_loss": "none",
    },
    "STGNN": {
        "hidden_size": 64, "num_layers": 2, "dropout": 0.3,
        "lr": 0.001, "G_method": "mtgnn", "G_k": 3, "G_dim": 10, "G_alpha": 3,
        "T_ks": [2, 3, 4, 5], "T_dilation": 1,  # 修改：原[2,3,6,7]导致感受野13>12
        "R_routing": ["T", "S", "T"], "mid_channel": 64,
        "skip_channel": 64, "end_channel": 64,
        "S_hop": 2, "S_hopalpha": 0.5, "S_fusion": "concat", "G_mix": 0.5,
        "scaler": "standard",
        "learner": "adam",
        "learning_rate": 0.001,
        "lr_decay": False,
        "clip_grad_norm": False,
        "train_loss": "none",
    },
}


def set_random_seed(seed):
    """固定全局随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    datasets_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'datasets')
    )
    config['dataset_dir'] = datasets_dir
    return config


def setup_logger(log_dir, model_name, element):
    """
    设置日志系统，仿照HyperGKAN的日志格式
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'train_{timestamp}_{model_name}_{element}.log'
    log_filepath = os.path.join(log_dir, log_filename)

    # 创建logger
    logger_name = f'Baseline_{model_name}_{element}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 清除之前的handler（避免重复）
    logger.handlers.clear()

    # 文件handler
    fh = logging.FileHandler(log_filepath, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # 控制台handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    logger.info(f'Log file: {log_filepath}')
    return logger, log_filepath


def build_config_dict(yaml_cfg, model_name, element):
    """
    将YAML配置 + 模型默认配置 合并为框架所需的config字典
    """
    config = {}

    # 基本信息
    config['task'] = 'MTS_pred'
    config['model'] = model_name
    config['dataset'] = element
    config['element'] = element

    # 如果是RNN变体(GRU/LSTM)，内部模型类都是RNN
    if model_name.upper() in ['GRU', 'LSTM']:
        config['rnn_type'] = model_name.upper()
        config['model'] = 'RNN'

    # 全局设置
    global_cfg = yaml_cfg.get('global', {})
    config['seed'] = global_cfg.get('seed', 42)
    config['gpu'] = global_cfg.get('device', 'cuda') == 'cuda'
    config['gpu_id'] = global_cfg.get('gpu_id', 0)

    # 设备
    if config['gpu'] and torch.cuda.is_available():
        config['device'] = torch.device(f"cuda:{config['gpu_id']}")
    else:
        config['device'] = torch.device('cpu')

    # 数据集设置
    elem_cfg = ELEMENT_MAP.get(element, {})
    config['output_dim'] = elem_cfg.get('output_dim', 1)
    config['input_window'] = 12
    config['output_window'] = 12

    # Context特征
    config['context_features'] = yaml_cfg.get('context_features', {})

    # 采样配置
    config['num_stations'] = yaml_cfg.get('num_stations', None)
    config['train_sample_ratio'] = yaml_cfg.get('train_sample_ratio', 1.0)
    config['val_sample_ratio'] = yaml_cfg.get('val_sample_ratio', 1.0)
    config['test_sample_ratio'] = yaml_cfg.get('test_sample_ratio', 1.0)

    # 训练配置
    train_cfg = yaml_cfg.get('training', {})
    config['batch_size'] = train_cfg.get('batch_size', 16)
    config['accumulation_steps'] = train_cfg.get('accumulation_steps', 4)
    config['max_epoch'] = train_cfg.get('max_epoch', 100)

    # 模型特定参数
    model_key = model_name.upper()
    if model_key in ['GRU', 'LSTM']:
        model_key = model_name.upper()
    model_defaults = MODEL_DEFAULTS.get(model_name, {})
    for key, val in model_defaults.items():
        if key not in config:
            config[key] = val

    # 覆盖训练相关参数
    if train_cfg.get('use_early_stop') is not None:
        config['use_early_stop'] = train_cfg['use_early_stop']
    if train_cfg.get('patience') is not None:
        config['patience'] = train_cfg['patience']

    # 输出配置
    output_cfg = yaml_cfg.get('output', {})
    config['output_base_dir'] = output_cfg.get('base_dir', './outputs')

    # 其他框架需要的参数
    config['num_workers'] = 0
    config['saved_model'] = True
    config['load_best_epoch'] = True
    config['log_every'] = 1
    config['log_level'] = 'INFO'

    # 评估
    config['evaluator'] = 'StateEvaluator'
    config['metrics'] = ['MAE', 'RMSE']
    config['evaluator_mode'] = 'single'
    config['save_mode'] = ['csv']

    return config


def get_model(config, data_feature):
    """加载模型"""
    model_name = config['model']
    try:
        model_cls = getattr(importlib.import_module('libcity.model.prediction'), model_name)
        return model_cls(config, data_feature)
    except AttributeError:
        raise AttributeError(f'Model {model_name} not found')


class ConfigWrapper:
    """ConfigParser的轻量替代品，直接使用dict"""
    def __init__(self, config_dict):
        self.config = config_dict

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        raise KeyError(f'{key} is not in the config')

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return self.config.__iter__()


def _write_test_summary(summary_path, model_name, element, timestamp,
                        config_dict, train_result, test_result, total_params, exp_dir):
    """
    将测试结果汇总写入 test_summary.txt
    """
    sep = '=' * 60
    thin_sep = '-' * 60

    lines = []
    lines.append(sep)
    lines.append('  TEST RESULT SUMMARY')
    lines.append(sep)
    lines.append(f'  Model:           {model_name}')
    lines.append(f'  Element:         {element}')
    lines.append(f'  Timestamp:       {timestamp}')
    lines.append(f'  Output dir:      {exp_dir}')
    lines.append(sep)

    # 模型信息
    lines.append('')
    lines.append('[Model Info]')
    lines.append(f'  Trainable params:    {total_params:,}')
    lines.append(f'  Device:              {config_dict.get("device", "N/A")}')
    lines.append(f'  Random seed:         {config_dict.get("seed", "N/A")}')

    # 训练信息
    lines.append('')
    lines.append('[Training Info]')
    lines.append(f'  Total epochs:        {train_result["total_epochs"]}')
    lines.append(f'  Best epoch:          {train_result["best_epoch"] + 1}')
    lines.append(f'  Training time:       {train_result["total_time_min"]:.1f} min')
    lines.append(f'  Best val loss:       {train_result["best_val_loss"]:.4f}')
    lines.append(f'  Best val MAE:        {train_result["best_val_mae"]:.4f}')
    lines.append(f'  Best val RMSE:       {train_result["best_val_rmse"]:.4f}')

    # 实验配置
    lines.append('')
    lines.append('[Experiment Config]')
    lines.append(f'  Batch size:          {config_dict.get("batch_size", "N/A")}')
    lines.append(f'  Accumulation steps:  {config_dict.get("accumulation_steps", "N/A")}')
    eff_bs = config_dict.get("batch_size", 16) * config_dict.get("accumulation_steps", 1)
    lines.append(f'  Effective batch:     {eff_bs}')
    lines.append(f'  Max epoch:           {config_dict.get("max_epoch", "N/A")}')
    lines.append(f'  Learning rate:       {config_dict.get("learning_rate", "N/A")}')
    lines.append(f'  Early stop:          {config_dict.get("use_early_stop", "N/A")}')
    lines.append(f'  Patience:            {config_dict.get("patience", "N/A")}')
    lines.append(f'  Num stations:        {config_dict.get("num_stations", "all")}')
    lines.append(f'  Train sample ratio:  {config_dict.get("train_sample_ratio", 1.0)}')

    # Context特征
    ctx = config_dict.get('context_features', {})
    ctx_names = ['longitude', 'latitude', 'altitude', 'year', 'month', 'day', 'hour', 'region']
    ctx_keys = [f'use_{n}' for n in ctx_names]
    active_ctx = [n for n, k in zip(ctx_names, ctx_keys) if ctx.get(k, False)]
    lines.append(f'  Context features:    {len(active_ctx)}/8 ({", ".join(active_ctx)})')

    # 测试结果
    lines.append('')
    lines.append(sep)
    lines.append('  TEST RESULTS')
    lines.append(sep)
    lines.append(f'  Test MAE:            {test_result["mae"]:.4f}')
    lines.append(f'  Test RMSE:           {test_result["rmse"]:.4f}')

    # 每时间步结果
    lines.append('')
    lines.append(thin_sep)
    lines.append(f'  {"Step":>4}  {"MAE":>12}  {"RMSE":>12}')
    lines.append(thin_sep)

    df = test_result['step_results']
    for step_idx in df.index:
        t_mae = df.loc[step_idx, 'MAE']
        t_rmse = df.loc[step_idx, 'RMSE']
        lines.append(f'  {step_idx:>4}  {t_mae:>12.4f}  {t_rmse:>12.4f}')

    lines.append(thin_sep)
    lines.append(f'  {"Avg":>4}  {test_result["mae"]:>12.4f}  {test_result["rmse"]:>12.4f}')
    lines.append(sep)

    # 输出文件列表
    lines.append('')
    lines.append('[Output Files]')
    lines.append(f'  Predictions: {test_result.get("predictions_file", "N/A")}')
    lines.append(f'  Metrics CSV: {test_result.get("metrics_file", "N/A")}')
    lines.append(f'  This summary: {summary_path}')
    lines.append(sep)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _cleanup_temp_cache(exp_id):
    """
    清理 StateExecutor.__init__ 在 libcity/cache/ 下自动创建的空临时目录
    """
    import shutil
    temp_dir = f'./libcity/cache/{exp_id}'
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # 非关键操作，静默失败


def run_single_experiment(yaml_cfg, model_name, element, output_base):
    """
    运行单个实验：一个模型 + 一个数据集
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 输出目录
    exp_dir = os.path.join(output_base, f'{timestamp}_{model_name}_{element}')
    ensure_dir(exp_dir)
    cache_dir = os.path.join(exp_dir, 'model_cache')
    eval_dir = os.path.join(exp_dir, 'evaluate_cache')
    ensure_dir(cache_dir)
    ensure_dir(eval_dir)

    # 设置日志
    logger, log_path = setup_logger(exp_dir, model_name, element)

    logger.info('=' * 60)
    logger.info(f'Baseline Experiment')
    logger.info('=' * 60)
    logger.info(f'  Model:   {model_name}')
    logger.info(f'  Element: {element}')
    logger.info(f'  Output:  {exp_dir}')
    logger.info(f'  Time:    {timestamp}')
    logger.info('=' * 60)

    # 构建配置
    config_dict = build_config_dict(yaml_cfg, model_name, element)

    # 更新缓存目录
    exp_id = f'{model_name}_{element}_{timestamp}'
    config_dict['exp_id'] = exp_id

    config = ConfigWrapper(config_dict)

    # 设置随机种子
    seed = config.get('seed', 42)
    set_random_seed(seed)
    logger.info(f'Random seed: {seed}')

    # 保存config到输出目录
    config_save_path = os.path.join(exp_dir, 'config.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        # 过滤掉不可序列化的对象
        saveable = {k: v for k, v in config_dict.items()
                    if not isinstance(v, torch.device)}
        saveable['device'] = str(config_dict['device'])
        json.dump(saveable, f, indent=2, ensure_ascii=False)
    logger.info(f'Config saved to: {config_save_path}')

    # ---- 加载数据 ----
    logger.info('=' * 60)
    logger.info('Loading data...')

    # 注入logger到 logging系统
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        for handler in logger.handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    dataset = PklDataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    logger.info(f'Data loaded successfully')
    logger.info(f'  Num nodes:    {data_feature["num_nodes"]}')
    logger.info(f'  Feature dim:  {data_feature["feature_dim"]}')
    logger.info(f'  Output dim:   {data_feature["output_dim"]}')
    logger.info(f'  Adj mx shape: {data_feature["adj_mx"].shape}')

    # ---- 创建模型 ----
    logger.info('=' * 60)
    logger.info('Creating model...')

    model = get_model(config, data_feature)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model: {model_name}')
    logger.info(f'Trainable parameters: {total_params:,}')

    # ---- 创建执行器 ----
    # 修改executor的缓存目录到输出目录
    config['exp_id'] = exp_id
    # 临时修改缓存路径
    original_cache_dir = f'./libcity/cache/{exp_id}/model_cache'
    original_eval_dir = f'./libcity/cache/{exp_id}/evaluate_cache'
    ensure_dir(original_cache_dir)
    ensure_dir(original_eval_dir)

    executor = MeteoExecutor(config, model, data_feature)

    # 将executor的目录重定向到output
    executor.cache_dir = cache_dir
    executor.evaluate_res_dir = eval_dir
    executor.summary_writer_dir = exp_dir

    logger.info(f'Executor: MeteoExecutor')
    logger.info(f'  Cache dir: {cache_dir}')
    logger.info(f'  Eval dir:  {eval_dir}')

    # ---- 训练 ----
    logger.info('=' * 60)
    logger.info('Starting training...')

    try:
        train_result = executor.train(train_data, valid_data)
        logger.info(f'Training finished, best val loss: {train_result["best_val_loss"]:.4f}')
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logger.error(f'OOM Error! Try reducing batch_size or increasing accumulation_steps')
            logger.error(f'Current batch_size={config.get("batch_size")}, '
                         f'accumulation_steps={config.get("accumulation_steps")}')
            torch.cuda.empty_cache()
            raise
        else:
            raise

    # ---- 保存最优模型 ----
    model_cache_file = os.path.join(cache_dir, f'{model_name}_{element}_best.m')
    executor.save_model(model_cache_file)
    logger.info(f'Best model saved to: {model_cache_file}')

    # ---- 清理中间checkpoint文件，只保留最优模型 ----
    cleaned = 0
    for f in os.listdir(cache_dir):
        if f.endswith('.tar'):
            os.remove(os.path.join(cache_dir, f))
            cleaned += 1
    if cleaned > 0:
        logger.info(f'Cleaned {cleaned} intermediate checkpoint(s), kept only best model')

    # ---- 评估 ----
    logger.info('=' * 60)
    test_result = executor.evaluate(test_data)

    # ---- 生成测试结果 Summary ----
    summary_path = os.path.join(exp_dir, 'test_summary.txt')
    _write_test_summary(
        summary_path=summary_path,
        model_name=model_name,
        element=element,
        timestamp=timestamp,
        config_dict=config_dict,
        train_result=train_result,
        test_result=test_result,
        total_params=total_params,
        exp_dir=exp_dir,
    )
    logger.info(f'Test summary saved to: {summary_path}')

    logger.info('')
    logger.info('=' * 60)
    logger.info('Experiment completed!')
    logger.info(f'  Output directory: {exp_dir}')
    logger.info('=' * 60)

    # ---- 清理框架自动创建的空临时目录 ----
    _cleanup_temp_cache(exp_id)

    return test_result


def main():
    parser = argparse.ArgumentParser(description='Baseline Models for Meteorological Prediction')
    parser.add_argument('--config', type=str, default='baseline_config.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Override: run only this model (e.g. STGCN)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override: run only this dataset (e.g. Temperature)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID (0-5 for server, must be set before import torch)')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='[Deprecated] Use --gpu instead')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override: random seed')
    args = parser.parse_args()

    # 处理 GPU 参数（兼容 --gpu_id 和 --gpu）
    gpu_override = args.gpu if args.gpu is not None else args.gpu_id

    # 加载YAML配置
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    yaml_cfg = load_yaml_config(config_path)

    # 命令行覆盖
    if gpu_override is not None:
        yaml_cfg.setdefault('global', {})['gpu_id'] = 0
    if args.seed is not None:
        yaml_cfg.setdefault('global', {})['seed'] = args.seed

    # 确定要运行的模型列表
    if args.model:
        models = [args.model]
    else:
        models = yaml_cfg.get('models', ['STGCN'])

    # 确定要运行的数据集列表
    if args.dataset:
        elements = [args.dataset]
    else:
        ds_sel = yaml_cfg.get('dataset_selection', {})
        elements = [name for name, active in ds_sel.items() if active]
        if not elements:
            elements = ['Temperature']

    # 输出目录
    output_base = yaml_cfg.get('output', {}).get('base_dir', './outputs')
    ensure_dir(output_base)

    # 打印实验计划
    print('=' * 60)
    print('  Baseline Meteorological Prediction Experiment')
    print('=' * 60)
    print(f'  Models:   {models}')
    print(f'  Elements: {elements}')
    print(f'  Output:   {output_base}')
    print(f'  Seed:     {yaml_cfg.get("global", {}).get("seed", 42)}')
    print(f'  Stations: {yaml_cfg.get("num_stations", "all")}')
    print(f'  Train sample ratio: {yaml_cfg.get("train_sample_ratio", 1.0)}')
    print('=' * 60)
    print()

    # 批量运行
    results = {}
    total = len(models) * len(elements)
    idx = 0

    for element in elements:
        for model_name in models:
            idx += 1
            print(f'\n{"#" * 60}')
            print(f'# Experiment {idx}/{total}: {model_name} on {element}')
            print(f'{"#" * 60}\n')

            try:
                result = run_single_experiment(yaml_cfg, model_name, element, output_base)
                results[f'{model_name}_{element}'] = result
                print(f'\n[OK] {model_name} on {element} completed successfully!')
            except Exception as e:
                print(f'\n[FAIL] {model_name} on {element} failed: {e}')
                import traceback
                traceback.print_exc()
                results[f'{model_name}_{element}'] = f'FAILED: {e}'

    # 打印汇总
    print('\n' + '=' * 60)
    print('  All Experiments Summary')
    print('=' * 60)
    for key, result in results.items():
        if isinstance(result, str):
            print(f'  {key}: {result}')
        elif isinstance(result, dict) and 'mae' in result:
            print(f'  {key}: MAE={result["mae"]:.4f}  RMSE={result["rmse"]:.4f}  [OK]')
        else:
            print(f'  {key}: Completed [OK]')
    print('=' * 60)


if __name__ == '__main__':
    main()
