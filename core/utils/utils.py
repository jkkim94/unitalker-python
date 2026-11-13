import argparse
import logging
import numpy as np
import random
import torch
import os
from copy import deepcopy

from . import config
from models.unitalker.dataset.dataset_config import dataset_config


def get_audio_encoder_dim(audio_encoder: str):
    if audio_encoder == "microsoft/wavlm-base-plus":
        return 768
    if audio_encoder == "microsoft/wavlm-large":
        return 1024
    if audio_encoder == "facebook/wav2vec2-large-xlsr-53":
        return 1024
    raise ValueError("wrong audio_encoder")


def filer_list(in_list: list, key: str):
    return [line for line in in_list if line.startswith(key)]


def filter_unitalker_state_dict(in_dict):
    out_dict = {}
    for k, v in in_dict.items():
        if "pca_layer_dict" not in k:
            out_dict[k] = v
    return out_dict


def load_ckpt(model, ckpt_path, re_init_decoder_and_head: bool = False):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    id_embedding_key = "decoder.learnable_style_emb.weight"
    if id_embedding_key in checkpoint and len(checkpoint[id_embedding_key]) != len(model.state_dict()[id_embedding_key]):
        target_id_num = len(model.state_dict()[id_embedding_key])
        checkpoint[id_embedding_key] = checkpoint[id_embedding_key][-1][None].repeat(target_id_num, 1)
    if re_init_decoder_and_head is True:
        ckpt_keys = list(checkpoint.keys())
        encoder_keys = filer_list(ckpt_keys, "audio_encoder")
        reinit_keys = list(set(ckpt_keys) - set(encoder_keys))
        for k in reinit_keys:
            if k in model.state_dict():
                checkpoint[k] = model.state_dict()[k]
    model.load_state_dict(checkpoint, strict=False)


def get_template_verts(data_root, dataset_name: str, subject: int):
    template_path = os.path.join(data_root, dataset_config[dataset_name]["dirname"], "id_template.npy")
    return np.load(template_path)[subject]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def get_average_meter_dict(args, phase):
    if phase == "train":
        metric_name_list = ["rec_L2_loss", "pca_loss", "blendshape_loss", "mouth_loss", "viseme_loss"]
    elif phase in ("val", "test"):
        # LVE/LVD: vertex-space only (MetahumanBlendShapeLoss / mesh).
        # mouth_control_mse/mae: MetahumanControlLoss lip-subset (NOT vertex LVE).
        metric_name_list = [
            "rec_L2_loss",
            "LVE",
            "LVD",
            "blendshape_loss",
            "control_MSE",
            "mouth_control_mse",
            "mouth_control_mae",
        ]
    else:
        raise ValueError("wrong phase for average meter")
    average_meter_dict_template = {k: AverageMeter() for k in metric_name_list}
    mixed_dataset_name = "mixed_dataset"
    dataset_name_list = args.dataset + [mixed_dataset_name]
    average_meter_dict = {}
    for dataset_name in dataset_name_list:
        average_meter_dict[dataset_name] = deepcopy(average_meter_dict_template)
    return average_meter_dict, metric_name_list


def log_datasetloss(logger, epoch: int, phase: str, dataset_loss_dict: dict, only_mix: bool = True):
    base_str = f"{phase.upper():<5} Epoch: {epoch:03d} "
    for dataset_name, loss_info in dataset_loss_dict.items():
        if only_mix and dataset_name != "mixed_dataset":
            continue
        for loss_type, avg_meter in loss_info.items():
            base_str += f"{loss_type}_{dataset_name}:{avg_meter.avg: .8f}, "
    logger.info(base_str)
    return base_str


def write_to_tensorboard(writer, epoch: int, phase: str, dataset_loss_dict: dict, only_mix: bool = False):
    for dataset_name, loss_info in dataset_loss_dict.items():
        if only_mix and dataset_name != "mixed_dataset":
            continue
        for loss_type, avg_meter in loss_info.items():
            key = f"{phase}/{loss_type}_{dataset_name}"
            writer.add_scalar(key, avg_meter.avg, epoch)


def write_to_wandb(
    wandb_logger,
    epoch: int,
    phase: str,
    dataset_loss_dict: dict,
    only_mix: bool = False,
    commit: bool = True,
    step_override=None,
):
    if wandb_logger is None:
        return
    from core.trainer.wandb_logger import is_unitalker_wandb_dataset, is_unitalker_wandb_metric

    step = epoch if step_override is None else step_override
    metrics = {}
    for dataset_name, loss_info in dataset_loss_dict.items():
        if only_mix and dataset_name != "mixed_dataset":
            continue
        if not is_unitalker_wandb_dataset(dataset_name):
            continue
        for loss_type, avg_meter in loss_info.items():
            if not is_unitalker_wandb_metric(loss_type):
                continue
            key = f"{phase}/{loss_type}_{dataset_name}"
            metrics[key] = avg_meter.avg
    if metrics:
        metrics[f"{phase}/epoch"] = epoch
        wandb_logger.log(metrics, step=step, commit=commit)


def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--config", type=str, default="config/unitalker.yaml", help="config file")
    parser.add_argument("--condition_id", type=str, default="common", help="condition id")
    parser.add_argument("opts", help=" ", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.condition_id = args.condition_id
    cfg.dataset = cfg.dataset.split(",")
    cfg.demo_dataset = cfg.demo_dataset.split(",")
    if isinstance(cfg.duplicate_list, str):
        cfg.duplicate_list = cfg.duplicate_list.split(",")
        cfg.duplicate_list = [int(i) for i in cfg.duplicate_list]
    return cfg


def get_logger(log_file: str):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode="w")
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d]=>%(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    for hdl in logger.handlers:
        logger.removeHandler(hdl)
    logger.addHandler(handler)
    return logger


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
