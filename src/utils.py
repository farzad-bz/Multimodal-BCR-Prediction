import os
import sys
import torch
import random
import numpy as np
import wandb
import logging
from glob import glob
from omegaconf import OmegaConf



def create_logger_and_dirs(cfg, fold):
    """
    Create a logger that writes to a log file, stdout and wandb logger.
    """
    wandb_logger = None
    if cfg.wandb_logging.enabled:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger = wandb.init(project=cfg.wandb_logging.project,
                              config=cfg_dict,
                              name = cfg.wandb_logging.run_name + f'_fold{fold}',
                              mode=cfg.wandb_logging.mode)
        cfg.wandb_id = wandb.run.id
    
    experiment_index = len(glob(f"{cfg.exp.output_dir}/{cfg.exp.name}_fold{fold}'/*"))
    cfg.exp.experiment_dir = os.path.join(cfg.exp.output_dir, cfg.exp.name, f'{experiment_index:03d}')
    os.makedirs(cfg.exp.experiment_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    
    with open(os.path.join(cfg.exp.experiment_dir, "config_used.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f.name)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{cfg.exp.experiment_dir}/log.txt")],
        force=True,  
    )
    logger = logging.getLogger(__name__)

    # catch uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    return cfg, logger, wandb_logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
