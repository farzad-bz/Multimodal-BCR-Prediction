import torch
import numpy as np
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
import transformers
transformers.utils.logging.set_verbosity_error()
import argparse
from omegaconf import OmegaConf
from src.utils import create_logger_and_dirs, get_device, set_seed
from src.data_utils import prepare_data, make_loaders
from src.models import get_image_encoder
from src.trainer import train_one_fold


def main(cfg):
    cfg, logger, wandb_logger = create_logger_and_dirs(cfg)
    set_seed(seed=cfg.exp.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    modalities, clinical_df, MRIs = prepare_data(cfg)

    # ======= Main: CV over predefined folds with seeds =======
    unique_folds = np.sort(pd.unique(clinical_df["fold"].values))

    fold_cidx = []
    for fold in unique_folds:
        ld_tr, ld_va, (T_va, E_va) = make_loaders(cfg, modalities, clinical_df, MRIs, fold)
        c_index = train_one_fold(cfg, get_image_encoder(cfg, device), modalities, ld_tr, ld_va, T_va, E_va, device=device)
        fold_cidx.append(c_index)

    print(f"DeepSurv - CV C-index: mean:{np.mean(fold_cidx):.4f}, std:{np.std(fold_cidx)}")
    print("C-index for different folds:", fold_cidx)
    print("===END===")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicting biochemical recurrence (BCR) using multimodal data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/base.yaml)"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)