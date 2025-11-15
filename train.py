import os
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
    set_seed(seed=cfg.exp.seed)
    device = get_device()

    modalities, clinical_df, MRIs = prepare_data(cfg)

    # ======= Main: CV over predefined folds with seeds =======
    unique_folds = np.sort(pd.unique(clinical_df["fold"].values))

    fold_cidx = []
    for fold in unique_folds:
        cfg, logger, wandb_logger = create_logger_and_dirs(cfg, fold)
        logger.info("=" * 10  + f"  Start training for validation fold {fold}  " + "=" * 10)
        ld_tr, ld_va, (T_va, E_va) = make_loaders(cfg, modalities, clinical_df, MRIs, fold)
        best_model, c_index = train_one_fold(cfg, get_image_encoder(cfg, device), modalities, ld_tr, ld_va, T_va, E_va, fold, logger=logger, wandb_logger=wandb_logger, device=device)
        fold_cidx.append(c_index)
        torch.save(best_model, os.path.join(cfg.exp.experiment_dir, "SurvivalModelMM_best.pth"))
        if wandb_logger:
            wandb_logger.summary[f"C-index for fold {fold}"] = c_index
            wandb_logger.finish()

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