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
        torch.save(best_model.state_dict(), os.path.join(cfg.exp.experiment_dir, f"SurvivalModelMM_fold{fold}_best.pt"))
        if wandb_logger:
            wandb_logger.summary[f"C-index for fold {fold}"] = c_index
            wandb_logger.finish()

    exp = {'name': [cfg.exp.name],
        'ImageEncoder':  [cfg.image_encoder.type],
        'modalities':  ['+'.join(cfg.data.modalities)],
        'Augmentation':  [cfg.data.aug],
        'fold_0 C-index':  [float(f'{fold_cidx[0]:.4f}')],
        'fold_1 C-index':  [float(f'{fold_cidx[1]:.4f}')],
        'fold_2 C-index':  [float(f'{fold_cidx[2]:.4f}')],
        'fold_3 C-index':  [float(f'{fold_cidx[3]:.4f}')],
        'fold_4 C-index':  [float(f'{fold_cidx[4]:.4f}')],
        'Mean C-index':  [float(f'{np.mean(fold_cidx):.4f}')],
        'StDev C-index':  [float(f'{np.std(fold_cidx):.4f}')]}
    
    if os.path.exists('results.csv'):
        results_df = pd.read_csv('results.csv', index_col=0)
        results_df = pd.concat([results_df, pd.DataFrame(exp)], ignore_index=True)
    else:
        results_df = pd.DataFrame(exp)
    results_df.to_csv('results.csv')
    
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