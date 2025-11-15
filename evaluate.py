import argparse
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data_utils import MultimodalSurvDataset, prepare_data
from src.models import SurvivalModelMM, get_image_encoder
from src.utils import get_device, set_seed


def _infer_fold_from_path(model_path: str) -> Optional[int]:
    match = re.search(r"fold(\d+)", os.path.basename(model_path))
    return int(match.group(1)) if match else None


def _build_eval_loader(cfg, modalities, clinical_df, mri_vols, fold: int) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    if "fold" not in clinical_df.columns:
        raise ValueError("Clinical dataframe must contain a 'fold' column for evaluation.")

    va_mask = clinical_df["fold"].values == fold
    if not np.any(va_mask):
        raise ValueError(f"No validation samples found for fold {fold}.")

    tr_mask = ~va_mask
    features = None
    if "clinical" in modalities:
        feats_df = clinical_df[cfg.data.features].astype(float)
        scaler = StandardScaler()
        if np.any(tr_mask):
            scaler.fit(feats_df.iloc[tr_mask])
        else:  # fallback: fit on validation set if no training rows
            scaler.fit(feats_df.iloc[va_mask])
        features = scaler.transform(feats_df.iloc[va_mask])

    times = pd.to_numeric(clinical_df["time_to_follow-up/BCR"], errors="coerce").values.astype(float)
    events = pd.to_numeric(clinical_df["BCR"], errors="coerce").values.astype(int)
    times = np.where(times <= 0, 0.1, times)

    def _slice_vol(modality):
        if modality in mri_vols:
            return mri_vols[modality][va_mask]
        return None

    dataset = MultimodalSurvDataset(
        times[va_mask],
        events[va_mask],
        X=features,
        t2_vols=_slice_vol("t2"),
        hbv_vols=_slice_vol("hbv"),
        adc_vols=_slice_vol("adc"),
        transform=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(getattr(cfg.train, "batch_size", 1)),
        shuffle=False,
        drop_last=False,
        num_workers=int(getattr(cfg.data, "num_workers", 0)),
        pin_memory=bool(getattr(cfg.data, "pin_memory", False)),
    )
    return loader, times[va_mask], events[va_mask]


def evaluate(cfg, model_path: str, fold: Optional[int] = None) -> float:
    if fold is None:
        fold = _infer_fold_from_path(model_path)
    if fold is None:
        raise ValueError("Fold must be provided either via --fold or encoded in the model filename (e.g., fold0).")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    set_seed(cfg.exp.seed)
    device = get_device()

    modalities, clinical_df, mri_vols = prepare_data(cfg)
    eval_loader, times, events = _build_eval_loader(cfg, modalities, clinical_df, mri_vols, fold)

    model = SurvivalModelMM(modalities=modalities, d_emb=cfg.fusion_model.embed_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    image_encoder = get_image_encoder(cfg, device)

    risks = []
    with torch.no_grad():
        for inputs, _, _ in eval_loader:
            mm_inputs = {}
            if "clinical" in inputs:
                mm_inputs["clinical"] = inputs["clinical"].to(device)

            for modality in ("t2", "hbv", "adc"):
                if modality in inputs:
                    vol = inputs[modality].to(device)
                    if cfg.image_encoder.type == "M3D-CLIP":
                        emb = image_encoder.encode_image(vol)[:, 0]
                    else:
                        emb = image_encoder.forward(vol)
                        emb = torch.amax(emb, dim=[2, 3, 4])
                    mm_inputs[modality] = emb

            risk = model(mm_inputs)
            risks.append(risk.detach().cpu().numpy())

    risk_scores = np.concatenate(risks, axis=0)
    c_index = concordance_index(times, -risk_scores, events)
    print(f"Fold {fold} C-index: {c_index:.4f}")
    return c_index


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained multimodal survival model.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--model_path", required=True, help="Path to the trained SurvivalModelMM checkpoint.")
    parser.add_argument("--fold", type=int, default=None, help="Validation fold to evaluate (defaults to parsing from filename).")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    evaluate(cfg, args.model_path, args.fold)

if __name__ == "__main__":
    main()
