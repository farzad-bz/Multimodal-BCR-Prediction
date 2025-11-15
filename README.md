# Multimodal BCR Prediction
This repository contains a compact pipeline for predicting biochemical recurrence (BCR) of prostate cancer by fusing pre-operative multi-parametric MRI (mpMRI) volumes with clinical covariates. The default experiment fine-tunes a lightweight survival head on top of frozen M3D-CLIP volumetric embeddings and evaluates performance with cross-validated concordance index (C-index).

## Repository layout
- `configs/base.yaml` – central configuration for data paths, modalities, optimizer and logging options.
- `data/` – expected location of the processed clinical spreadsheet, five-fold split definition and per-modality MRI tensors (`{patient_id}_{t2|hbv|adc}.npy`).
- `src/data_utils.py` – dataset utilities that scale clinical features, build TorchIO augmentation pipelines and load NumPy MRI volumes into PyTorch `DataLoader`s.
- `src/models.py` – modular fusion head (`SurvivalModelMM`) plus the helper that loads the frozen `GoodBaiBai88/M3D-CLIP` encoder.
- `src/trainer.py` – survival-training loop with ranking loss, per-fold validation and early stopping based on C-index.
- `train.py` – orchestrates data prep, cross-validated training and logging.

## Prerequisites
Create an environment with Python 3.9+ and install the dependencies manually (the minimal set implied by the source code is `torch`, `torchvision`, `torchio`, `transformers`, `omegaconf`, `pandas`, `numpy`, `scikit-learn`, `lifelines`, and `wandb`). GPU acceleration is recommended for the M3D-CLIP encoder.

## Data expectations
1. `data/clinical_data_processed.csv` – rows indexed by subject, containing at least the six features listed in `configs/base.yaml` plus `fold`, `patient_id`, `time_to_follow-up/BCR`, and `BCR`.
2. `data/data_split_5fold.csv` – optional helper for constructing the `fold` column.
3. `data/preprcoessed_mpMRI/` – NumPy volumes saved per patient and modality as `{patient_id}_t2.npy`, `{patient_id}_hbv.npy`, `{patient_id}_adc.npy`. Add or remove modalities via `data.modalities`.

If you modify feature names or add modalities, keep the YAML lists in sync so that `prepare_data` can build the modality-to-dimension mapping correctly.

## Running an experiment
```bash
python train.py --config configs/base.yaml
```
Key config knobs:
- `data.modalities` toggles which inputs are fused (e.g., comment in hbv/adc to rely only on clinical + T2).
- `fusion_model.embed_dim` controls the shared representation size for each modality before fusion.
- `train.batch_size`, `train.epochs`, `train.stop_patience` adjust runtime and early stopping.
- `wandb_logging.enabled` can be turned off for offline runs.

Each run creates `outputs/<exp.name>/<run_id>/` containing logs, the exact config snapshot and best model weights. Validation C-index per fold is printed to stdout for quick tracking.

## Evaluation and extension
- Extend `train.py` or implement `evaluate.py` for held-out testing by reusing `src/data_utils.make_loaders`.
- To benchmark tabular-only baselines, populate `Classical_ML_models.py` with scikit-learn survival models and feed the standardized clinical features from `prepare_data.py`.
- To try different medical encoders, replace `image_encoder.pretrained_path` with another Hugging Face repo that exposes `encode_image`.
