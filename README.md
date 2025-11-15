#  Multimodal BCR Prediction
Predict biochemical recurrence (BCR) in prostate cancer by fusing clinical covariates with pre-operative multi-parametric MRI (mpMRI). The pipeline freezes a volumetric encoder (M3D-CLIP or MedicalNet), feeds modality embeddings to a lightweight survival head, and reports five-fold cross-validated concordance (C-index). Training summaries are automatically appended to `results.csv` for easy leaderboard tracking.

---

## 🧭 Repository Map
| Path | Purpose |
| --- | --- |
| `configs/*.yaml` | Ready-to-run configs spanning clinical-only baselines, MedicalNet, and M3D-CLIP modality mixes (LR sweeps, augmentation toggles, etc.). |
| `prepare_and_preparoces_data.py` | Helper script that assembles the processed clinical CSV and `{patient_id}_{modality}.npy` volumes. |
| `src/data_utils.py` | Loads clinical/MRI tensors, applies TorchIO augmentations, and builds PyTorch `DataLoader`s. |
| `src/models.py` | Implements `SurvivalModelMM` plus helpers that download frozen volumetric encoders from the Hugging Face Hub. |
| `src/trainer.py` | Pairwise-ranking training loop with LR scheduling, early stopping, and per-fold validation logging. |
| `train.py` | Cross-validation driver: prepares data, trains each fold, writes checkpoints, and logs metrics to `results.csv`. |
| `Classical_ML_models.py`, `evaluate.py` | Scaffolding for clinical-only baselines and held-out evaluation scripts. |

---

## ⚙️ Environment Setup
1. Use Python ≥3.9 with CUDA-enabled PyTorch for the volumetric encoders.
2. Install dependencies via pip (Conda users can initialize an environment first, then run the same commands):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🗂️ Data Requirements
1. **Clinical spreadsheet — `data/clinical_data_processed.csv`**
   - Contains `patient_id`, `fold` (integers 0–4), `time_to_follow-up/BCR` (>0), `BCR` (0/1), plus the engineered features referenced in each config (default: `ISUP`, `positive_lymph_nodes`, `lymphovascular_invasion`, `invasion_seminal_vesicles`, `positive_lymph_nodes_missing`, `age_at_prostatectomy`).
   - You can reuse `data/data_split_5fold.csv` to populate the `fold` column or build your own splits; just ensure every fold index appears in the CSV.

2. **MRI tensors**
   - Preprocess the modalities you plan to use (`t2`, `hbv`, `adc`): bias-field correction, resampling, cropping, normalization.
   - Save volumes as `np.save(f"{patient_id}_{modality}.npy", arr)` with shape `(D, H, W)` so TorchIO can add the channel dimension.
   - Store arrays under:
     - `data/preprcoessed_mpMRI_M3D-CLIP/` when training M3D-CLIP configs.
     - `data/preprcoessed_mpMRI_MedicalNet/` when training MedicalNet configs.
   - Every `patient_id` in the clinical CSV must have all modalities enabled in the chosen config.

3. **Automation (optional)**
   - `prepare_and_preparoces_data.py` and `Multimodal-Quiz/MRI_preprocessing.ipynb` demonstrate scripted/notebook-based preprocessing pipelines.

---

## 🧪 Configuring Experiments
Each YAML config defines:
- `exp.*` → experiment name, random seed, output directory root.
- `data.*` → file paths, modality list (`clinical`, `t2`, `hbv`, `adc`), engineered features, dataloader workers, pin-memory, augmentation flag.
- `fusion_model.embed_dim` → dimension each modality projects into before fusion.
- `image_encoder.*` → encoder choice (`MedicalNet` via `TencentMedicalNet/MedicalNet-Resnet10`, `M3D-CLIP` via `GoodBaiBai88/M3D-CLIP`) and the matching embedding size.
- `optim.*` / `train.*` → AdamW LR + weight decay, ReduceLROnPlateau factor/patience, gradient clipping, batch size, grad accumulation, epochs, early-stop patience.
- `wandb_logging.*` → enable online logging or switch to offline mode for air-gapped runs.

Duplicate a template (e.g., `configs/M3D-CLIP_hbv_and_clinical.yaml`) to explore new modality mixes, augmentation policies, or LR sweeps.

---

## 🚀 Training Workflow
```bash
python train.py --config configs/M3D-CLIP_t2_and_clinical.yaml
```
`train.py` performs:
1. Deterministic seeding + device selection (`src/utils.py`).
2. Loading of clinical tables, MRI tensors, and the frozen encoder weights (`src/data_utils.prepare_data`, `src.models.get_image_encoder`).
3. Five-fold cross-validation based on the `fold` column:
   - Build TorchIO-augmented loaders via `src/data_utils.make_loaders`.
   - Train `SurvivalModelMM` with the pairwise ranking loss (`src/losses.py`) and ReduceLROnPlateau scheduling (`src/trainer.py`).
   - Save the best checkpoint per fold to `outputs/<exp.name>/<run_id>/SurvivalModelMM_fold{fold}_best.pt`.
   - Stream metrics to stdout, optionally to Weights & Biases, and append C-index stats to `results.csv` (per fold, mean, std, encoder settings, modalities, augmentation flag).

Repeat with different configs (clinical-only baseline, MedicalNet sweeps, M3D-CLIP variants) to grow a comparable leaderboard inside `results.csv`.

---

## 📈 Evaluation & Extensions

The models are released as PyTorch checkpoints and hosted on [Hugging Face](https://huggingface.co/farzadbz/BCR_prediction_model) for easy reuse.

---
```python
from huggingface_hub import hf_hub_download
import torch
from src.models import SurvivalModelMM

# Download model
model_path = hf_hub_download(repo_id="farzadbz/BCR_prediction_model", filename="your desired model path")
ckpt = torch.load(model_path, map_location="cpu")
model = SurvivalModelMM(modalities=modalities, d_emb=embed_dim).to(device)
model.load_state_dict(ckpt)

```

- `evaluate.py` – load a saved checkpoint and reuse `src/data_utils.make_loaders` for held-out testing.
- `Classical_ML_models.py` – implement CoxPH, Random Survival Forests, or other scikit-learn baselines on the clinical-only features.
- `SurvivalModelMM` is modality-agnostic; add new inputs by declaring their dimensions in the `modalities` dict and providing embeddings during training/inference.

