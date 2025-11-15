# Multimodal BCR Prediction
Use pre-operative multi-parametric MRI (mpMRI) together with clinical covariates to predict biochemical recurrence (BCR) in prostate cancer. The pipeline freezes an M3D-CLIP volumetric encoder, trains a lightweight survival head, and reports cross-validated concordance index (C-index) for reproducible benchmarking.

---

## 🗂️ Repository Layout
- `configs/base.yaml` – central configuration for data paths, modalities, optimization, and logging.
- `data/` – processed clinical spreadsheet, five-fold split helpers, and `{patient_id}_{t2|hbv|adc}.npy` MRI tensors.
- `src/data_utils.py` – TorchIO augmentations, modality loaders, and standardized `DataLoader` builders.
- `src/models.py` – the modular fusion head (`SurvivalModelMM`) plus `get_image_encoder`.
- `src/trainer.py` – ranking-loss training with per-fold validation, LR scheduling, and early stopping.
- `train.py` – main entry point that wires data prep, training, and logging.

---

## ⚙️ Environment Setup
Create a Python 3.9+ environment and install dependencies (`torch`, `torchvision`, `torchio`, `transformers`, `omegaconf`, `pandas`, `numpy`, `scikit-learn`, `lifelines`, `wandb`, etc.). A CUDA-capable GPU speeds up M3D-CLIP embedding.

```bash
# 🐍 Virtualenv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 🐚 Conda (optional)
conda create -n multimodal-bcr python=3.9 -y
conda activate multimodal-bcr
pip install -r requirements.txt
```

---

## 🧼 Data Preparation & Preprocessing
1. **📄 Clinical spreadsheet**
   - Ensure unique `patient_id` per row in the raw CSV.
   - Create a `fold` column (from `data/data_split_5fold.csv` or your own split).
   - Add survival labels `time_to_follow-up/BCR` and `BCR`.
   - Engineer the six features listed in `configs/base.yaml` (including binary indicators for missing values) and save to `data/clinical_data_processed.csv`.

2. **🧲 mpMRI volumes**
   - Run your MRI preprocessing (bias-field correction, resample, crop, normalize, etc.) for each modality (`t2`, `hbv`, `adc`).
   - Save tensors as `numpy.save(f"{patient_id}_{modality}.npy", tensor)` under `data/preprcoessed_mpMRI/` with shape `(D, H, W)` so TorchIO can add the channel dimension.

3. **🛠️ Optional automation**
   - Use `prepare_and_preparoces_data.py` or the `Multimodal-Quiz/MRI_preprocessing.ipynb` notebook to batch the steps above.
   - Double-check that every `patient_id` appearing in the clinical CSV has all requested modality files before training.

---

## 📦 Data Expectations
1. `data/clinical_data_processed.csv` – indexed by subject with the required features, `fold`, `patient_id`, `time_to_follow-up/BCR`, and `BCR`.
2. `data/data_split_5fold.csv` – optional helper for constructing folds.
3. `data/preprcoessed_mpMRI/` – `{patient_id}_{modality}.npy` tensors for each modality you plan to use.

Keep `configs/base.yaml` synchronized with the actual feature names and modality availability.

---

## 🚀 Running an Experiment
```bash
python train.py --config configs/base.yaml
```
Key knobs inside the config:
- `data.modalities` – toggle which inputs are fused (e.g., enable `hbv`/`adc` to add MRI channels).
- `fusion_model.embed_dim` – shared representation size per modality before fusion.
- `train.batch_size`, `train.epochs`, `train.stop_patience` – runtime and early stopping behavior.
- `wandb_logging.enabled` – set `false` for offline experiments.

Each run writes `outputs/<exp.name>/<run_id>/` with logs, the exact config snapshot, model checkpoints, and metrics (C-index per fold printed to stdout).

---

## 🔍 Evaluation & Extension
- Extend `train.py` or complete `evaluate.py` for held-out testing using `src/data_utils.make_loaders`.
- Populate `Classical_ML_models.py` with scikit-learn baselines to compare against the multimodal model.
- Swap `image_encoder.pretrained_path` if you want to experiment with other 3D medical encoders that expose `encode_image`.

Happy experimenting! 🧪🚀
