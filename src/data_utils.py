import os
import torch
import pandas as pd
import numpy as np
import torchio as tio
from sklearn.preprocessing import StandardScaler

def prepare_data(cfg):
    clinical_df = pd.read_csv(cfg.data.clinical_df_path, index_col=0)
    features = cfg.data.features
    modalities = {}
    for modality in cfg.data.modalities:
        modalities[modality] = len(features) if modality=='clinical' else cfg.image_encoder.embed_dim 

    MRIs = {}
    for key in modalities:
        if key in ['t2', 'hbv', 'adc']:
            MRIs[key] = []
            for i,row in clinical_df.iterrows():
                MRIs[key].append(np.load(os.path.join(cfg.data.mpMRI_dir, f'{row["patient_id"]}_{key}.npy')))
            MRIs[key] = np.array(MRIs[key])
            print(f'Modality {key} has been loaded!')
    return modalities, clinical_df, MRIs



class MultimodalSurvDataset(torch.utils.data.Dataset):
    """
    Returns:
        inputs: dict of modality_name -> tensor
        T: time-to-event (scalar)
        E: event indicator (scalar)

    Modalities supported:
        - "clinical": (num_features,)
        - "t2":  (1, D, H, W) after TorchIO
        - "hbv": (1, D, H, W)
        - "adc": (1, D, H, W)
    """
    def __init__(
        self,
        T,         # times np.array (N,)
        E,         # events np.array (N,)
        X=None,         # clinical np.array (N, num_features) or None
        t2_vols=None,   
        hbv_vols=None,  
        adc_vols=None,  
        transform=None, 
    ):
        self.has_clinical = X is not None
        if self.has_clinical:
            self.X = torch.tensor(X, dtype=torch.float32)

        self.T = torch.tensor(T, dtype=torch.float32)
        self.E = torch.tensor(E, dtype=torch.float32)

        self.t2_vols  = None if t2_vols  is None else torch.tensor(t2_vols,  dtype=torch.float32)
        self.hbv_vols = None if hbv_vols is None else torch.tensor(hbv_vols, dtype=torch.float32)
        self.adc_vols = None if adc_vols is None else torch.tensor(adc_vols, dtype=torch.float32)

        self.transform = transform

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, idx):
        inputs = {}

        # ----- clinical -----
        if self.has_clinical:
            inputs["clinical"] = self.X[idx]  # (num_features,)

        # ----- MRI volumes -----
        subject_dict = {}
        if self.t2_vols is not None:
            # (D, H, W) -> (1, D, H, W)
            t2 = self.t2_vols[idx].unsqueeze(0)
            subject_dict["t2"] = tio.ScalarImage(tensor=t2)

        if self.hbv_vols is not None:
            hbv = self.hbv_vols[idx].unsqueeze(0)
            subject_dict["hbv"] = tio.ScalarImage(tensor=hbv)

        if self.adc_vols is not None:
            adc = self.adc_vols[idx].unsqueeze(0)
            subject_dict["adc"] = tio.ScalarImage(tensor=adc)

        if len(subject_dict) > 0:
            subject = tio.Subject(**subject_dict)

            if self.transform is not None:
                subject = self.transform(subject)

            # after transform: each is (1, D, H, W)
            if "t2" in subject:
                inputs["t2"] = subject.t2.data
            if "hbv" in subject:
                inputs["hbv"] = subject.hbv.data
            if "adc" in subject:
                inputs["adc"] = subject.adc.data

        t = self.T[idx]
        e = self.E[idx]
        return inputs, t, e
    
    
    

def make_loaders(cfg, modalities, df, MRIs, fold_val):
    va_mask = (df["fold"].values == fold_val)
    tr_mask = ~va_mask

    MRIs_tr = {}
    MRIs_va = {}
    for key in MRIs.keys():
        MRIs_tr[key] = MRIs[key][tr_mask]
        MRIs_va[key] = MRIs[key][va_mask]


    if 'clinical' in modalities:
        X = df[cfg.data.features].astype(float).copy()

        X_tr, X_va = X.iloc[tr_mask], X.iloc[va_mask]
        
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_va = scaler.transform(X_va)
    else:
        X_tr = None
        X_va = None
        
        
    T = pd.to_numeric(df["time_to_follow-up/BCR"], errors="coerce").values.astype(float)
    E = pd.to_numeric(df["BCR"], errors="coerce").values.astype(int)
    T[T <= 0] = 0.1
    T_tr, T_va = T[tr_mask], T[va_mask]
    E_tr, E_va = E[tr_mask], E[va_mask]
        

    train_transform = None
    if cfg.data.aug:
        train_transform = tio.Compose([
        # Geometric
            tio.RandomFlip(axes=('IS',), flip_probability=0.5),  # left-right
            tio.RandomAffine(
                scales=(0.8, 1.2),           # small scaling
                degrees = 0,
                translation=(4, 10, 10),     # small translations
                isotropic=False,
                image_interpolation='linear',
                p=0.6
            ),

            # Intensity
            tio.RandomGamma(log_gamma=(-0.4, 0.4), p=0.6),
            tio.RandomNoise(std=(0.0, 0.02), p=0.3),

            tio.RandomElasticDeformation(num_control_points=5, max_displacement=3, p=0.4),
        ])

    val_transform = None   # usually no augmentation for validation

    ds_tr = MultimodalSurvDataset(T_tr, E_tr,
                        X=X_tr,
                        t2_vols=MRIs_tr['t2'] if 't2' in modalities.keys() else None,
                        hbv_vols=MRIs_tr['hbv'] if 'hbv' in modalities.keys() else None,
                        adc_vols=MRIs_tr['adc'] if 'adc' in modalities.keys() else None,
                        transform=train_transform)
    
    ds_va = MultimodalSurvDataset(T_va, E_va,
                        X=X_va,
                        t2_vols=MRIs_va['t2'] if 't2' in modalities.keys() else None,
                        hbv_vols=MRIs_va['hbv'] if 'hbv' in modalities.keys() else None,
                        adc_vols=MRIs_va['adc'] if 'adc' in modalities.keys() else None,
                        transform=val_transform)

    # shuffle train, no shuffle val
    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=cfg.train.batch_size, shuffle=True, drop_last=False, num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)
    ld_va = torch.utils.data.DataLoader(ds_va, batch_size=cfg.train.batch_size, shuffle=False, drop_last=False, num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)
    return ld_tr, ld_va, (T_va, E_va)  # keep raw arrays for C-index