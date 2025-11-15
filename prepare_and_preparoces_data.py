from glob import glob
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_mha(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X)
    return img, arr


clinical_data_paths = glob('./clinical_data/*.json')

# Read all JSON files and combine into a single DataFrame
data_list = []
for file_path in clinical_data_paths:
    with open(file_path, 'r') as f:
        data = json.load(f)
        data = {'patient_id': int(os.path.basename(file_path).replace('.json', '')), **data}
        data_list.append(data)

clinical_df = pd.DataFrame(data_list)
#corss reference with data split file (data_split_5fold.csv)
clinical_df = pd.merge(pd.read_csv('./data_split_5fold.csv'), clinical_df, on='patient_id', how='inner')


# A function to fill missing values using KNN Imputer:
def fill_missing_values(df, column_to_fill, important_columns, n_neighbors=5, add_missing_indicator=True, is_int=True):
    imputer = KNNImputer(n_neighbors=3)
    # create missing indicator
    if add_missing_indicator:
        df[f"{column_to_fill}_missing"] = df[column_to_fill].isna().astype(int)

    # important columns for predicting positive_lymph_nodes values
    cols = [column_to_fill] + important_columns

    X = df[cols].copy()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    imputer  = KNNImputer(n_neighbors=5, weights="distance")
    X_imp    = imputer.fit_transform(X_scaled)

    # IMPORTANT: back to original scale first
    X_imp_unscaled = scaler.inverse_transform(X_imp)
    X_imp_df = pd.DataFrame(X_imp_unscaled, columns=cols, index=df.index)

    # now round on the original (0..1) scale
    if is_int:
        df[column_to_fill] = (
            X_imp_df[column_to_fill].round().astype(int)
    )   

    return df

# Tertiary → less common pattern, only recorded if present, so we fill NaN with 0 when not present
clinical_df["tertiary_gleason"] = clinical_df["tertiary_gleason"].fillna(0)
# Distribution is long-tailed to the right, therefore we apply a log transformation
clinical_df["pre_operative_PSA_log"] = np.log1p(clinical_df["pre_operative_PSA"])

# Chagne BCR values to numeric
clinical_df["BCR"] = pd.to_numeric(clinical_df["BCR"], errors="coerce")
# BCR == 0 → BCR_PSA == 0 no recurrence, PSA never rose again, removed NaN.
clinical_df.loc[clinical_df["BCR"] == 0.0, "BCR_PSA"] = 0.0
#Because pathologic T stages are ordered by disease severity, we can safely treat trandorm them as ordinal
pT_mapping = {
    "2": 2.0, "2a": 2.25, "2b": 2.5, "2c": 2.75,
    "3": 3.0, "3a": 3.25, "3b": 3.5, "3c": 3.5,
    "4": 4.0, "4a": 4.0,  "4b": 4.0, "4c": 4.0, 
} # There are no 4a, 4b, 4c, or 3c in the pT standars stages, (might be a typo), therefore map them to the same value as 4 or 3b respectively.
clinical_df["pT_stage_num"] = clinical_df["pT_stage"].map(pT_mapping)

# Fill unknown values for positive_lymph_nodes using KNN imputer
# map to 0/1/nan
clinical_df['positive_lymph_nodes'] = clinical_df['positive_lymph_nodes'].map({"0":0, "1":1, "x":np.nan})
if clinical_df['positive_lymph_nodes'].isna().any():
    clinical_df = fill_missing_values(
        clinical_df,
        column_to_fill="positive_lymph_nodes",
        important_columns=[
                        "primary_gleason",
                        "secondary_gleason",
                        "tertiary_gleason",
                        "pT_stage_num",
                        "ISUP",
                        "pre_operative_PSA_log"],
        n_neighbors=5,
        is_int=True
    )
    
# Fill unknown values for positive_lymph_nodes using KNN imputer    
clinical_df['invasion_seminal_vesicles'] = clinical_df['invasion_seminal_vesicles'].map({"0":0, "1":1, "x":np.nan})
if clinical_df['invasion_seminal_vesicles'].isna().any():
    clinical_df = fill_missing_values(
        clinical_df,
        column_to_fill="invasion_seminal_vesicles",
        important_columns=["pT_stage_num",
                        "ISUP",
                        "primary_gleason",
                        "secondary_gleason",
                        "tertiary_gleason",
                    ],
        n_neighbors=5,
        is_int=True
    )
    
    
    
# Fill unknown values for positive_lymph_nodes using KNN imputer    
clinical_df['capsular_penetration'] = clinical_df['capsular_penetration'].map({"0":0, "1":1, "x":np.nan})
if clinical_df['capsular_penetration'].isna().any():
    clinical_df = fill_missing_values(
        clinical_df,
        column_to_fill="capsular_penetration",
        important_columns=["pT_stage_num",
                        "ISUP",
                        "primary_gleason",
                        "secondary_gleason",
                        "tertiary_gleason",
                        "pre_operative_PSA_log",
                        "invasion_seminal_vesicles",
                    ],
        n_neighbors=5,
        is_int=True
    )

# For positive_surgical_margins, all values are known (0, and 1), and ther are no '2' values which corresponds to unknown. But for safety, we check check that too.
clinical_df['positive_surgical_margins'] = clinical_df['positive_surgical_margins'].map({0:0, 1:1, 2:np.nan})
if clinical_df['positive_surgical_margins'].isna().any():
    clinical_df = fill_missing_values(
        clinical_df,
        column_to_fill="positive_surgical_margins",
        important_columns=["pT_stage_num",
                        "ISUP",
                        "primary_gleason",
                        "secondary_gleason",
                        "tertiary_gleason",
                        "pre_operative_PSA_log",
                        "invasion_seminal_vesicles",
                    ],
        n_neighbors=5,
        is_int=True
    )


# Chagne lymphovascular_invasion values to numeric
clinical_df["lymphovascular_invasion"] = pd.to_numeric(clinical_df["lymphovascular_invasion"], errors="coerce")



# Add three columns for earlier_therapy, showing radiotherapy, hormone_therapy, and cryotherapy
therapy_keywords = ["radiotherapy", "cryotherapy", "hormones"]
for t in therapy_keywords:
    clinical_df[t] = clinical_df["earlier_therapy"].str.contains(t, na=False).astype(int)
clinical_df.loc[clinical_df["earlier_therapy"] == "unknown", therapy_keywords] = np.nan
for t in therapy_keywords:
    if clinical_df[t].isna().any():
        clinical_df = fill_missing_values(
            clinical_df,
            column_to_fill=t,
            important_columns=["pT_stage_num",
                            "ISUP",
                            "primary_gleason",
                            "secondary_gleason",
                            "tertiary_gleason",
                            "pre_operative_PSA_log",
                            "invasion_seminal_vesicles",
                        ],
            n_neighbors=5,
            is_int=True
        )

clinical_df.to_csv('./data/clinical_data_processed.csv')

for col in clinical_df.columns:
    print('=**='*20)
    print(col, ": ")
    print(clinical_df[col].dtype, clinical_df[col].nunique(), clinical_df[col].isna().sum())
    if clinical_df[col].nunique() <= 10:
        print(clinical_df[col].value_counts())
        




def resample_to_reference(moving_img, reference_img, is_label=False):
    """
    moving_img: SimpleITK.Image (e.g. ADC, HBV, mask)
    reference_img: SimpleITK.Image (e.g. T2)
    is_label: True for segmentation masks (use NN), False for images (use linear)
    """
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(interp)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(moving_img)
    return resampled


# FUNCTION 1 (From previous step)
def resample_to_spacing(image, target_spacing=(0.5, 0.5, 3.0), is_label=False):
    """
    Resamples a SimpleITK image to a target spacing, PRESERVING its physical extent.
    """
    
    if is_label:
        interpolator = sitk.sitkNearestNeighbor
        default_pixel_value = 0
    else:
        interpolator = sitk.sitkLinear
        default_pixel_value = 0.0

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    original_size_np = np.array(original_size)
    original_spacing_np = np.array(original_spacing)
    target_spacing_np = np.array(target_spacing)

    new_size_np = original_size_np * (original_spacing_np / target_spacing_np)
    new_size = [int(round(s)) for s in new_size_np]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(sitk.Transform())
    
    resampled_image = resampler.Execute(image)
    return resampled_image

# FUNCTION 2 (The alignment function you were using)
def resample_to_reference(image, reference_image, is_label=False):
    """
    Resamples a SimpleITK image to match the grid of a reference image.
    Uses ResampleImageFilter's SetReferenceImage for robustness.
    """
    
    if is_label:
        interpolator = sitk.sitkNearestNeighbor
        default_pixel_value = 0 # Mask background
    else:
        interpolator = sitk.sitkLinear
        default_pixel_value = 0.0 # Image background

    resampler = sitk.ResampleImageFilter()
    
    # This is the key: it copies spacing, size, origin, and direction
    resampler.SetReferenceImage(reference_image) 
    
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetOutputPixelType(image.GetPixelID()) # Use the input image's pixel type
    
    resampled_image = resampler.Execute(image)
    return resampled_image

def crop_or_pad_to_size(image, target_size=(32, 256, 256), pad_value=0):

    assert image.ndim == 3, "Input must be a 3D array"

    D, H, W = image.shape
    TD, TH, TW = target_size

    # if image is larger than target size, crop it
    for dim in range(3):
        if image.shape[dim] > target_size[dim]:
            start_idx = (image.shape[dim] - target_size[dim]) // 2
            end_idx = start_idx + target_size[dim]
            if dim == 0:
                image = image[start_idx:end_idx, :, :]
            elif dim == 1:
                image = image[:, start_idx:end_idx, :]
            elif dim == 2:
                image = image[:, :, start_idx:end_idx]

    # compute padding sizes
    pad_d = max(TD - D, 0)
    pad_h = max(TH - H, 0)
    pad_w = max(TW - W, 0)

    # symmetric padding: split into left/right
    pad_before_d = pad_d // 2
    pad_after_d  = pad_d - pad_before_d

    pad_before_h = pad_h // 2
    pad_after_h  = pad_h - pad_before_h

    pad_before_w = pad_w // 2
    pad_after_w  = pad_w - pad_before_w

    padded = np.pad(
        image,
        pad_width=(
            (pad_before_d, pad_after_d),
            (pad_before_h, pad_after_h),
            (pad_before_w, pad_after_w)
        ),
        mode='constant',
        constant_values=pad_value
    )

    return padded



target_size = (32, 256, 256)
spacing = (0.3, 0.3, 3.0)
out_dir = './data/preprcoessed_mpMRI'

for patient_id in tqdm(clinical_df['patient_id'].tolist()):

    t2_path = f"./radiology/mpMRI/{patient_id}/{patient_id}_0001_t2w.mha"
    hbv_path = f"./radiology/mpMRI/{patient_id}/{patient_id}_0001_hbv.mha"
    adc_path = f"./radiology/mpMRI/{patient_id}/{patient_id}_0001_adc.mha"
    mask_path = f"./radiology/prostate_mask_t2w/{patient_id}_0001_mask.mha"

    # load all
    t2_img  = sitk.ReadImage(t2_path)
    adc_img = sitk.ReadImage(adc_path)
    hbv_img = sitk.ReadImage(hbv_path)
    mask_img = sitk.ReadImage(mask_path)

    # resample ADC/HBV/mask to T2 space
    t2_rs  = resample_to_spacing(t2_img, spacing, is_label=False)
    adc_rs  = resample_to_reference(adc_img,  t2_rs, is_label=False)
    hbv_rs  = resample_to_reference(hbv_img,  t2_rs, is_label=False)
    mask_rs = resample_to_reference(mask_img, t2_rs, is_label=True)

    # convert to numpy
    t2_arr  = sitk.GetArrayFromImage(t2_rs)
    adc_arr = sitk.GetArrayFromImage(adc_rs)
    hbv_arr = sitk.GetArrayFromImage(hbv_rs)
    mask_arr = sitk.GetArrayFromImage(mask_rs)

    t2_arr[mask_arr==0] = 0
    adc_arr[mask_arr==0] = 0
    hbv_arr[mask_arr==0] = 0
    
    min_idx = np.array(np.where(mask_arr > 0)).min(axis=1)
    max_idx = np.array(np.where(mask_arr > 0)).max(axis=1)

    t2_arr = t2_arr[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
    adc_arr = adc_arr[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
    hbv_arr = hbv_arr[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
    mask_arr = mask_arr[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
    
    t2_arr = crop_or_pad_to_size(t2_arr, target_size=target_size)
    adc_arr = crop_or_pad_to_size(adc_arr, target_size=target_size)
    hbv_arr = crop_or_pad_to_size(hbv_arr, target_size=target_size)
    mask_arr = crop_or_pad_to_size(mask_arr, target_size=target_size)

    t2_arr = np.clip(t2_arr/t2_arr.max(), 0, 1)
    adc_arr = np.clip(adc_arr/adc_arr.max(), 0, 1)
    hbv_arr = np.clip(hbv_arr/hbv_arr.max(), 0, 1)

    np.save(f'./{out_dir}/{patient_id}_t2.npy', t2_arr)
    np.save(f'./{out_dir}/{patient_id}_adc.npy', adc_arr)
    np.save(f'./{out_dir}/{patient_id}_hbv.npy', hbv_arr)
    np.save(f'./{out_dir}/{patient_id}_mask.npy', mask_arr)