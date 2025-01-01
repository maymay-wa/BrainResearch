import os
from tabulate import tabulate
from nilearn import plotting
from nilearn import image
from nilearn.image import math_img, resample_to_img, get_data
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.plotting import plot_stat_map, view_img
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import ants
import nibabel as nib

# Fetch the Harvard-Oxford atlas
ATLAS = fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
# Load the atlas image
ATLAS_IMAGE = image.load_img(ATLAS.maps)
ATLAS_DATA = get_data(ATLAS_IMAGE)
# Extract region names from the atlas
LABEL_NAMES = ATLAS['labels']
# Extract unique region labels from the atlas
REGION_LABELS = np.unique(ATLAS_DATA)

def get_subject_file_pairs(data_dir, subjectDF):
    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)

        # Skip non-directories or irrelevant files
        if not os.path.isdir(subject_path) or subject.startswith('.') or subject == 'derivatives':
            continue

        for index, row in subjectDF.iterrows():
            subject = row['participant_id']
            subject_path = os.path.join(data_dir, f"sub-{subject:03d}")

            # Locate BL and FU files
            baseline_file = os.path.join(subject_path, 'ses-BL', 'anat', f"sub-{subject:03d}_ses-BL_T1w.nii.gz")
            followup_file = os.path.join(subject_path, 'ses-FU', 'anat', f"sub-{subject:03d}_ses-FU_T1w.nii.gz")

            # Verify that both files exist
            if os.path.exists(baseline_file) and os.path.exists(followup_file):
                # Add the file paths to the DataFrame
                subjectDF.at[index, 'Baseline File Path'] = 'Data/' + os.path.relpath(baseline_file, data_dir)
                subjectDF.at[index, 'Followup File Path'] = 'Data/' + os.path.relpath(followup_file, data_dir)
            else:
                print(f"Skipping subject {subject}: missing files.")

def nifti_to_ants(nifti_img):
    # Extract voxel data, spacing, and origin from the NIfTI image
    voxel_data = nifti_img.get_fdata()  # Get voxel data
    affine = nifti_img.affine  # Get affine matrix
    spacing = tuple(np.linalg.norm(affine[:3, :3], axis=0))  # Compute voxel spacing
    origin = tuple(affine[:3, 3])  # Extract origin (translation)
    
    # Convert to ANTs image
    ants_image = ants.from_numpy(voxel_data, origin=origin, spacing=spacing)
    return ants_image

def register_and_convert_to_nifti(fixed_image, moving_image, transform_type='Rigid'):
    # Perform the registration
    registered = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=transform_type
    )['warpedmovout']

    # Extract voxel data and spacing from the registered image
    voxel_data = registered.numpy()  # Voxel data as NumPy array
    spacing = registered.spacing  # Voxel spacing (dx, dy, dz)
    origin = registered.origin  # Image origin (x, y, z)
    
    # Construct a 4x4 affine matrix
    affine = np.eye(4)  # Identity matrix
    affine[:3, :3] = np.diag(spacing)  # Set voxel spacing
    affine[:3, 3] = origin  # Set translation (origin)
    # Create and return a NIfTI image
    return nib.Nifti1Image(voxel_data, affine)

def loadImage(imgPath):
    atlas_nifti = ATLAS['maps']
    # Convert the atlas NIfTI to an ANTs image
    atlas_image = nifti_to_ants(atlas_nifti)
    # Load the moving image and convert to an ANTs image
    img_nifti = nib.load(imgPath)
    img_image = nifti_to_ants(img_nifti)

    # Register and convert to NIfTI
    nifti = register_and_convert_to_nifti(fixed_image=atlas_image, moving_image=img_image)

    # Resample and smooth the NIfTI image
    img_resampled = resample_to_img(
        source_img=nifti,
        target_img=ATLAS_IMAGE,
        force_resample=True,
        copy_header=True,
        interpolation='nearest'
    )
    img = image.smooth_img(img_resampled, fwhm=6)
    return img

def findDifferingAreas(index, df, img1, img2):
    # Creates the difference map
    diff = image.math_img("img1 - img2", img1=img1, img2=img2)
    diff_data = get_data(diff)
    # Loop through each region to compute statistics
    for label in REGION_LABELS:
        regionMask = (ATLAS_DATA == label)
        regionChanges = diff_data[regionMask]
        region_mean = round(float(np.mean(regionChanges)), 2)
        if 0 <= label < len(LABEL_NAMES):
            region_name = LABEL_NAMES[label] 
        else:
            region_name = "Unknown Region"
        # Insert the region_mean into the correct column and row in the DataFrame
        df.loc[index, f"{region_name} Change"] = region_mean

participants_df = pd.read_csv('Data/participants.tsv', sep="\t")
# Add columns for file paths
filePaths = ['Baseline File Path', 'Followup File Path']
for label in filePaths:
    participants_df[label] = None
get_subject_file_pairs('Data', participants_df)
for label in LABEL_NAMES:
    participants_df[label + ' Change'] = None

for index, row in participants_df.iterrows():
    baseLinePath = participants_df.loc[index,'Baseline File Path']
    followUpPath = participants_df.loc[index,'Followup File Path']
    baseLine = loadImage(baseLinePath)
    followUp = loadImage(followUpPath)
    findDifferingAreas(index, participants_df, baseLine, followUp)
    
participants_df = participants_df.drop(columns=['Baseline File Path', 'Followup File Path'])

import subprocess
output_excel_path = 'participants_with_changes_test.xlsx'
participants_df.to_excel(output_excel_path, index=False)
subprocess.run(["open", output_excel_path])

participants_df['avg cudit'] = (participants_df['cudit total baseline'] + participants_df['cudit total follow-up']) / 2
cudit_columns = ['avg cudit']
brain_region_columns = [col for col in participants_df.columns if 'Change' in col]
# Combine into one dataset
correlation_data = participants_df[cudit_columns + brain_region_columns]
# Compute correlations
correlation_matrix = correlation_data.corr()
cudit_correlations = correlation_matrix.loc[cudit_columns, brain_region_columns]
#cudit_correlations.T.sort_values(by='avg cudit', ascending=False)