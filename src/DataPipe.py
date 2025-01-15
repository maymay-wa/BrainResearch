import os
import ants
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_anat, plot_img
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import math_img, resample_to_img, get_data, load_img

class DataPipe:
    def __init__(self, data_dir='Data', participants_tsv='Data/participants.tsv'):
        """
        Constructor for the Data class.
        data_dir : str
            The root directory containing subject data.
        participants_tsv : str
            Path to the participants.tsv file.
        """
        # Sets the working directory to be where the file is found
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        self.data_dir = os.path.abspath(data_dir)
        self.participants_tsv = os.path.abspath(participants_tsv)

        self.data_dir = data_dir
        self.participants_tsv = participants_tsv
        # Load participants DataFrame
        self.participants_df = pd.read_csv(self.participants_tsv, sep="\t")
        # Fetch atlas upon initialization
        self.fetch_harvard_oxford_atlas()
        # Prepare columns in participants_df for file paths and volumetric data
        self.prepare_dataframe_columns()

    def fetch_harvard_oxford_atlas(self):
        """
        Fetches and loads the Harvard-Oxford atlas, storing relevant attributes.
        """
        # Fetch the Harvard-Oxford atlas
        try:
            ATLAS = fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
            # Load the atlas image
            self.ATLAS_MAPS = ATLAS.maps
            self.ATLAS_IMAGE = load_img(self.ATLAS_MAPS)
            self.ATLAS_DATA = get_data(self.ATLAS_IMAGE)
            # Extract region names from the atlas
            self.LABEL_NAMES = ATLAS['labels']
            # Extract unique region labels from the atlas
            self.REGION_LABELS = np.unique(self.ATLAS_DATA)
        except Exception as e:
            logging.error(f"Error fetching/loading Harvard-Oxford atlas: {e}")
            raise  # In many cases, you'd want to stop execution if the atlas isn't available.

    def prepare_dataframe_columns(self):
        """
        Adds columns to the participants DataFrame for baseline/follow-up file 
        paths and the volumetric data for each region.
        """
        # Prepare the new columns for file paths
        file_cols = ['Baseline File Path', 'Followup File Path']
        new_cols = {col: [None] * len(self.participants_df) for col in file_cols}
        
        # Prepare columns for volume and difference metrics for each region
        for label in self.LABEL_NAMES:
            # Skip background or unknown labels
            if label in ("Background", "Unknown Region"):
                continue
            new_cols[f"{label} Volume Avg"] = [None]*len(self.participants_df)
            new_cols[f"{label} Volume Change"] = [None]*len(self.participants_df)
            new_cols[f"{label} Change"] = [None]*len(self.participants_df)

        # Create an empty DataFrame and horizontally concatenate
        columns_df = pd.DataFrame(new_cols, index=self.participants_df.index)
        self.participants_df = pd.concat([self.participants_df, columns_df], axis=1)

    def get_subject_file_pairs(self):
        """
        Given a DataFrame of participants and the data directory,
        locate subject baseline (BL) and follow-up (FU) files
        and store them in the DataFrame columns.
        """
        # Iterate through the data directory
        for subject_folder in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject_folder)
            
            # Skip non-directories or irrelevant files
            if (
                not os.path.isdir(subject_path) 
                or subject_folder.startswith('.') 
                or subject_folder == 'derivatives'
            ):
                continue

        # Now match participants to BL/FU files
        for index, row in self.participants_df.iterrows():
            subject = row['participant_id']
            subject_path = os.path.join(self.data_dir, f"sub-{subject:03d}")
            
            # Locate BL and FU files
            baseline_file = os.path.join(
                subject_path, 
                'ses-BL', 
                'anat', 
                f"sub-{subject:03d}_ses-BL_T1w.nii.gz"
            )
            followup_file = os.path.join(
                subject_path, 
                'ses-FU', 
                'anat', 
                f"sub-{subject:03d}_ses-FU_T1w.nii.gz"
            )

            # Verify that both files exist
            if os.path.exists(baseline_file) and os.path.exists(followup_file):
                # Add the file paths to the DataFrame
                self.participants_df.at[index, 'Baseline File Path'] = (
                    'Data/' + os.path.relpath(baseline_file, self.data_dir)
                )
                self.participants_df.at[index, 'Followup File Path'] = (
                    'Data/' + os.path.relpath(followup_file, self.data_dir)
                )
            else:
                print(f"Skipping subject {subject}: missing files.")

    def register_and_convert_to_nifti(self, fixed_image, moving_image, subject_id, session, transform_type='Affine'):
        """
        Registers `moving_image` to `fixed_image` using ANTs, writes out the
        transformed result to disk as a NIfTI file, and returns the nibabel image.
        """
        out_path = f"output/registered_output_sub-{subject_id}_ses-{session}.nii.gz"
        
        # If the file already exists, load it and skip registration
        if os.path.exists(out_path):
            try:
                file = nib.load(out_path)
                return file
            except Exception as e:
                logging.error(f"Error loading existing registered image for subject {subject_id}, session {session}: {e}")
                raise
        
        # Preprocess the moving image (bias field correction + smoothing)
        moving_image = ants.n4_bias_field_correction(moving_image)
        moving_image = ants.smooth_image(moving_image, 2)

        # Registration
        try:
            outputImage = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform=transform_type
            )['warpedmovout']
        except Exception as e:
            logging.error(f"Error registering images for subject {subject_id}, session {session}: {e}")
            raise

        # Write out the registered result
        ants.image_write(outputImage, out_path)
        
        # Load the resulting file as a nibabel image
        try:
            file = nib.load(out_path)
            return file
        except Exception as e:
            logging.error(f"Error loading registered image from disk for subject {subject_id}, session {session}: {e}")
            raise

    def loadImage(self, imgPath, subject_id, session):
        """
        Reads an image from a given path, then registers it against the
        Harvard-Oxford atlas, returning a NIfTI image resampled to the atlas space.
        """
        # Path to your local Harvard-Oxford atlas file if needed:
        # (Should match what's inside your `fetch_atlas_harvard_oxford` location)
        atlasPath = '/Users/mayerunterberg/nilearn_data/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'
        atlas_image = ants.image_read(atlasPath)
        ants_img = ants.image_read(imgPath)
        
        # Register and convert to NIfTI
        nifti = self.register_and_convert_to_nifti(
            fixed_image=atlas_image,
            moving_image=ants_img,
            subject_id=subject_id,
            session=session
        )

        # Resample the result to the official atlas image just in case
        # (the loaded self.ATLAS_IMAGE might be the same, but we do this 
        #  to ensure consistent shape/resolution)
        img_resampled = resample_to_img(
            source_img=nifti,
            target_img=self.ATLAS_IMAGE,
            force_resample=True,
            copy_header=True,
            interpolation='nearest'
        )
        return img_resampled

    def findDifferingAreasAndVolume(self, index, img1, img2, threshold=None):
        """
        Given two images, computes the difference map, calculates mean difference
        and volume changes per region, and records those metrics in the DataFrame.
        """
        diff = math_img("img1 - img2", img1=img1, img2=img2)
        diff_data = get_data(diff)
        
        # For volume calculations, get data for each image
        img1_data = get_data(img1)
        img2_data = get_data(img2)
        
        # Use the ATLAS_IMAGE affine to find voxel volume
        voxel_sizes = np.abs(np.diag(self.ATLAS_IMAGE.affine)[:3])  # shape (3,)
        voxel_volume = np.prod(voxel_sizes)  # in mm^3

        for label in self.REGION_LABELS:
            # region_name from self.LABEL_NAMES
            if 0 <= label < len(self.LABEL_NAMES):
                region_name = self.LABEL_NAMES[label]
            else:
                region_name = "Unknown Region"
                continue
            
            if region_name in ("Background", "Unknown Region"):
                continue

            regionMask = (self.ATLAS_DATA == label)
            
            # Mean difference in region
            regionChanges = diff_data[regionMask]
            region_mean_diff = round(float(np.mean(regionChanges)), 2)
            
            # Volume difference (count of non-zero or above-threshold voxels)
            if threshold is not None:
                img1_masked = ((img1_data > threshold) & regionMask)
                img2_masked = ((img2_data > threshold) & regionMask)
            else:
                img1_masked = ((img1_data != 0) & regionMask)
                img2_masked = ((img2_data != 0) & regionMask)
                
            region_img1_voxels = np.count_nonzero(img1_masked)
            region_img2_voxels = np.count_nonzero(img2_masked)
            
            region_img1_volume = region_img1_voxels * voxel_volume
            region_img2_volume = region_img2_voxels * voxel_volume
            region_volume_diff = round(region_img1_volume - region_img2_volume, 2)
            avgVolume = round((region_img1_volume + region_img2_volume) / 2, 2)
            
            # Save to the DataFrame
            self.participants_df.loc[index, f"{region_name} Change"] = region_mean_diff
            self.participants_df.loc[index, f"{region_name} Volume Avg"] = avgVolume
            self.participants_df.loc[index, f"{region_name} Volume Change"] = region_volume_diff

    def process_all_subjects(self):
        """
        Main loop to process each subject in the participants DataFrame:
        1. Load baseline and follow-up files.
        2. Register them to the atlas space.
        3. Compute differences in each region and update participants_df.
        """
        for idx, row in self.participants_df.iterrows():
            baseLinePath = row['Baseline File Path']
            followUpPath = row['Followup File Path']
            if pd.isna(baseLinePath) or pd.isna(followUpPath):
                continue  # skip if missing file
            
            subject_id = row['participant_id']
            baseLine = self.loadImage(baseLinePath, subject_id, 'BL')
            followUp = self.loadImage(followUpPath, subject_id, 'FU')
            
            self.findDifferingAreasAndVolume(idx, baseLine, followUp)

    def display_brain_and_difference(self, baseline_file, followup_file):
        """
        Displays the brain images for baseline, follow-up, and their difference.
        """
        baseline_img = nib.load(baseline_file)
        followup_img = nib.load(followup_file)

        baseline_data = baseline_img.get_fdata()
        followup_data = followup_img.get_fdata()
        diff_data = followup_data - baseline_data
        diff_img = nib.Nifti1Image(diff_data, affine=baseline_img.affine)

        # Plot the difference map
        plot_img(diff_img, title="Difference MRI", cmap='coolwarm', colorbar=True)
        plt.show()

        # Plot the follow-up and baseline images
        plot_anat(followup_img, title="Follow-Up MRI")
        plot_anat(baseline_img, title="Baseline MRI")
        plt.show()
    
    def display_before_registry(self, baseline_file, followup_file):
        """
        Displays the brain images for baseline, follow-up, and their difference.
        """
        baseline_img = nib.load(baseline_file)
        followup_img = nib.load(followup_file)

        # Plot the follow-up and baseline images
        plot_anat(followup_img, title="Follow-Up MRI Before Registry")
        plot_anat(baseline_img, title="Baseline MRI Before Registry")
        plt.show()