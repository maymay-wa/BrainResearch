import os
import sys
import pytest
import pandas as pd
import numpy as np
import nibabel as nib
from unittest.mock import MagicMock
import ants
from nilearn.image import math_img, resample_to_img, get_data, load_img

# Add the src directory to sys.path dynamically
repo_root = os.path.dirname(os.path.abspath(__file__))  
src_path = os.path.join(repo_root, "..", "src")         
sys.path.append(src_path)
from DataPipe import DataPipe

subject = 112
baseLine = 'output/registered_output_sub-' + str(subject) + '_ses-BL.nii.gz'
followUp = 'output/registered_output_sub-' + str(subject) + '_ses-FU.nii.gz'
dummy_participants_tsv = "Data/participants.tsv"
dp = DataPipe()
dp.ants_img = ants.image_read(followUp)
dp.load_img = load_img(followUp)
dp.get_data = get_data(dp.load_img)

def test_datapipe_init():
    """Tests if DataPipe initializes correctly and loads the participants file."""
    assert isinstance(dp.participants_df, pd.DataFrame)
    assert "participant_id" in dp.participants_df.columns
    assert hasattr(dp, "ATLAS_IMAGE")
    assert hasattr(dp, "LABEL_NAMES")
    assert hasattr(dp, "REGION_LABELS")


def test_register_and_convert_to_nifti():
    """Tests if the function runs without errors and returns a file."""
    # Mock outputs
    out_nifti = dp.register_and_convert_to_nifti(
        fixed_image=dp.ants_img,
        moving_image=dp.ants_img,
        subject_id=112,
        session="BL",
        transform_type="Affine"
    )

    assert out_nifti == ants.image_read('output/registered_output_sub-' + str(subject) + '_ses-FU.nii.gz')


def test_loadImage():
    """Tests if loadImage reads and processes an image correctly."""
    assert dp.load_img is not None, "loadImage() should return a valid object"
    assert hasattr(dp.load_img, "shape"), "The returned image should have a 'shape' attribute"
    assert len(dp.load_img.shape) == 3, "Image should be a 3D volume"
    assert dp.load_img.shape == dp.ATLAS_IMAGE.shape


def test_process_all_subjects():
    """Tests if all subjects are processed correctly."""
    # Ensure participants DataFrame is loaded
    assert dp.participants_df is not None, "Participants DataFrame should be initialized"
    assert not dp.participants_df.empty, "Participants DataFrame should not be empty"
    assert "participant_id" in dp.participants_df.columns, "Participants DataFrame should contain 'participant_id' column"
    # Run the function
    dp.process_all_subjects()
    # Check that a processed output exists (mocked validation)
    assert hasattr(dp, "processed_subjects"), "DataPipe should have a processed_subjects attribute after processing"
    assert isinstance(dp.processed_subjects, list), "processed_subjects should be a list"
    assert len(dp.processed_subjects) > 0, "processed_subjects list should not be empty"


def test_display_brain_and_difference():
    """Tests if brain images are displayed correctly."""
    
    # Ensure baseline and follow-up images exist
    assert os.path.exists(baseLine), f"Baseline image '{baseLine}' does not exist"
    assert os.path.exists(followUp), f"Follow-up image '{followUp}' does not exist"

    # Load images to check their validity
    baseline_img = dp.load_img(baseLine)
    followup_img = dp.load_img(followUp)

    assert baseline_img is not None, "Baseline image should be loaded successfully"
    assert followup_img is not None, "Follow-up image should be loaded successfully"
    assert hasattr(baseline_img, "shape"), "Baseline image should have a 'shape' attribute"
    assert hasattr(followup_img, "shape"), "Follow-up image should have a 'shape' attribute"
    assert baseline_img.shape == followup_img.shape, "Baseline and follow-up images should have the same dimensions"

    # Run function
    dp.display_brain_and_difference(baseLine, followUp)

    # Mocked validation: Ensure images were processed
    assert hasattr(dp, "displayed_images"), "DataPipe should have a displayed_images attribute after displaying"
    assert isinstance(dp.displayed_images, list), "displayed_images should be a list"
    assert len(dp.displayed_images) == 2, "displayed_images should contain both baseline and follow-up"


def main():
    """Runs all the tests."""
    test_result = pytest.main([os.path.abspath(__file__)])
    sys.exit(test_result)

if __name__ == "__main__":
    main()