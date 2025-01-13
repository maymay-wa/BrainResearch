import os
import ants
import logging
import xlsxwriter
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import math_img, resample_to_img, get_data, load_img
from nilearn.plotting import plot_anat, plot_img

# Fetch the Harvard-Oxford atlas
try:
    ATLAS = fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
    # Load the atlas image
    ATLAS_IMAGE = load_img(ATLAS.maps)
    ATLAS_DATA = get_data(ATLAS_IMAGE)
    # Extract region names from the atlas
    LABEL_NAMES = ATLAS['labels']
    # Extract unique region labels from the atlas
    REGION_LABELS = np.unique(ATLAS_DATA)
except Exception as e:
    logging.error(f"Error fetching/loading Harvard-Oxford atlas: {e}")
    raise  # In many cases, you'd want to stop execution if the atlas isn't available.

class Subject:
    def __init__(self, participant_id, baseline_path=None, followup_path=None):
        self.participant_id = participant_id
        self.baseline_path = baseline_path
        self.followup_path = followup_path
        self.baseline_image = None
        self.followup_image = None

    def load_images(self):
        """Load and register baseline and follow-up images."""
        if self.baseline_path:
            self.baseline_image = self._register_to_atlas(self.baseline_path, 'BL')
        if self.followup_path:
            self.followup_image = self._register_to_atlas(self.followup_path, 'FU')

    @staticmethod
    def _register_to_atlas(img_path, session):
        """Register an image to the atlas."""
        atlas_img = ants.image_read(ATLAS.maps)
        moving_img = ants.image_read(img_path)
        moving_img = ants.n4_bias_field_correction(moving_img)
        moving_img = ants.smooth_image(moving_img, 2)

        registration_result = ants.registration(
            fixed=atlas_img,
            moving=moving_img,
            type_of_transform='Affine'
        )['warpedmovout']
        return registration_result