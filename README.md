# Brain MRI Analysis with Harvard-Oxford Atlas

This project analyzes differences in brain MRI scans across two time points (baseline and follow-up). Using the Harvard-Oxford cortical atlas, it identifies brain regions with significant changes and provides statistical insights.

## Features

- Preprocesses MRI images (resampling, smoothing).
- Computes voxel-wise differences between baseline and follow-up scans.
- Maps changes to brain regions using the Harvard-Oxford atlas.
- Outputs a ranked list of regions by mean change.
- Supports modular and Object-Oriented Programming (OOP) design for extensibility.

## Project Structure
├── Data/
│   ├── sub-213/
│   │   ├── ses-BL/anat/sub-213_ses-BL_T1w.nii.gz
│   │   ├── ses-FU/anat/sub-213_ses-FU_T1w.nii.gz
│   ├── sub-222/
│       ├── ses-BL/anat/sub-222_ses-BL_T1w.nii.gz
│       ├── ses-FU/anat/sub-222_ses-FU_T1w.nii.gz
├── Scripts/
│   ├── atlas_processing.py
│   ├── image_processing.py
├── results/
│   ├── sub-213_results.txt
│   ├── sub-222_results.txt
├── README.md
└── .gitignore

1.	Atlas Loading
The Harvard-Oxford cortical atlas is used to map MRI voxels to specific brain regions.
	2.	Image Processing
	•	MRI images are resampled to align with the atlas.
	•	A Gaussian smoothing filter is applied to reduce noise.
	3.	Difference Computation
	•	Voxel-wise differences between baseline and follow-up images are calculated.
	4.	Region Analysis
	•	Changes are aggregated by brain region using the atlas.
	•	A ranked list of regions with mean changes is generated.

License

This project is open-source and available under the MIT License.