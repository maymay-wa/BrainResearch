# **Brain MRI Analysis with Harvard-Oxford Atlas**

This project processes and analyzes differences in brain MRI scans between two time points (baseline and follow-up). By leveraging the Harvard-Oxford cortical atlas, the pipeline identifies brain regions with significant changes and ranks them based on their mean differences.

---

## **Features**
- **Image Preprocessing:** Resampling and Gaussian smoothing for noise reduction.
- **Difference Analysis:** Computes voxel-wise differences between MRI scans.
- **Atlas Integration:** Maps differences to specific brain regions using the Harvard-Oxford atlas.
- **Statistical Insights:** Outputs ranked lists of regions with the highest mean changes.
- **OOP Design:** Modular and extensible structure for easy customization.

---

## **Workflow**
### 1. **Atlas Loading**
   - The Harvard-Oxford cortical atlas is used to associate MRI voxels with specific brain regions.

### 2. **Image Preprocessing**
   - Resamples MRI images to align with the atlas.
   - Applies a Gaussian smoothing filter to reduce noise and artifacts.

### 3. **Difference Computation**
   - Calculates voxel-wise differences between baseline and follow-up images.

### 4. **Region Analysis**
   - Aggregates differences by brain region.
   - Generates a ranked list of regions with the highest mean changes.