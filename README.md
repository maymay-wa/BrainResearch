# **Brain MRI Analysis with Harvard-Oxford Atlas**

This project processes and analyzes differences in brain MRI scans between two time points (baseline and follow-up). By leveraging the Harvard-Oxford cortical atlas, the pipeline identifies brain regions with significant changes, correlates these changes with CUDIT scores, and ranks them based on their mean differences.

---

## **Features**
- **Image Preprocessing:**
  - Resamples MRI images to align with the atlas.
  - Applies Gaussian smoothing for noise reduction.
- **Difference Analysis:**
  - Computes voxel-wise differences between MRI scans.
  - Maps differences to specific brain regions using the Harvard-Oxford atlas.
- **Statistical Correlation:**
  - Calculates correlations between brain region changes and behavioral data (e.g., CUDIT scores).
  - Outputs ranked lists of regions with the strongest correlations to cannabis use patterns.
- **Visualization:**
  - Heatmaps and scatter plots to explore relationships between cannabis use and brain region changes.
- **OOP Design:**
  - Modular and extensible structure for easy customization and future expansion.

---

## **Workflow**

### 1. **Atlas Loading**
   - The Harvard-Oxford cortical atlas is used to associate MRI voxels with specific brain regions.
   - Brain regions are dynamically mapped, and differences are computed for each participant.

### 2. **Image Preprocessing**
   - MRI images are resampled to match the atlas resolution.
   - A Gaussian smoothing filter is applied to reduce noise and artifacts.

### 3. **Difference Computation**
   - Voxel-wise differences between baseline and follow-up images are calculated.
   - These differences are aggregated by brain region to compute mean changes.

### 4. **Region Analysis**
   - Differences are mapped to specific brain regions using the atlas.
   - A ranked list of brain regions with the highest mean changes is generated.

### 5. **Behavioral Correlation**
   - Average CUDIT scores are calculated for each participant.
   - Correlations between average CUDIT scores and brain region changes are computed.
   - The strongest correlations are visualized in heatmaps and scatter plots.

### 6. **Export and Visualization**
   - Results are exported to Excel for further analysis and reporting.
   - Automatically opens the Excel file after processing on macOS.

---

## **Key Code Components**
### **1. Excel Export and Visualization**
The pipeline exports the results to an Excel file, making it easy to explore and share.

diagram.py creates the diagram for the project

pDocs was used to create the documentation and diagrams was used to create the architecture diagram

data from: https://openneuro.org/datasets/ds000174/versions/1.0.1

## **Structure**
- src has the code used to run the project
- dara is expected to be stored in the Data folder in the src folder
- Further project documentation is found in the docs folder and have been copied to the src folder for convenience
- tests has tests for the main functions using the pytest module

## **Instructions**

1. Clone the repo
2. Download the relevant data from the link given above
3. Run the main.py