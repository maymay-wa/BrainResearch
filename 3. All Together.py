import os
import logging
import ants
import xlsxwriter
import numpy as np
import pandas as pd
import nibabel as nib
import unittest
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import math_img, resample_to_img, get_data, load_img
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import subprocess

logging.basicConfig(level=logging.INFO)




def get_subject_file_pairs(data_dir, subjectDF):
    """
    Given a DataFrame of participants and a data directory,
    locate subject baseline (BL) and follow-up (FU) files
    and store them in the DataFrame columns.
    """
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


def register_and_convert_to_nifti(fixed_image, moving_image, subject_id, session, transform_type='Affine'):
    """
    Registers `moving_image` to `fixed_image` using ANTs, writes out the
    transformed result to disk as a NIfTI file, and returns the nibabel image.
    """
    # Construct the output filename
    out_path = f"output/registered_output_sub-{subject_id}_ses-{session}.nii.gz"
    # If the file already exists, just load it and skip registration
    if os.path.exists(out_path):
        try:
            file = nib.load(out_path)
        except Exception as e:
            logging.error(f"Error registering images for subject {subject_id} session {session}: {e}")
            raise
        return file

    moving_image = ants.n4_bias_field_correction(moving_image)
    moving_image = ants.smooth_image(moving_image, 2)

    # Otherwise performs the registration
    outputImage = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=transform_type
    )['warpedmovout']
    
    # mask = ants.get_mask(outputImage)
    # seg_results = ants.atropos(
    #     a=outputImage,
    #     x=mask,               # Optional mask of the brain if you have it
    #     i='KMeans[3]',       # 3 classes: GM, WM, CSF
    #     m='[0.1, 1x1x1]',      # smoothing + mask dilation settings
    #     c='[5,0]'
    # )

    # # 2. Create a GM mask: True where segmentation == 1
    # outputImage = ants.mask_image(
    #     image=registered,          # your T1 ANTsImage
    #     mask=seg_results['segmentation'],
    #     level=1                    # the label for GM in your segmentation
    # )

    # Write out to disk
    out_path = f"output/registered_output_sub-{subject_id}_ses-{session}.nii.gz"
    ants.image_write(outputImage, out_path)
    try:
        file = nib.load(out_path)
    except Exception as e:
        logging.error(f"Error registering images for subject {subject_id} session {session}: {e}")
        raise
    return file


def loadImage(imgPath, subject_id, session):
    """
    Reads an image from a given path, then registers it against the
    Harvard-Oxford atlas, returning a NIfTI image resampled to the atlas space.
    """
    atlasPath = '/Users/mayerunterberg/nilearn_data/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'
    atlas_image = ants.image_read(atlasPath)
    ants_img = ants.image_read(imgPath)
    
    # Register and convert to NIfTI
    nifti = register_and_convert_to_nifti(
            fixed_image=atlas_image,
            moving_image=ants_img,
            subject_id=subject_id,
            session=session
        )
        
    img_resampled = resample_to_img(
        source_img=nifti,
        target_img=ATLAS_IMAGE,
        force_resample=True,
        copy_header=True,
        interpolation='nearest'
    )
    return img_resampled

def findDifferingAreasAndVolume(index, df, img1, img2, threshold=None):
    """
    Given two images, computes the difference map, calculates mean difference
    and volume changes per region, and records those metrics in the DataFrame.
    """
    # 1) Create the difference map
    diff = math_img("img1 - img2", img1=img1, img2=img2)
    diff_data = get_data(diff)
    
    # 2) For volume calculations, get data for each image
    img1_data = get_data(img1)
    img2_data = get_data(img2)
    
    # 3) We also need voxel volume, typically from the ATLAS_IMAGE’s affine
    # or if each registered image uses the same affine, we can use img1.affine
    voxel_sizes = np.abs(np.diag(ATLAS_IMAGE.affine)[:3])  # shape (3,)
    voxel_volume = np.prod(voxel_sizes)  # in mm^3
    # Loop through each region to compute statistics
    for label in REGION_LABELS:
        # regionMask indicates voxels belonging to this label
        # Determine region name
        if 0 <= label < len(LABEL_NAMES):
            region_name = LABEL_NAMES[label]
        else:
            region_name = "Unknown Region"
            continue
        if region_name in ("Background"):
            continue

        regionMask = (ATLAS_DATA == label)
        
        # A) Mean difference
        regionChanges = diff_data[regionMask]
        region_mean_diff = round(float(np.mean(regionChanges)), 2)
        
        # B) Volume difference
        # Count nonzero (or > 0) voxels in each image within the region
        # In structural MRI, you might need a threshold or binarization approach.
        # For a simple approach, let's count all *non-zero* voxels in that region:
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
        df.loc[index, f"{region_name} Change"] = region_mean_diff
        df.loc[index, f"{region_name} Volume Avg"] = avgVolume
        df.loc[index, f"{region_name} Volume Change"] = region_volume_diff


def main():
    participants_df = pd.read_csv('Data/participants.tsv', sep="\t")
    # Add columns for file paths
    # Prepare all the new column names
    new_cols = {}
    filePaths = ['Baseline File Path', 'Followup File Path']
    for label in filePaths:
        new_cols[f"{label}"] = [None]*len(participants_df)
    for label in LABEL_NAMES:
        if label in ("Background", "Unknown Region"):
            continue
        new_cols[f"{label} Volume Avg"] = [None]*len(participants_df)
    for label in LABEL_NAMES:
        if label in ("Background", "Unknown Region"):
            continue
        new_cols[f"{label} Volume Change"] = [None]*len(participants_df)
    for label in LABEL_NAMES:
        if label in ("Background", "Unknown Region"):
            continue
        new_cols[f"{label} Change"] = [None]*len(participants_df)

    # Create an empty DataFrame with those columns
    columns_df = pd.DataFrame(new_cols, index=participants_df.index)

    # Concatenate horizontally (axis=1)
    participants_df = pd.concat([participants_df, columns_df], axis=1)
    get_subject_file_pairs('Data', participants_df)


    for idx, row in participants_df.iterrows():
        baseLinePath = row['Baseline File Path']
        followUpPath = row['Followup File Path']
        if pd.isna(baseLinePath) or pd.isna(followUpPath):
            continue  # skip if missing file
        baseLine = loadImage(baseLinePath, row['participant_id'], 'BL')
        followUp = loadImage(followUpPath, row['participant_id'], 'FU')
        findDifferingAreasAndVolume(idx, participants_df, baseLine, followUp)


    import subprocess
    output_excel_path = 'participants_with_changes.xlsx'

    # Use the XlsxWriter engine so we can format cells
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        participants_df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        cell_format = workbook.add_format({
            'font_size': 14,
            'align': 'center',
            'valign': 'vcenter'
        })
        for col_num, col_name in enumerate(participants_df.columns):
            column_data = participants_df[col_name].astype(str)
            max_len = max(column_data.map(len).max(), len(col_name))
            worksheet.set_column(col_num, col_num, max_len + 2, cell_format)
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


    # Load the data
    file_path = 'participants_with_changes.xlsx'  # Ensure the file is in the working directory
    df = pd.read_excel(file_path)
    df['avg cudit'] = (df['cudit total baseline'] + df['cudit total follow-up']) / 2
    df = df.drop(columns=['Baseline File Path', 'Followup File Path'])

    # Data Preprocessing
    # Identify and encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = df.copy()

    # Encode all categorical columns except the target
    for col in categorical_cols:
        if col != 'avg cudit':  # Don't encode the target
            df_encoded[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Remove other CUDIT columns from the features
    # Exclude columns related to CUDIT scores from features
    columns_to_exclude = [
        'gender', 'avg cudit', 'cudit total baseline', 'cudit total follow-up',
        'audit total baseline', 'audit total follow-up', 'participant_id', 
        'group', 'age at onset first CB use', 'age at onset frequent CB use',
        'age at baseline'
    ]
    X = df_encoded.drop(columns=columns_to_exclude, errors='ignore')  # Exclude irrelevant columns
    y = df_encoded['avg cudit']

    # Handle missing values by imputing with mean
    X.fillna(X.mean(), inplace=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Feature Importance
    feature_importances = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Display the top 5 most important features
    top_5_features = feature_importances.head(5)
    print("Top 5 Features:")
    print(top_5_features)

    # Visualize Top 5 Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_5_features, y=top_5_features.index, palette='viridis', hue=None)
    plt.title('Top 5 Feature Importance for Predicting avg cudit Scores')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Load the data
    file_path = 'participants_with_changes.xlsx'  # Ensure this file is in the working directory
    df = pd.read_excel(file_path)
    df['avg cudit'] = (df['cudit total baseline'] + df['cudit total follow-up']) / 2

    # Drop irrelevant columns
    df = df.drop(columns=['Baseline File Path', 'Followup File Path'])

    # Encode categorical columns and drop unnecessary columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    X = df_encoded.drop(columns=columns_to_exclude, errors='ignore')
    y = df_encoded['avg cudit']

    # Impute missing values with the mean
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Prepare a DataFrame to store results
    results = pd.DataFrame(columns=['Feature', 'MSE', 'R²', 'P-value'])

    # Perform linear regression for each feature with significance testing
    for feature in X_imputed.columns:
        # Prepare single feature for regression
        X_feature = X_imputed[[feature]]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Use statsmodels for significance testing
        X_feature_with_const = sm.add_constant(X_feature)  # Add intercept
        ols_model = sm.OLS(y, X_feature_with_const).fit()
        if len(ols_model.pvalues) > 1:  # Ensure the p-value for the feature exists
            p_value = ols_model.pvalues.iloc[1]
        else:
            p_value = np.nan
        
        new_row = pd.DataFrame({'Feature': [feature], 'MSE': [mse], 'R²': [r2], 'P-value': [p_value]})
        # Check if the new_row is valid before concatenating
        if not new_row.isna().all(axis=None) and not new_row.empty:
            results = pd.concat([results, new_row], ignore_index=True)

    # Sort the results by R² in descending order
    results = results.sort_values(by='R²', ascending=False)

    # Display the results
    print("Linear Regression Results with Significance Testing:")
    print(results.head(10))

    # Save results to Excel if needed
    results.to_excel('linear_regression_with_significance.xlsx', index=False)


if __name__ == "__main__":
    main()