import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from DataPipe import DataPipe

# Suppress warnings related to DataFrame concatenation
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries")

class CannabisAnalysis:
    """
    This class processes neuroimaging and behavioral data, 
    runs linear regression, computes p-values, and visualizes 
    top predictive features for CUDIT scores.
    """
    
    def __init__(self):
        """Initialize DataPipe object and load participants' data."""
        self.data_processor = DataPipe()
        self.load_data()

    def load_data(self):
        """Load the dataset, process subject pairs, and extract volumetric data."""
        print("Loading data and processing subjects...")
        self.data_processor.get_subject_file_pairs()
        self.data_processor.process_all_subjects()
        self.df = self.data_processor.participants_df
        self.df['avg_cudit'] = (self.df['cudit total baseline'] + self.df['cudit total follow-up']) / 2

    def preprocess_data(self):
        """
        Prepares the dataset:
        - Drops unnecessary columns
        - Encodes categorical variables
        - Imputes missing values
        """
        print("Preprocessing data...")

        # Drop irrelevant columns
        columns_to_exclude = [
            'gender', 'avg_cudit', 'cudit total baseline', 'cudit total follow-up',
            'audit total baseline', 'audit total follow-up', 'participant_id',
            'group', 'age at onset first CB use', 'age at onset frequent CB use', 'age at baseline',
            'Temporal Fusiform Cortex, posterior division Change', 'Inferior Temporal Gyrus, anterior division Volume Avg',
            'Baseline File Path', 'Followup File Path'
        ]

        # Encode categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        df_encoded = self.df.copy()
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

        # Select predictor variables (X) and target variable (y)
        self.X = df_encoded.drop(columns=columns_to_exclude, errors='ignore')
        self.y = df_encoded['avg_cudit']

        # Handle missing values using mean imputation
        imputer = SimpleImputer(strategy="mean")
        self.X_imputed = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)

    def run_linear_regression(self):
        """
        Performs linear regression for each feature with significance testing:
        - Computes R² scores
        - Calculates p-values for statistical significance
        """
        print("Running linear regression and computing significance values...")
        results = pd.DataFrame(columns=['Feature', 'R²', 'P-value'])

        for feature in self.X_imputed.columns:
            X_feature = self.X_imputed[[feature]]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_feature, self.y, test_size=0.4, random_state=20)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict and compute R² score
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            # Use statsmodels for significance testing
            X_feature_with_const = sm.add_constant(X_feature)
            ols_model = sm.OLS(self.y, X_feature_with_const).fit()
            p_value = ols_model.pvalues.iloc[1] if len(ols_model.pvalues) > 1 else np.nan

            # Append results
            results = pd.concat([results, pd.DataFrame({'Feature': [feature], 'R²': [r2], 'P-value': [p_value]})], ignore_index=True)

        # Sort by R² value
        self.results = results.sort_values(by='R²', ascending=False)
        print("Top 10 Features based on R² Score:")
        print(self.results.head(10))

        # Save results to an Excel file
        self.results.to_excel('linear_regression_with_significance.xlsx', index=False)
    
    def run_random_forest(self):
        """
        Trains a Random Forest model to predict avg CUDIT score.
        Displays the top 10 most important features.
        """
        print("Training Random Forest model...")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(self.X_imputed, self.y, test_size=0.3, random_state=42)

        # Train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Compute R² score
        y_pred = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Random Forest R² Score: {r2:.4f}")

        # Get feature importances
        feature_importances = pd.DataFrame({'Feature': self.X.columns, 'Importance': rf_model.feature_importances_})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

        # Select only the top 10 most important features
        top_10_features = feature_importances.head(10)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_features['Importance'], y=top_10_features['Feature'], palette='coolwarm')
        plt.title("Top 10 Feature Importance from Random Forest Model")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def plot_top_features(self, p_threshold=0.05, top_n=5):
        """
        Visualizes the top predictive features using a bar plot.
        Filters by p-value and selects the top N features.
        """
        print(f"Visualizing top {top_n} predictive features...")
        top_features = self.results[self.results['P-value'] < p_threshold].head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=top_features['R²'],
            y=top_features['Feature'],
            palette='viridis'
        )

        plt.title(f'Top {top_n} Features by R² Score for Predicting avg_cudit (p < {p_threshold})', fontsize=16)
        plt.xlabel('R² Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

    def visualize_brain_differences(self, subject_id):
        """
        Displays baseline vs. follow-up MRI difference map for a given subject.
        """
        print(f"Visualizing brain differences for subject {subject_id}...")
        base_path = f'output/registered_output_sub-{subject_id}_ses-BL.nii.gz'
        followup_path = f'output/registered_output_sub-{subject_id}_ses-FU.nii.gz'
        self.data_processor.display_brain_and_difference(base_path, followup_path)

    def run_analysis(self):
        """
        Main function to execute the full pipeline:
        - Preprocess data
        - Run regression
        - Run Random Forest model
        - Visualize correlation heatmap
        - Display brain differences for a sample subject
        """
        self.preprocess_data()
        self.run_linear_regression()
        self.run_random_forest()
        self.plot_top_features()
        self.visualize_brain_differences(subject_id=112)


def main():
    analysis = CannabisAnalysis()
    analysis.run_analysis()

# Run the full analysis
if __name__ == "__main__":
    main()