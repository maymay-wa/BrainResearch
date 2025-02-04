from Analysis import Analysis

def main():
    analysis = Analysis()
    analysis.preprocess_data()
    analysis.run_linear_regression()
    analysis.plot_top_features()
    analysis.run_random_forest()
    analysis.visualize_brain_differences(subject_id=112)
    analysis.visualize_brain_differences_registry(subject_id=103)

# Run the full analysis
if __name__ == "__main__":
    main()