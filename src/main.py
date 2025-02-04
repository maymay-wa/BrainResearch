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
from Analysis import Analysis

def main():
    analysis = Analysis()
    analysis.run_analysis()

# Run the full analysis
if __name__ == "__main__":
    main()