import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def fold_time_series(time_point, period, div_period):
    '''
    Folds a time point into the interval [0, period/div_period).
    This function "folds" the time points of a star's light curve based on its period and divided period (div_period).
    Folding means mapping the time points into a single cycle of the periodic signal.
    Useful for analyzing stars since their brightness changes periodically.
    '''
    return (time_point -
            (time_point // (period / div_period)) * period / div_period)


def get_bin_means(X_df, num_bins, band):
    '''
    This function creates binned features for the light curves:
        Input: A DataFrame X_df and a band (e.g., 'r' for red or 'b' for blue).
        Splits the folded time points into num_bins bins (subdivisions of the period).
        For each bin, computes the mean light intensity of the light points that fall into it.
        Handles missing values by assigning NaN to bins with no data.
    '''
    feature_array = np.empty((len(X_df), num_bins))

    for k, (_, x) in enumerate(X_df.iterrows()):
        period = x['period']
        div_period = x['div_period']
        real_period = period / div_period
        bins = [i * real_period / num_bins for i in range(num_bins + 1)]

        time_points = np.array(x['time_points_' + band])
        light_points = np.array(x['light_points_' + band])
        time_points_folded = \
            np.array([fold_time_series(time_point, period, div_period)
                      for time_point in time_points])
        time_points_folded_digitized = \
            np.digitize(time_points_folded, bins) - 1

        for i in range(num_bins):
            this_light_points = light_points[time_points_folded_digitized == i]
            if len(this_light_points) > 0:
                feature_array[k, i] = np.mean(this_light_points)
            else:
                feature_array[k, i] = np.nan  # missing

    return feature_array


# Custom transformer to add the 'real_period' column
def add_real_period(X_df):
    X_df = X_df.copy()  # Ensure the original data is not modified
    X_df['real_period'] = X_df['period'] / X_df['div_period']
    return X_df


transformer_r = FunctionTransformer(
    lambda X_df: get_bin_means(X_df, 5, 'r')
)

transformer_b = FunctionTransformer(
    lambda X_df: get_bin_means(X_df, 5, 'b')
)

# Create the FunctionTransformer
real_period_transformer = FunctionTransformer(add_real_period)

# Column Selection and Feature Engineering:
cols = [
    'magnitude_b',
    'magnitude_r',
    'period',
    'asym_b',
    'asym_r',
    'log_p_not_variable',
    'sigma_flux_b',
    'sigma_flux_r',
    'quality',
    'div_period',
    'real_period'
]

common = ['period', 'div_period']
transformer = make_column_transformer(
    (transformer_r, common + ['time_points_r', 'light_points_r']),
    (transformer_b, common + ['time_points_b', 'light_points_b']),
    (StandardScaler(), cols),
    (PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), ['magnitude_b', 'period', 'sigma_flux_b']),
    ('passthrough', cols)
)

# Pipeline Definition:
pipe = make_pipeline(
    real_period_transformer,
    transformer,
    SimpleImputer(strategy='mean'),
    RandomForestClassifier(max_depth=5, n_estimators=10)
)


# Hyperparameter tuning
param_grid = {
    'randomforestclassifier__max_depth': [5, 10, 15],
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(pipe, param_grid, cv=3)

# Final pipeline
def get_estimator():
    return grid_search


