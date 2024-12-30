import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# begin problem file
import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Classification of variable stars from light curves'
_target_column_name = 'type'
_ignore_column_names = []
_prediction_label_names = [1.0, 2.0, 3.0, 4.0]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='acc', precision=4),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


# READ DATA
def csv_array_to_float(csv_array_string):
    return list(map(float, csv_array_string[1:-1].split(',')))


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


def read_data(path, df_filename, vf_filename):
    df = pd.read_csv(os.path.join(path, 'data', df_filename), index_col=0)
    y_array = df[_target_column_name].values.astype(int)
    X_dict = df.drop(_target_column_name, axis=1).to_dict(orient='records')
    vf_raw = pd.read_csv(os.path.join(path, 'data', vf_filename),
                         index_col=0, compression='gzip')
    vf_dict = vf_raw.map(csv_array_to_float).to_dict(orient='records')
    X_dict = [merge_two_dicts(d_inst, v_inst) for d_inst, v_inst
              in zip(X_dict, vf_dict)]
    return pd.DataFrame(X_dict), y_array


def get_train_data(path='.'):
    df_filename = 'train.csv'
    vf_filename = 'train_varlength_features.csv.gz'
    return read_data(path, df_filename, vf_filename)


def get_test_data(path='.'):
    df_filename = 'test.csv'
    vf_filename = 'test_varlength_features.csv.gz'
    return read_data(path, df_filename, vf_filename)

# end problem file

def fold_time_series(time_point, period, div_period):
    return (time_point -
            (time_point // (period / div_period)) * period / div_period)


def get_bin_means(X_df, num_bins, band):
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


transformer_r = FunctionTransformer(
    lambda X_df: get_bin_means(X_df, 5, 'r')
)

transformer_b = FunctionTransformer(
    lambda X_df: get_bin_means(X_df, 5, 'b')
)

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
]

common = ['period', 'div_period']
transformer = make_column_transformer(
    (transformer_r, common + ['time_points_r', 'light_points_r']),
    (transformer_b, common + ['time_points_b', 'light_points_b']),
    ('passthrough', cols)
)

pipe = make_pipeline(
    transformer,
    SimpleImputer(strategy='most_frequent'),
    RandomForestClassifier(max_depth=5, n_estimators=10)
)


def get_estimator():
    return pipe

# get the training data
X_df, y = get_train_data()

print(X_df.head())
print(X_df.columns)
print(y[:5])

# labels
label_names = {1: 'binary', 2: 'cepheid', 3: 'rr_lyrae', 4: 'mira'}
labels = list(label_names.keys())
y_series = pd.Series(y).replace(label_names)
print(y_series.head())

_ = y_series.value_counts().plot(kind="bar")