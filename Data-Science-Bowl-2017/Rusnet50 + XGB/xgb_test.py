import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import gmean


def get_data():
    with open('train_features.csv', 'r') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        train = [list(map(lambda x: int(x), row)) for row in data]
    with open('test_features.csv', 'r') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        test = [list(map(lambda x: int(x), row)) for row in data]
    return np.array(train), np.array(test)


def train_xgboost():
    df = pd.read_csv('stage1_labels.csv')

    x = get_data()[0]
    y = df['cancer'].as_matrix()

    skf = StratifiedKFold(n_splits=10, random_state=48, shuffle=True)
    result = []
    clfs = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index, :], x[test_index, :]
        trn_y, val_y = y[train_index], y[test_index]

        # clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=96, learning_rate=0.03757, nthread=8,
        #                       subsample=0.85, colsample_bytree=0.9, seed=96)

        clf = xgb.XGBRegressor(max_depth=5, n_estimators=2500, min_child_weight=116, learning_rate=0.03757, nthread=8,
                               subsample=0.85, colsample_bytree=0.9, seed=96)

        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=100)
        result.append(clf.best_score)
        clfs.append(clf)

    return clfs, result


def make_submite():
    clfs, result = train_xgboost()
    print(np.average(result))
    df = pd.read_csv('stage1_sample_submission.csv')
    x = get_data()[1]

    preds = [np.clip(clf.predict(x), 0.0001, 1) for clf in clfs]

    pred = gmean(np.array(preds), axis=0)

    df['cancer'] = pred
    df.to_csv('submission1.csv', index=False)


if __name__ == '__main__':
    make_submite()
