__author__      = "Dima S."

import os
from string import punctuation
import numpy as np
import pandas as pd
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
import datetime

########################################################################################################
########################################################################################################

TRAIN_DATA = os.path.join(os.getcwd(), 'data', 'train.json')
TEST_DATA = os.path.join(os.getcwd(), 'data', 'test.json')
LEAKAGE_FEATURE = os.path.join(os.getcwd(), 'data', 'listing_image_time.csv')
BUILD_MANAGER_FEATURE = os.path.join(os.getcwd(), 'data', 'Builiding_manager_features.csv')

train_df = pd.read_json(TRAIN_DATA)
test_df = pd.read_json(TEST_DATA)

leakage_df = pd.read_csv(LEAKAGE_FEATURE)
leakage_df.columns = ["listing_id", "time_stamp"]
# reassign the only one timestamp from April, all others from Oct/Nov
leakage_df.loc[80240, "time_stamp"] = 1478129766

build_manager_df = pd.read_csv(BUILD_MANAGER_FEATURE)

train_ids = train_df[['listing_id', 'interest_level']]
test_ids = test_df['listing_id']

EncodeCategoricalFeatures = True
DATA_IS_READY = False
LEAVE_SELECTED_FEATURES = False
########################################################################################################
########################################################################################################


def runXGB(train_X, train_y, test_X, test_y=None, seed_val=221, num_rounds=10000):
    """ Creates and trains xgb classifier. Can be used for test and CV

    :param train_X: 
    :param train_y: 
    :param test_X: 
    :param test_y: 
    :param seed_val: 
    :param num_rounds: 
    :return: 
    """
    params = {'objective': 'multi:softprob',
              'eta': 0.02,
              'max_depth': 5,
              'silent': 1,
              'num_class': 3,
              'eval_metric': "mlogloss",
              'min_child_weight': 1,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'seed': seed_val
              }

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        evals = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(params=params, dtrain=xgtrain, num_boost_round=num_rounds,
                          evals=evals, early_stopping_rounds=50)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(params=params, dtrain=xgtrain, num_boost_round=num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


def manager_interest_level(train_df1, test_df1):
    """ Adding manager_interest features to dataframes
    
    :param train_df1: 
    :param test_df1: 
    :return: 
    """
    index=list(range(train_df1.shape[0]))
    random.shuffle(index)
    a=[np.nan]*len(train_df1)
    b=[np.nan]*len(train_df1)
    c=[np.nan]*len(train_df1)

    for i in range(5):
        building_level={}
        for j in train_df1['manager_id'].values:
            building_level[j]=[0,0,0]
        test_index=index[int((i*train_df1.shape[0])/5):int(((i+1)*train_df1.shape[0])/5)]
        train_index=list(set(index).difference(test_index))
        for j in train_index:
            temp=train_df1.iloc[j]
            if temp['interest_level']=='low':
                building_level[temp['manager_id']][0]+=1
            if temp['interest_level']=='medium':
                building_level[temp['manager_id']][1]+=1
            if temp['interest_level']=='high':
                building_level[temp['manager_id']][2]+=1
        for j in test_index:
            temp=train_df1.iloc[j]
            if sum(building_level[temp['manager_id']])!=0:
                a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
                b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
                c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
    train_df1['manager_level_low']=a
    train_df1['manager_level_medium']=b
    train_df1['manager_level_high']=c



    a=[]
    b=[]
    c=[]
    building_level={}
    for j in train_df1['manager_id'].values:
        building_level[j]=[0,0,0]
    for j in range(train_df1.shape[0]):
        temp=train_df1.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1

    for i in test_df1['manager_id'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0]*1.0/sum(building_level[i]))
            b.append(building_level[i][1]*1.0/sum(building_level[i]))
            c.append(building_level[i][2]*1.0/sum(building_level[i]))
    test_df1['manager_level_low']=a
    test_df1['manager_level_medium']=b
    test_df1['manager_level_high']=c
    return train_df1, test_df1


def building_interest_level(train_df1, test_df1):
    """ Adding building interest level to dataframes
    
    :param train_df1: 
    :param test_df1: 
    :return: 
    """
    index=list(range(train_df1.shape[0]))
    random.shuffle(index)
    a=[np.nan]*len(train_df1)
    b=[np.nan]*len(train_df1)
    c=[np.nan]*len(train_df1)

    for i in range(5):
        building_level={}
        for j in train_df1['building_id'].values:
            building_level[j]=[0,0,0]
        test_index=index[int((i*train_df1.shape[0])/5):int(((i+1)*train_df1.shape[0])/5)]
        train_index=list(set(index).difference(test_index))
        for j in train_index:
            temp=train_df1.iloc[j]
            if temp['interest_level']=='low':
                building_level[temp['building_id']][0]+=1
            if temp['interest_level']=='medium':
                building_level[temp['building_id']][1]+=1
            if temp['interest_level']=='high':
                building_level[temp['building_id']][2]+=1
        for j in test_index:
            temp=train_df1.iloc[j]
            if sum(building_level[temp['building_id']])!=0:
                a[j]=building_level[temp['building_id']][0]*1.0/sum(building_level[temp['building_id']])
                b[j]=building_level[temp['building_id']][1]*1.0/sum(building_level[temp['building_id']])
                c[j]=building_level[temp['building_id']][2]*1.0/sum(building_level[temp['building_id']])
    train_df1['building_level_low']=a
    train_df1['building_level_medium']=b
    train_df1['building_level_high']=c



    a=[]
    b=[]
    c=[]
    building_level={}
    for j in train_df1['building_id'].values:
        building_level[j]=[0,0,0]
    for j in range(train_df1.shape[0]):
        temp=train_df1.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['building_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['building_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['building_id']][2]+=1

    for i in test_df1['building_id'].values:
        if i not in building_level.keys():
            a.append(np.nan)
            b.append(np.nan)
            c.append(np.nan)
        else:
            a.append(building_level[i][0]*1.0/sum(building_level[i]))
            b.append(building_level[i][1]*1.0/sum(building_level[i]))
            c.append(building_level[i][2]*1.0/sum(building_level[i]))
    test_df1['building_level_low']=a
    test_df1['building_level_medium']=b
    test_df1['building_level_high']=c
    return train_df1, test_df1


def data_concat(df1, df2):
    """ Merging train and test datasets into one
    
    :param df1: 
    :param df2: 
    :return: 
    """
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop(labels=['interest_level'], axis=1)
    return df


def merge_with_img_df(df, image_date):
    """ Adding leakage feature, posted by Kazanova at the and of the competition, to the main dataframe
    
    :param df: 
    :param image_date: 
    :return: 
    """
    image_date["img_date"] = pd.to_datetime(image_date["time_stamp"], unit="s")
    image_date["img_days_passed"] = (image_date["img_date"].max() - image_date["img_date"]).astype(
        "timedelta64[D]").astype(int)
    image_date["img_date_month"] = image_date["img_date"].dt.month
    image_date["img_date_week"] = image_date["img_date"].dt.week
    image_date["img_date_day"] = image_date["img_date"].dt.day
    image_date["img_date_dayofweek"] = image_date["img_date"].dt.dayofweek
    image_date["img_date_dayofyear"] = image_date["img_date"].dt.dayofyear
    image_date["img_date_hour"] = image_date["img_date"].dt.hour
    image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(
        lambda x: 1 if x < 10 else 2 if x < 20 else 3)

    df = pd.merge(df, image_date, on="listing_id", how="left")
    return df


def merge_build_manag_df(df, build_manager_df):
    """ Adding leakage feature, posted by Kazanova at the and of the competition, to the main dataframe

    :param df: 
    :param image_date: 
    :return: 
    """

    df = pd.merge(df, build_manager_df, on="listing_id", how="left")
    return df


def data_split(df):
    """ Splits main dataframe back to train and test
    
    :param df: 
    :return: 
    """
    target_num_map = {'high': 0, 'medium': 1, 'low': 2}
    train_df1 = df[df['listing_id'].isin(train_ids['listing_id'].values)]
    train_df1['interest_level'] = train_df1['listing_id'].apply(
        lambda x: train_ids.loc[train_ids['listing_id'] == x, 'interest_level'].values[0])
    train_df1['interest_level'] = train_df1['interest_level'].apply(lambda x: target_num_map[x])

    test_df1 = df[df['listing_id'].isin(test_ids.values)]

    return train_df1, test_df1


def new_features_df(df):
    """ Feature engineering function
    
    :param df: 
    :return: 
    """
    remove_punct_map = dict.fromkeys(map(ord, punctuation))

    # price and rooms additional features
    df["logprice"] = np.log(df["price"])
    df["price_t"] = df["price"] / df["bedrooms"]
    df["room_sum"] = df["bedrooms"] + df["bathrooms"]
    df['price_per_room'] = df['price'] / df['room_sum']

    # location feature
    df["pos"] = df['longitude'].round(3).astype(str) + '_' + df['latitude'].round(3).astype(str)
    vals = df['pos'].value_counts()
    dvals = vals.to_dict()
    df["density"] = df['pos'].apply(lambda x: dvals.get(x, vals.min()))
    df = df.drop(labels=['pos'], axis=1)

    # deal with *created* feature
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour

    # deal with *photos* feature
    # nothing else we can do except count photos
    df["num_photos"] = df["photos"].apply(len)

    # deal with *features* feature
    # TODO: tokenize or similar
    df["num_features"] = df["features"].apply(len)

    # deal with *description* feature
    # TODO: try word2vec representation or similar
    df['desc'] = df['description']
    df['desc'] = df['desc'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))
    df['desc'] = df['desc'].apply(lambda x: x.replace('!<br /><br />', ''))
    df['desc'] = df['desc'].apply(lambda x: x.replace('!<br /><br />', ''))
    df['desc'] = df['desc'].apply(lambda x: x.translate(remove_punct_map))
    df["num_description_words"] = df["desc"].apply(lambda x: len(list(filter(None, x.split(" ")))))

    # deal with *description* feature
    # if present == 1, else == 0
    df['building_id_present'] = df['building_id'].apply(lambda x: 1 if x != '0' else 0)

    manager_count = df.groupby('manager_id').count().iloc[:, -1].to_dict()
    df['manager_count'] = df['manager_id'].apply(lambda x: manager_count.get(x, 0))

    building_count = df.groupby('building_id').count().iloc[:, -1].to_dict()
    building_count.pop('0')
    df['building_count'] = df['building_id'].apply(lambda x: building_count.get(x, 1))

    return df


def add_desc_features(df):
    """ Feature engineering with description
    
    :param df: 
    :return: 
    """
    general = ['elevator', 'hardwood', 'doorman', 'dishwasher', 'no fee', 'reduced fee' 'laundry', 'fitness', 'cat', 'dog',
               'roof', 'outdoor space', 'dining', 'internet', 'balcon', 'pool', 'new construction', 'terrace',
               'exclusive', 'loft', 'garden', 'wheelchair', 'fireplace', 'garage', 'furnished', 'multi-level']
    pre_war = ['prewar', 'pre-war', 'pre war']
    for item in general:
        df['feature'+'_'+item] = df['features'].apply(lambda x: 1 if item in " ".join(x).lower() else 0)
    df['feature_prewar'] = df['features'].apply(lambda x: 1 if any(item in " ".join(x).lower() for item in pre_war) else 0)
    return df


def remove_categorical_features(df):
    """ Remove categorical features from df. Last steps before training
    
    :param df: 
    :return: 
    """
    categogical = ['created', 'description', 'photos', 'street_address', 'features', 'desc', 'img_date']
    df = df.drop(labels=categogical, axis=1)
    return df


def encode_categorical_features(df):
    """ Label encoding rest of categorical features. One-hot encoding is not applicable, too mach manager_ids
    
    :param df: 
    :return: 
    """
    # categorical = ['building_id', 'display_address', 'manager_id']
    categorical = [c for c in df.columns if df[c].dtype.name == 'object']
    for f in categorical:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[f].values))
        df[f] = lbl.transform(list(df[f].values))
    return df


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_submission(predictions, test_id, log_loss):
    result1 = pd.DataFrame(predictions, columns=["high", "medium", "low"])
    result1['listing_id'] = test_id.values
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(log_loss) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


if __name__ == '__main__':
    if not DATA_IS_READY:
        print('PREPROCESSING DATA..............')
        print()
        train_df, test_df = manager_interest_level(train_df, test_df)
        train_df, test_df = building_interest_level(train_df, test_df)

        df = data_concat(train_df, test_df)
        df = merge_with_img_df(df, leakage_df)
        df = merge_build_manag_df(df, build_manager_df)
        df = new_features_df(df)
        df = add_desc_features(df)
        df = remove_categorical_features(df)

        categorical_columns = [c for c in df.columns if df[c].dtype.name == 'object']
        numerical_columns = [c for c in df.columns if df[c].dtype.name != 'object']

        print('DATA VARIABLES\n')
        print('=' * 30)
        print('categogical columns:')
        print()
        print('\n'.join(categorical_columns))
        print('=' * 30)
        print('numerical columns:')
        print()
        print('\n'.join(numerical_columns))
        print('=' * 30)
        print()

        if EncodeCategoricalFeatures:
            df = encode_categorical_features(df)
            print('=' * 30)
            print('Categorical features were encode with LabelEncoder()')
            print()
        else:
            df = df.drop(labels=categorical_columns, axis=1)
            print('=' * 30)
            print('Categorical features were dropped')
            print()

        with open('data/features.txt', 'w') as ftrs:
            for item in df.columns:
                ftrs.write(item + '\n')

        train_df_new, test_df_new = data_split(df)
        train_df_new.to_csv('data/train_df_new.csv', index=False)
        test_df_new.to_csv('data/test_df_new.csv', index=False)
    else:
        train_df_new = pd.read_csv('data/train_df_new.csv')
        test_df_new = pd.read_csv('data/test_df_new.csv')

    train_ids_new = train_df['listing_id']
    test_ids_new = test_df['listing_id']

    if LEAVE_SELECTED_FEATURES:
        with open('data/selected_features.txt') as file:
            features = file.read().splitlines()
            train_df_new = train_df_new[features + ['interest_level']]
            test_df_new = test_df_new[features]
    else:
        train_df_new = train_df_new.drop(labels=['time_stamp', 'latitude', 'longitude'], axis=1)
        test_df_new = test_df_new.drop(labels=['time_stamp', 'latitude', 'longitude'], axis=1)

    X = train_df_new.drop(labels=['listing_id', 'interest_level'], axis=1)
    y = train_df_new['interest_level']
    Xgtest = xgb.DMatrix(test_df_new.drop(labels=['listing_id'], axis=1))

    cv_scores = []
    models = []
    skf = StratifiedKFold(n_splits=4, shuffle=True)

    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        preds, model = runXGB(X_train, y_train, X_test, y_test)
        cv_scores.append(log_loss(y_test, preds))
        models.append(model)

    num_fold = 0
    yfull_test = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_prediction = model.predict(Xgtest)
        yfull_test.append(test_prediction)
    print('Log-los = ', np.mean(cv_scores))
    test_res = merge_several_folds_mean(yfull_test, nfolds)
    create_submission(test_res, test_ids_new, np.mean(cv_scores))
