import pdb
import pandas as pd
import numpy as np
import copy
import datetime

def add_combined_mean_prop_features(train_data, test_data):
    train_data_clone = copy.deepcopy(train_data)
    test_data_clone = copy.deepcopy(test_data)
    test_data_clone['srch_id'] = test_data_clone['srch_id'].astype(int)
    test_data_clone['srch_id'] = test_data_clone['srch_id'].apply((lambda x: x + 4958347))
    data_clone = train_data_clone.append(test_data_clone)

    cols = ['prop_starrating', 'prop_review_score',
            'prop_log_historical_price', 'prop_location_score1', 'prop_location_score2']
    
    for var in cols:
        print('Adding mean, median and std of: ',var)  # For progression purposes
        selector = pd.DataFrame(data_clone.groupby('srch_id')[str(var)].mean())
        selector.columns = ['mean' + str(var)]
        #data_clone = pd.merge(data_clone, selector, how='left', on='srch_id')
        train_data_clone = pd.merge(train_data_clone, selector, how='left', on='srch_id')
        test_data_clone = pd.merge(test_data_clone, selector, how='left', on='srch_id')

        selector = pd.DataFrame(data_clone.groupby('srch_id')[str(var)].median())
        selector.columns = ['median' + str(var)]
        train_data_clone = pd.merge(train_data_clone, selector, how='left', on='srch_id')
        test_data_clone = pd.merge(test_data_clone, selector, how='left', on='srch_id')

        selector = pd.DataFrame(data_clone.groupby('srch_id')[str(var)].std())
        selector.columns = ['std' + str(var)]
        train_data_clone = pd.merge(train_data_clone, selector, how='left', on='srch_id')
        test_data_clone = pd.merge(test_data_clone, selector, how='left', on='srch_id')
    
    
    test_data_clone['srch_id'] = test_data_clone['srch_id'].apply((lambda x: x - 4958347))

    return train_data_clone, test_data_clone

def add_range_feature(data):
    data_clone = copy.deepcopy(data)
    selector = data_clone.groupby('srch_id')['price_usd'].max() - data.groupby('srch_id')['price_usd'].min()
    selector.columns = ['srch_id_max_price_diff']
    data_clone = pd.merge(data_clone, selector, how='left', on='srch_id')
    return data_clone

def add_mean_prop_features(data):
    data_clone = copy.deepcopy(data)
    cols = ['prop_starrating', 'prop_review_score',
            'prop_log_historical_price', 'prop_location_score1', 'prop_location_score2']
    
    for var in cols:
        print('Adding mean, median and std of: ',var)  # For progression purposes
        selector = pd.DataFrame(data_clone.groupby('srch_id')[str(var)].mean())
        selector.columns = ['mean' + str(var)]
        data_clone = pd.merge(data_clone, selector, how='left', on='srch_id')

        selector = pd.DataFrame(data_clone.groupby('srch_id')[str(var)].median())
        selector.columns = ['median' + str(var)]
        data_clone = pd.merge(data_clone, selector, how='left', on='srch_id')

        selector = pd.DataFrame(data_clone.groupby('srch_id')[str(var)].std())
        selector.columns = ['std' + str(var)]
        data_clone = pd.merge(data_clone, selector, how='left', on='srch_id')

    return data_clone

def add_datetime_features(data):
    data_clone = copy.deepcopy(data)
    data_clone['day_of_the_week'] = data_clone['date_time'].apply((lambda x: datetime.datetime.date(x).weekday()))
    data_clone['day'] = data_clone['date_time'].apply((lambda x: datetime.datetime.date(x).day))
    data_clone['month'] = data_clone['date_time'].apply((lambda x: datetime.datetime.date(x).month))
    data_clone['year'] = data_clone['date_time'].apply((lambda x: datetime.datetime.date(x).year))
    data_clone['week'] = data_clone['date_time'].apply((lambda x: datetime.datetime.date(x).isocalendar()[1]))
    data_clone.drop(columns=['date_time'], inplace=True)
    return data_clone

def add_normalised_prop_features(df):
    df_clone = copy.deepcopy(df)
    filled_prop_df = pd.read_csv('preprocessed_data/filled_prop_df.csv')
    selector = pd.DataFrame(filled_prop_df.groupby('prop_id')['price_usd'].mean())
    selector.columns =  ['price_usd_mean']
    df_clone = pd.merge(df_clone, selector, how='left', on='prop_id')
    
    #selector = pd.DataFrame(filled_prop_df.groupby('prop_id')['prop_location_score2'].mean())
    #selector.columns =  ['prop_location_score2_new']
    #df_clone = pd.merge(df_clone, selector, how='left', on='prop_id')
    
    #df_clone.drop(columns=['prop_location_score2'], inplace=True)
    
    return df_clone

def add_features(data):
    augmented_data = add_datetime_features(data)
    return augmented_data





