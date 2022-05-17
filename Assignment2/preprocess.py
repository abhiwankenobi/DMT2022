import pdb
from turtle import pd
import pandas as pd
import numpy as np
import copy
import datetime
import pdb



def add_click_and_bool_rates():
    pass

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

def impute_data(data, split='train'):
    data_clone = copy.deepcopy(data)
    
    if split == 'train':
        comp_features = {i: 0 for i in data_clone.columns[27:51]}
        data_clone.loc[data_clone['gross_bookings_usd'].isnull(), 'gross_bookings_usd'] = 0
    elif split == 'test':
        comp_features = {i: 0 for i in data_clone.columns[26:50]}
    
    data_clone.fillna(comp_features, inplace=True)

    data_clone['visitor_hist_starrating'].fillna(0, inplace=True)
    data_clone['visitor_hist_adr_usd'].fillna(0, inplace=True)

    data_clone['prop_review_score'].fillna(0, inplace=True)
    data_clone['prop_location_score2'].fillna(0, inplace=True)

    data_clone['orig_destination_distance'].fillna(data_clone['orig_destination_distance'].mean(), inplace=True)
    data_clone['srch_query_affinity_score'].fillna(-330, inplace=True)
    return data_clone

def remove_competitor_data(data, split='train'):
    data_clone = copy.deepcopy(data)
    if split == 'train':
        data_clone.drop(columns=data_clone.columns[27:51], inplace=True)
    elif split == 'test':
        data_clone.drop(columns=data_clone.columns[26:50], inplace=True)
    
    data_clone.drop(columns= ['visitor_hist_starrating', 'visitor_hist_adr_usd'], inplace=True) 
    return data_clone

def handle_outliers(data):
    data_clone = copy.deepcopy(data)
    data_clone = data_clone[(data_clone['price_usd'] != 0) & (data_clone['price_usd'] < 10000) ]
    return data_clone

def preprocess_data(data, split='train'):
    preprocessed_data = impute_data(data, split)
    preprocessed_data = remove_competitor_data(preprocessed_data, split)
    preprocessed_data = add_datetime_features(preprocessed_data)
    preprocessed_data = add_mean_prop_features(preprocessed_data)
    
    if split == 'train':
        preprocessed_data = handle_outliers(preprocessed_data)
    return preprocessed_data



    