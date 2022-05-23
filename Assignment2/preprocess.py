import pdb
import pandas as pd
import numpy as np
import copy
import datetime



def impute_data(data, split='train'):
    data_clone = copy.deepcopy(data)
    
    if split == 'train':
        comp_features = {i: 0 for i in data_clone.columns[27:51]}
        data_clone.loc[data_clone['gross_bookings_usd'].isnull(), 'gross_bookings_usd'] = 0
    elif split == 'test':
        comp_features = {i: 0 for i in data_clone.columns[26:50]}
    
    data_clone.fillna(comp_features, inplace=True)

    data_clone['visitor_hist_starrating'].fillna(-1, inplace=True)
    data_clone['visitor_hist_adr_usd'].fillna(-1, inplace=True)

    data_clone['prop_review_score'].fillna(-1, inplace=True)
    data_clone['prop_location_score2'].fillna(0, inplace=True)

    data_clone['orig_destination_distance'].fillna(data_clone['orig_destination_distance'].mean(), inplace=True)
    data_clone['srch_query_affinity_score'].fillna(-330, inplace=True)
    print('Done imputing')
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
    
    if split == 'train':
        preprocessed_data = handle_outliers(preprocessed_data)
    return preprocessed_data



    