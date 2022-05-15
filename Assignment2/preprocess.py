from turtle import pd
import pandas as pd
import numpy as np
import copy

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

def preprocess_data(data, split='train'):
    preprocessed_data = impute_data(data, split)
    preprocessed_data = preprocessed_data.drop(columns=['date_time'], inplace=False)
    return preprocessed_data



    