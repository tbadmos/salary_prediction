#helper is group of function created to make coding modular

import pandas as pd
import numpy as np
import joblib

import xgboost

#function creates list of categorical features in dataframe
def cat_feat_list(data, target):

    ''' return list of categorical features and also excludes the target in the dataframe

        Parameters:
        data: dataframe object
        target(string): column label is not listed in returned list'''
    
    c= []
    for feature in data.columns:
        if data[feature].dtype == object and feature != target :
            c.append(feature)
    return c





#ordinal encoder for test data
def test_ord_enc(data):
    '''Uses ordinal encoder objects from train data set to encode features on test data. Returns dataframe with new labels containing encoded feature values

        Parameters:
        data: test dataframe '''
    enc_dict = joblib.load('features_ordinal_econders')
    feature = cat_feat_list(data, None)
    for i in feature:
        data[i + '_cat'] = enc_dict[i].fit_transform(data[[i]]) + 1
    print('Data encoded')
    return data 


#function bins values of features into specified groups
def add_binned_feat(data, feature, num_bins, bin_labels, new_feat_name):
    '''categorizes feature with numeric values into groups

        Parameters:
        data: input pandas dataframe
        feature(string): column label in dataframe whose numeric values will be binned
        num_bins(integer): number of groups to create
        bin_labels(list of integers): number assigned to each group
        new_feat_name(string): name of new column to add to the input dataframe which will contain the corresponding group value of the feature value

        retuns dataframe with the new column'''
    
    data[new_feat_name] = pd.cut(data[feature], num_bins, labels = bin_labels ).astype('int64')
    print (feature, ' values binned')
    return data

#function creates list of numerical features in dataframe
def num_feat_list(data, target):
    ''' return list of features whose values are numerical and also excludes the target in the dataframe

        Parameters:
        data: dataframe object
        target(string): column label is not listed in returned list'''
    
    c= []
    for feature in data.columns:
        if data[feature].dtype != object and feature != target :
            c.append(feature)
    return c    

#function for prediction in web_app

def predict(to_predict_params):

    #convert to pandas dataframe
    input_table = pd.DataFrame(to_predict_params, index = [1]) 

    #cast to numeric
    input_table = input_table.astype({'yearsExperience' : 'int64'})
    input_table = input_table.astype({'milesFromMetropolis': 'int64'})
   

    #encode inputs
    input_table = test_ord_enc(input_table)

    #bin years of experience and distance from metro
    input_table = add_binned_feat(input_table, 'yearsExperience', 6, [1,2,3,4,5,6], 'yearsExp_cat') 
    #order of range reversed because average salary inversly propotional to milesFromMetropolis
    input_table = add_binned_feat(input_table, 'milesFromMetropolis', 10, [i for i in range(10,0,-1)], 'mfm_cat') 

    model = joblib.load('final_xgb_model_saved')

    num_features = num_feat_list(input_table, 'salary')

    x_pred = input_table[ num_features ]

    y_pred = model.predict(x_pred)

    return int(np.round(y_pred[0]*1000, -1))

        