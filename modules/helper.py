#helper is group of function created to make coding modular

import sys
import pandas as pd
#import numpy as np
import joblib

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import GradientBoostingRegressor
import xgboost




def upload_file_csv(file_link):
    ''' Returns dataframe of a csv file with the input as the file's path'''
    #print(" Displaying nlot Snipet of Data")
    return  pd.read_csv(file_link)

def summary_stat(data):
    ''' returns basics stats of each column such as counts, mean, median, percentile in a table
        Also returns the data type of each column and the shape of the data
        Parameter:
        data: Pandas dataframe'''
    #print ('Shape of data is {} ' .format(data.shape), end ='\n\n')
    print('Data has %d rows by %d columns' %(data.shape[0], data.shape[1]), end ='\n\n')
    print ('Data type', data.dtypes, sep = '\n')
    return data.describe(include = 'all')



def uniq_values_in_feature(data, feature):
    '''Checks each column in dataframe and drops the columns if all its values are unique. Then prints out
        number of unique values in each column. 

       Parameters:
       data: Pandas dataframe object
       feature: list of column labels or a string

       Returns data without dropped columns'''

    print("")
    x =[]
    if type(feature) == list:
        for i in feature:
            uniq_vals = data[i].unique()
            l = len (uniq_vals)
            print('There are %s  unique values of %s' %(l, i) )
            #[print(val, end = ' ') for val in uniq_vals]
            if data.shape[0] == l:
               x.append(i)
            else: 
                continue
        return drop_feature(data, x)
    else:
        uniq_vals = data[feature].unique()
        l = len (uniq_vals)
        print('There are %s  unique values of %s' %(l, feature) )
        #[print(val, end = ',') for val in uniq_vals]


def missing_vals(data):
    ''' Prints out missing value in each column as a percents of the number of rows in input dataframe

         parameters:
         data: pandas dataframe object'''
    
    print('Percent missing of total')
    print(data.isnull().sum()/len(data))


def case_check(data):
    '''Checks if the values in each column (object data type) have uniform case(upper or lower) and prints out the result
       If values do not have uniform cases, all values are made upper cases  and the transformed data is returned

       Parameters:
       data: pandas dataframe object'''

    print('Checking if case in feature values are uniform')
    counter = data.shape[0]
    for i in data.columns:
        if data[i].dtype == object:
            total_uppercase = sum(data[i].apply(lambda x: x.isupper()))
            if counter == total_uppercase:
                print (i, '--->','All upppercase')
            elif total_uppercase == 0:
                print (i, '--->', 'All lowercase')
            else: 
                print (i, '--->', 'Mixed case')
                data[i] = data[i].str.upper()
                print(i, ' converted to uppercase')
    print('')             
    return data
             


def drop_missing (data, target):
    ''' drop missing value, empty strings and zero values in the target column of the dataframe input and returns the result

        parameters:
        data: pandas dataframe
        target(string): column label of dataframe to check for zero values '''
    
    int_rows = data.shape[0]
    missing_vals(data)
    #print('Output After dropping missing values')
    pd.options.mode.use_inf_as_na = True #sets empty strings as nan
    data.dropna(inplace = True)
    if target in data.columns:
        print ('\n', 'Dropped %d rows with zero %s values' %(int_rows - data[data[target] > 0].shape[0], target))
        return data[data[target] > 0]
    else:
        return data


def check_dup(data):
    ''' print number of duplicate rows in the input data

        parameters:
        data: pandas dataframe object'''
    
    duplicates = sum(data.duplicated(keep = 'first') == True)
    print('%d duplicates found' %duplicates)


def drop_dup(data):
    ''' drops duplicate rows and returns resulting dataframe

        parameters:
        data: pandas dataframe object'''
    
    duplicates = sum(data.duplicated() == True)
    print('\n', '%d duplicates found and removed' %duplicates)
    if duplicates > 0:
        #print('Output after dropping duplicates')
        return data.drop_duplicates()
    else:
        return data



def drop_feature(data, feature):
    ''' drops a column in dataframe and returns resulting dataframe

        parameters:
        data: pandas dataframe object
        feature(list of strings or string): column label(s) to drop'''
    
    print(feature, ' dropped')
    return data.drop( axis=1, columns = feature)



def group(data, features, fillter = 0):
    '''returns dataframe with values belonging to the same group aggregated in a list

        Parameters:
        data: pandas dataframe obnject
        features(list of strings or a string): column label(s) of the input dataframe to group by
        fillter(int): filter out groups with number of individuals greater than the fillter value '''
    
    grouped_data = data.groupby(features).agg(list)
    if fillter == 0:   
        return grouped_data
    else:   
        return grouped_data[grouped_data['salary'].apply(lambda x: len(x)> fillter)]


def group_mean(data, features, target):
    grouped_data = data.groupby(features).mean().sort_values(target)
    if fillter == 0:   
        return grouped_data



#show duplicated data on features set with thier corresponding target
def dup_feat(data, target_col):
    ''' drops the target column and checks for duplicates in remaining data set
        and returns the duplicates

        parameters:
        data: pandas dataframe object
        target_col('string'): column label to drop'''
    
    print ('Number of duplicates found ---> ', sum(data.drop(columns = target_col).duplicated()== True))
    return data[data.drop(columns = target_col).duplicated()]


def drop_dup1(data, target_col = None ):
    ''' Parameters:
        data: pandas dataframe object
        target_col(string): target label in dataframe

        if target_col = None, the function returns dataframe after dropping duplicate rows
        
        if target col is provided, the  function stores non-duplicate row in the dataframe that results after droping the target_col in the original dataframe (call the result x)
        It then uses the index of x to filter out the corresponding rows in original input dataframe and returns the result'''
    
    if target_col == None:
        print ('Number of duplicates found ---> ', sum(data.duplicated() == True))
        return data.drop_duplicates()
    else:
        print ('Number of duplicates found ---> ', sum(data.drop(columns = target_col).duplicated()== True)) 
        return data[data.drop(columns = target_col).duplicated()== False] 
    


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



def drop_feats(data, feature_list):

    '''drops feature(s) from a data frame and returns result

        Parameters:
        data: pandas dataframe object
        feature_list(string or list of strings): column(s) to drop from dataframe'''
    
    data.drop(columns = feature_list, inplace = True)
    return data


#convert features from one type to another
def conv_feat_type(data, feature, to_type ):
    
    '''Converts feature(s) values from one type to another(int, obj, etc)

        Parameters:
        data: Pandas dataframe object
        feature(string or list of strings): Feature(s) in dataframe'''
        
    for i in feature:
        if i.dtype != to_type:
            data[i] = data[i].astype(to_type)
            print(i, 'data type --->',  data[i].dtype)
        else:
            print(i, 'data type --->',  data[i].dtype)
    return data


#merger function
def merger(base_data, merger_data, merg_on_feat, how= None):

    '''Joins two dataframe together and returns result. See 'merge' method documentation on pandas dataframe objecy

        Parameters:
        base_data: Dataframe object onto which second dataframe is joined to
        merger_data: second dataframe which joins to first data_frame
        on(label or list): column or index level names to join on. These must be found in both DataFrames.
        how : {'left', 'right', 'outer', 'inner'}'''
    
    return base_data.merge(merger_data, on = merg_on_feat, how = how )


def color_map(data, pivot_row, pivot_column, target = 'salary'):

    '''' returns a color map between two features for a target lable'

        Parameters
        data: Dataframe object
        pivot_row(string): Feature in the y axis
        pivot_column(string): Feature in the x axis
        target(string): label in dataframe whose numerical values are colour coded in the color map'''

    
    group = data[[pivot_row, pivot_column, target]]
    group = group.groupby([pivot_row, pivot_column], as_index = False).mean()

    pivot_table = group.pivot(index = pivot_row, columns = pivot_column)

    fig1, ax = plt.subplots( figsize= (10,10))

    clr_map = ax.pcolormesh(pivot_table, cmap='RdBu')

    ax.set_xticks(np.arange(pivot_table.shape[1])+ 0.5)
    ax.set_yticks(np.arange(pivot_table.shape[0])+ 0.5)

    ax.set_xticklabels(pivot_table.columns.levels[1])
    ax.set_yticklabels(pivot_table.index)

    plt.xticks(rotation=90)


    #fig1.colorbar(clr_map)




 
def plot_feature_corr(data, feature1, feature2, target, order = 'y'):
    
    ''' Plots graphical representation between two categorical features and a numerical target

        Parameters:
        data: Dataframe objecy
        feature1(string): First categorical feature
        feature2(string): Second categorical feature
        target(string): Numberical target
        order('y' or 'n'): This orders the numerical target values('y) or not ('n') before ploting graph'''
    
    grouped_data = data[[feature1, feature2, target]].groupby([feature1, feature2], as_index = True).mean()
    grouped_data.reset_index(level = feature2, inplace = True)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_axes([0,0,1,1])
    
    if order == 'y':
        counter = 0
        for i in data[feature1].unique():
            try:
                ax.plot(grouped_data.loc[i].sort_values(target)[feature2], grouped_data.loc[i].sort_values(target)[target], 
                    color = 'C'+ str(counter), marker = 's', label=i )
            except:
                ax.plot(grouped_data.loc[i][feature2], grouped_data.loc[i][target], 
                    color = 'C'+ str(counter), marker = 's', label=i )    
            counter += 1
            
    elif order == 'n':
        counter = 0
        for i in data[feature1].unique():
            try:
                ax.plot(grouped_data.loc[i][feature2], grouped_data.loc[i][target], 
                    color = 'C'+ str(counter), marker = 's', label=i )
            except:
                ax.plot(grouped_data.loc[i][feature2], grouped_data.loc[i][target], 
                    color = 'C'+ str(counter), marker = 's', label=i )    
            counter += 1
            
    ax.set_ylabel('Average Salary ')
    ax.set_xlabel(feature2)
    ax.legend(  loc = 'lower right',  )



def feat_dist(data, feat_list):

    ''' Plots bar graphs of each feature. Show the distribution of values of features

        Parameters:
        data: Dataframe object
        feat_list(string or list of strings): feature to plot'''
    
    fig, ax = plt.subplots(len(feat_list),1, figsize = (20,20))
    cnt = 0
    for i in feat_list:
        ax[cnt].bar(data[i].sort_values().unique(), data[i].value_counts().sort_index())
        cnt += 1


#Ordinal encoder function #2 for train data
def cat_ord_enc(data, target, feature= None):

    '''Ordinal Encoder: Encodes features values based on average corresponding target value and returns dataframe with additional columns containing the codes
        saves ordinal_encoder object for each feature in a dictionary which can then be used to transform a test data set
        
        Parameters
        data: Dataframe object
        target(string): Numrical column use to encode features' values
        feature(string or list of strings): features in dataframe to encoded '''
    
    if feature == None:
        feature = cat_feat_list(data, target)
        
    if type(feature)== str:
        ordered_data = data.groupby(feature).mean().sort_values(target).index.unique() 
        ord_encoder = OrdinalEncoder(categories = [ordered_data], dtype= 'int64')
        data[feature + '_cat']= ord_encoder.fit_transform(data[[feature]])
        data[feature + '_cat'] = data[feature + '_cat'] + 1
        
    elif type(feature) == list:
        encoder_dict = {}
        for i in feature:
            ordered_data = data.groupby(i).mean().sort_values(target).index.unique() 
            ord_encoder  = OrdinalEncoder(categories = [ordered_data], dtype= 'int64')
            data[i + '_cat'] = ord_encoder.fit_transform(data[[i]]) + 1
            encoder_dict[i] = ord_encoder
            
        joblib.dump(encoder_dict, 'features_ordinal_econders')

    print('\n', 'Data encoded')        
    return data


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

#function creates list of features in dataframe
def feat_list(data, target):

    ''' return list of features in dataframe

        Parameters:
        data: dataframe object
        target(string): target is not listed in returned list'''
    
    c= []
    for feature in data.columns:
        if feature != target :
            c.append(feature)
    return c

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


# Fuction determines train and test data split. If user provides a separate test sample it is used, otherwise a portion of train sample is reserved for testing
def use_data(train_sample, target,  num_features = 'auto', test_sample = 'auto'):

    '''Fuction determines train and test split. If user provides a separate test data, it is used, otherwise a portion of train data is taken out for testing
       Returns 'x_train, y_train, x_test, y_test'. See Sklearn train_test_split module

       Parameters
       train_sample: train dataframe object
       test_sample: test dataframe object(optional)
       num_features: List of numerical features (optional)'''
    
    #gets all numerical features from data if auto is used. Otherwise uses the numerical features passed
    if num_features == 'auto':
                num_features = num_feat_list(train_sample, target)
    else: num_features == num_features
    
    #Check
    if type(test_sample) == str:
        x = train_sample[ num_features ]
        y = train_sample [target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

    else:
         x_train = train_sample[num_features]
         y_train = train_sample [target]
         x_test = test_sample[num_features]
         y_test = test_sample[target]
        
    return x_train, y_train, x_test, y_test


#Feature importance plot
def feature_importance(model, num_features):
    '''Plots relative importance of features for a machine learning model

        Parameters:
        model: A classifier or regressor object
        num_features: List of featrues
        '''
    
    feature_importances = []
    pos = np.arange(len(num_features)) + .5
    importance_percentage = []
    
    for i in range(len(num_features)):
        feature_importances.append(model.feature_importances_[i])
    
    for i in feature_importances:
        importance_percentage.append(i*100)
                  
    fig, ax = plt.subplots(figsize=(8,8))
    ax.barh(pos, importance_percentage, align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(num_features)
    ax.set_xlabel('Relative Importance %')
    plt.title('Feature Importance')
    plt.show()



#Linear regression model

def poly_reg(train_sample,  target, order,  num_features = 'auto', test_sample = 'auto'):

    '''plots the mean squared error of multiple linear regression models on train and test data set

        Parameters:
        train_sample: train dataframe
        target(string): target label
        order(list of integers): the orders of the linear regression models
        num_features(list of strings): features used to create the regression models(optiona). If not given it is automatically generated
        test_sample: test dataframe(optional). If not given a portion of the train_data is reserved for testing'''
    

    lrm = LinearRegression()

    #gets all numerical features from data if auto is used
    #if num_features == 'auto':
     #           num_features = num_feat_list(train_sample, target = target)
            
    x_train, y_train, x_test, y_test = use_data(train_sample, target, test_sample, num_features)

  
    mse_train = []
    mse_test = []
    for i in order:
        pr = PolynomialFeatures(degree=i)
        x_train_pr = pr.fit_transform(x_train)
        x_test_pr = pr.fit_transform(x_test)

        lrm.fit(x_train_pr, y_train)

        yhat_train = lrm.predict(x_train_pr)
        yhat_test = lrm.predict(x_test_pr)

        mse_train.append(mean_squared_error(y_train, yhat_train))
        mse_test.append(mean_squared_error(y_test, yhat_test))
      
    mse_train
    mse_test

    fig, (ax2) = plt.subplots(1,1, figsize=(10,10)) 

    ax2.plot(order, mse_train, color = 'red' , marker = '*', label = 'train' )

    ax2.plot(order, mse_test, color = 'green', marker = '*', label = 'test' )

    ax2.set_xlabel('Polynomial order')
    ax2.set_ylabel('MSE')
    ax2.legend(loc= 'lower left')

    plt.show


#===================

#Compares performance of 4 models- Decision Tree, Random Forest, Gradient boost, XGB
def best_model_mse(train_sample,  target,  num_features = 'auto', test_sample = 'auto'):

    ''' Compares performance of 4 models- Decision Tree, Random Forest, Gradient boost, XGBoost
        Prints out cross validation score for each model
        Prints out the mse for each model
        save an object of the model with the best mse as 'saved_best_model'
        and graphs the feature importance of the best model
        Returns an object of the best model

        Parameters:
        train_sample: train dataframe
        target(string): target label
        num_features(list of strings): features used to train models(optional). If not given it is automatically generated
        test_sample: test dataframe(optional). If not given a portion of the train_data is reserved for testing'''
    
    
    #gets all numerical features from data if auto is used
    if num_features == 'auto':
                num_features = num_feat_list(train_sample, target)
            
    x_train, y_train, x_test, y_test = use_data(train_sample, target, num_features, test_sample )

#models
    dt = tree.DecisionTreeRegressor(max_depth= 30, max_features = 'auto', 
                                        min_samples_split=130, random_state = 1, ccp_alpha = 0.0001 )

    rf = RandomForestRegressor(n_estimators = 100, min_samples_split = 20, min_samples_leaf = 8, verbose = 0,
                               max_features = 'sqrt', max_samples= 100000, random_state = 1, oob_score= True, n_jobs = -1, bootstrap = True )

    grad_bst = GradientBoostingRegressor(n_estimators = 100, max_depth= 9, max_features = 5, learning_rate = 0.1, 
                                   min_samples_split = 8, min_samples_leaf = 8, verbose = 0, random_state = 3)


    xg_bst = xgboost.XGBRegressor(n_jobs = -1, random_state = 2, n_estimators = 100, max_depth = 8, min_child_weight= 8, 
                               reg_lambda = 1, reg_alpha = 20, learning_rate = 0.09, base_score= 0.5, colsample_bytree = 0.8, subsample = 1)


    models = {dt : 'Dec_tree', rf: 'rand_forest', grad_bst : 'gradient_boost', xg_bst : 'xtreme_grad_boost'}
    #models = {dt : 'Dec_tree', rf: 'rand_forest'}

    #print ('CROSS VAL MSE SCORES')

    #for keys in models.keys():
    #    cross = cross_validate(keys, x_train, y_train, n_jobs= -1, scoring = ('neg_mean_squared_error'),
    #                           cv=5, return_train_score=True, return_estimator= False, verbose = 0)
    #   print(models[keys], '----> ', cross['test_score'])

    model_mse = {}
    for keys in models.keys():
        keys.fit(x_train, y_train)

        yhat_test = keys.predict(x_test)

        mse = mean_squared_error(yhat_test, y_test)
        model_mse[models[keys]] = mse

    print('MODEL MSEs')
    for key,value in model_mse.items():
        print(key, '--->', value)
        
    best_mod = min(model_mse.keys(), key=(lambda k: model_mse[k]))
    
    for key, value in models.items():
        if value == best_mod:
            joblib.dump(key, 'saved_best_model')
            
            print ('\n', 'Best model ----> ', value, '  (saved as: saved_best_model)')
         
            
            feature_importance(key, num_features)
            
            return key

#========================
def xg_boost(train_sample,  target,  num_features = 'auto', test_sample = 'auto', estimator = None):

    '''Extreme gradient boost regressor model. Trains on enitre train set when test_sample is set to 'None' and return model for production

    Parameters:
        train_sample: train dataframe
        target(string): target label
        num_features(list of strings): features used to train models(optional). If not given it is automatically generated
        test_sample: test dataframe(optional). If not given a portion of the train_data is reserved for testing. To train model on entire train data set test_sample== None
        estimator: extreme gradient boost object to train data. If none is provided, a default is used'''
    
    
    if estimator == None:

        xg_bst = xgboost.XGBRegressor(n_jobs = -1, random_state = 2, n_estimators = 100, max_depth = 8, min_child_weight= 8, 
                           reg_lambda = 1, reg_alpha = 20, learning_rate = 0.09, base_score= 0.5, colsample_bytree = 0.8, subsample = 1)
    else: 
        xg_bst = estimator
           
    #gets all numerical features from data if auto is used
    if num_features == 'auto':
                num_features = num_feat_list(train_sample, target)

    #trains on entire data set to get final model        
    if test_sample == None:
        x_train = train_sample[ num_features ]
        y_train = train_sample [target]
        
        xg_bst.fit(x_train, y_train)
        yhat_train = xg_bst.predict(x_train)
        train_mse = mean_squared_error(yhat_train, y_train)
        print('\n', 'The mean squared error for entire train data ', train_mse)

        feature_importance(xg_bst, num_features)
        
        model_file_name = 'final_xgb_model_saved'
        joblib.dump(xg_bst, model_file_name )
        print ('final xgb model saved as --->', model_file_name)
        
        return xg_bst
        
    #splits train sample for train and test or uses test sample provided        
    else: 
        
        x_train, y_train, x_test, y_test = use_data(train_sample, target, test_sample, num_features )
        
        #commented because it takes long to run
        #xgb_cross = cross_validate(xg_bst, x_train, y_train, n_jobs= -1, scoring = ('neg_mean_squared_error'), 
                             # cv=5, return_train_score=True, return_estimator= False, verbose = 3)
        #print('Cross validation scores:', '\n', rf_cross, )

        xg_bst.fit(x_train, y_train)
        
        feature_importance(xg_bst, num_features)     

        yhat_train = xg_bst.predict(x_train)
        yhat_test = xg_bst.predict(x_test)
        
        train_mse = mean_squared_error(yhat_train, y_train)
        mse = mean_squared_error(yhat_test, y_test)
        
        print('\n','The train mean squared error', train_mse)
        print('The test mean squared error ', mse)
        
        fig = plt.figure()

        ax1=  sns.distplot(y_test, hist= False, color = 'red', label= 'Actual')

        sns.distplot(yhat_test, hist=False, color = 'blue', ax=ax1, label= 'Predicted')
        
        ax1.set_title(label = "TEST SAMPLE DISTRIBUTION")
        
        
        joblib.dump(xg_bst, 'saved_xgb_model' )

        
        return xg_bst
#================

#cleans, processes raw train data and returns model for predicting
def combined_proc_modelling(train_features_link, train_target_link):

    ''' Function automates loading, cleaning, prepocessing of train data, then the training of entire train data on xgb model and returns trained model
        which is saved for test data prediction

        Parameters:
        train_feature_link(string): file path of train features data
        train_target_link(string): file path of train target data '''
    
    train_features = upload_file_csv(train_features_link) #get train features
    train_target = upload_file_csv(train_target_link)     #get train target
    
    x = merger(train_features, train_target, 'jobId', how= 'left') #merge features and target data set
    
    y = case_check(x) #checks that cases in each feature are uniform and corrects otherwise
    
    z = drop_missing(y, 'salary') #drops rows with missing values and rows with target values less than or equal to 0 
    
    m = drop_dup(drop_feature(z, 'jobId' )) #further removes duplicate rows after droping jobId column
    
    r = cat_ord_enc(m, 'salary')   #endcodes categorical features
    
    b = add_binned_feat(r, 'yearsExperience', 6, [1,2,3,4,5,6], 'yearsExp_cat') #bins years experience and milesfromMetropolis feautures
    a = add_binned_feat(b, 'milesFromMetropolis', 10, [i for i in range(10,0,-1)], 'mfm_cat') #average salary inversly propotional to milesFromMetropolis
    
    
    # trains entire processed data using xtreme gradeint boost algorithm to create final model that is saved   
    final_model = xg_boost(a,  'salary',  num_features = 'auto', test_sample = None, estimator = None) 

    joblib.dump(final_model, 'final_xgb_model_saved')
    
    return final_model


#predicts target given cleaned data and model for prediction
def model_predict(data, model):
   
    num_features = num_feat_list(data, 'salary')
    
    x_pred = data[ num_features ]
    
    y_pred = model.predict(x_pred)
    
    return print(y_pred)
    #return print(int(np.round(y_pred[0]*1000, -1)))


#clean and process test data, predicts target and exports data with predictions as csv file
def process_predict_pipe(test_features_link):

    ''' Function automates loading, cleaning, prepocessing of test data, predicts target using saved machine learning model which is saved as a csv file

        Parameters:
        test_feature_link(string): file path of test data '''
    
    x = upload_file_csv(test_features_link)
    
    y = case_check(x) #checks that cases in each feature are uniform and corrects otherwise
    
    m = drop_feature(y, 'jobId' ) # droping jobId column
    
    r = test_ord_enc(m)   #endcodes categorical features
    
    b = add_binned_feat(r, 'yearsExperience', 6, [1,2,3,4,5,6], 'yearsExp_cat') #bins years experience and milesfromMetropolis feautures
    a = add_binned_feat(b, 'milesFromMetropolis', 10, [i for i in range(10,0,-1)], 'mfm_cat') #order of range reversed because average salary inversly propotional to milesFromMetropolis
    
    model = joblib.load('final_xgb_model_saved')
    
    salary_predictions = model_predict(a, model)
    
    x['predicted_salary'] = salary_predictions
    
    output_file = 'predicted_salary.csv'
    
    x.to_csv(output_file)
    print('Predictions saved as csv to ---> ', output_file )
    
    return x


#function for prediction in web_app

def predict(to_predict_):

    #convert to pandas dataframe
    input_table = pd.DataFrame(to_predict_list, index = [1]) 

    #cast to numeric
    input_table = input_table.astype({'milesFromMetropolis': 'int64'})
    input_table = input_table.astype({'yearsExperience' : 'int64'})

    #encode inputs
    input_table = test_ord_enc(input_table)

    #bin years of experience and distance from metro
    input_table = add_binned_feat(input_table, 'yearsExperience', 6, [1,2,3,4,5,6], 'yearsExp_cat') #bins years experience and milesfromMetropolis feautures
    #order of range reversed because average salary inversly propotional to milesFromMetropolis
    input_table = add_binned_feat(input_table, 'milesFromMetropolis', 10, [i for i in range(10,0,-1)], 'mfm_cat') 

    model = joblib.load('final_xgb_model_saved')

    num_features = num_feat_list(input_table, 'salary')

    x_pred = input_table[ num_features ]

    y_pred = model.predict(x_pred)

    return print(int(np.round(y_pred[0]*1000, -1)))

        