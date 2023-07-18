import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
#import r2
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import make_scorer



def get_starbucks_Q1_2023_data_for_prediction():
    # prepare 2023 quarter 1 data for prediction of quarter 2 revenue
    this_is_it = pd.read_csv('values_for_prediction_ford_adjusted.csv')

    # select statistically significant features
    starbucks_revenue_prediction = this_is_it[['population','median_house_income', 'unemp_rate',
                                           'home_ownership_rate', 'government_spending',
                                           'gdp_deflated','violent_crime_rate',
                                           'cpi_all_items_avg','eci', 'dow', 's_and_p', 
                                           'Man_new_order', 'hdi', 'auto_loan', 'velocity_of_money', 
                                           'wti', 'brent_oil', 'case_shiller_index', 'number_of_disaster',
                                           'c_e_s_housing', 'c_e_s_health','ease_of_doing_business']]
    return starbucks_revenue_prediction

# scale data
def starbucks_scaled_df(train, test, starbucks_revenue_prediction):
    """
    This function scales the train, validate, and test data using the MinMaxScaler.

    Parameters:
    train (pandas DataFrame): The training data.
    test (pandas DataFrame): The test data.
    ford_revenue_prediction (pandas DataFrame): The data for Ford revenue prediction.

    Returns:
    Tuple of:
        X_train_scaled (pandas DataFrame): The scaled training data.
        X_test_scaled (pandas DataFrame): The scaled test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        X_train (pandas DataFrame): The original training data.
        ford_revenue_prediction_scaled (pandas DataFrame): The scaled Ford revenue prediction data.
    """

    X_train = train[['population','median_house_income', 'unemp_rate',
                     'home_ownership_rate', 'government_spending',
                     'gdp_deflated','violent_crime_rate',
                     'cpi_all_items_avg','eci', 'dow', 's_and_p', 
                     'Man_new_order', 'hdi', 'auto_loan', 'velocity_of_money', 
                     'wti', 'brent_oil', 'case_shiller_index', 'number_of_disaster',
                     'c_e_s_housing', 'c_e_s_health','ease_of_doing_business']]
    X_test = test[['population','median_house_income', 'unemp_rate',
                   'home_ownership_rate', 'government_spending',
                   'gdp_deflated','violent_crime_rate',
                   'cpi_all_items_avg','eci', 'dow', 's_and_p', 
                   'Man_new_order', 'hdi', 'auto_loan', 'velocity_of_money', 
                   'wti', 'brent_oil', 'case_shiller_index', 'number_of_disaster',
                   'c_e_s_housing', 'c_e_s_health','ease_of_doing_business']]

    y_train = train.adjusted_revenue_S
    y_test = test.adjusted_revenue_S

    # Making our scaler
    scaler = MinMaxScaler()
    
    # Fitting our scaler and using it to transform train and test data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns,
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_test.columns,
                                 index=X_test.index)
    
    # Scaling the Ford revenue prediction data
    starbucks_revenue_prediction_scaled = pd.DataFrame(scaler.transform(starbucks_revenue_prediction.values.reshape(1, -1)),
                                                  columns=starbucks_revenue_prediction.columns,
                                                  index=starbucks_revenue_prediction.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, starbucks_revenue_prediction_scaled

# create helper function for putting results into a dataframe
def starbucks_metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

# create the baseline model and post to dataframe
def starbucks_baseline_model(train, y_train):
    """
    Creates a baseline model using the mean of the target variable and evaluates its performance.

    Parameters:
        train (pandas DataFrame): The training data containing the feature variables.
        y_train (pandas Series): The target variable for the training data.

    Returns:
        pandas DataFrame: A DataFrame containing the evaluation metrics of the baseline model.

    The function creates a baseline model by setting the predicted value as the mean of the target variable (y_train).
    It calculates the root mean squared error (RMSE) and R^2 score of the baseline model using the y_train values
    and an array filled with the mean value. The RMSE and R^2 score are added to a DataFrame for comparison.

    Additionally, the function prints the baseline value and returns the DataFrame with the evaluation metrics.
    """
    #set baseline
    baseline = round(y_train.mean(),2)

    #make an array to send into my mean_square_error function
    baseline_array = np.repeat(baseline, len(train))

    # Evaluate the baseline rmse and r2
    rmse, r2 = starbucks_metrics_reg(y_train, baseline_array)

    # add results to a dataframe for comparison
    starbucks_metrics_df = pd.DataFrame(data=[
    {
        'model':'Baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    
    # print baseline
    baseline = round(y_train.mean(),2)
    print(f' Baseline mean is : {baseline}')
    return starbucks_metrics_df

def starbucks_LassoLars_model(X_train_scaled, y_train, starbucks_metrics_df):
    """
    Performs LassoLars regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        y_train (pandas Series): The target variable for the training data.
        starbucks_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the LassoLars model.
    """
    # Define the model and the hyperparameter grid
    model = LassoLars(normalize=False)
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01,0.1,0.25,0.5,0.75, 1],  
        'normalize': [True, False]
    }
    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model rmse and r2
    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)

    # Add evaluation metrics to the provided metrics DataFrame
    starbucks_metrics_df.loc[1] = ['LassoLars', abs(best_score), best_r2]

    return starbucks_metrics_df

def starbucks_Generalized_Linear_Model(X_train_scaled, y_train, starbucks_metrics_df):
    """
    Fits a Generalized Linear Model (GLM) and evaluates its performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        y_train (pandas Series): The target variable for the training data.
        starbucks_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the GLM.
    """
    # Define the model and the hyperparameter grid
    model = TweedieRegressor()
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01,0.1,0.25,0.5,0.75, 1],
        'power': [0, 1, 2]  # Example values for power
    }
    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    #best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)
   
    # Add evaluation metrics to the provided metrics DataFrame
    starbucks_metrics_df.loc[2] = ['Generalized Linear Model', abs(best_score), best_r2]

    return starbucks_metrics_df

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def starbucks_polynomial_regression(X_train_scaled, y_train, starbucks_metrics_df):
    """
    Performs polynomial regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the polynomial regression model.
    """
    # Create the pipeline
    pipeline = Pipeline([
        ('polynomialfeatures', PolynomialFeatures()),
        ('linearregression', LinearRegression())
    ])

    # Define the hyperparameter grid
    param_grid = {
        'polynomialfeatures__degree': [1, 2, 3, 4, 5]  
    }
    # Define the scoring functions
    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring = scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_

    # Get the best model rmse and r2
    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)
    

    # Add evaluation metrics to the provided metrics DataFrame
    starbucks_metrics_df.loc[3] = ['Polynomial Regression(PR)', abs(best_score), best_r2]

    return starbucks_metrics_df, best_model

def starbucks_prediction_Q2_2023_revenue(model, starbucks_revenue_prediction_scaled):


    # Pass the preprocessed single line of data to the best_model
    pred_value = model.predict(starbucks_revenue_prediction_scaled)

    # Print the predicted value
    print('Starbucks predicted 2023 Q2 revenue is' , pred_value)

def best_model_on_test(X_test_scaled, best_model, y_test):
    sb_pred_test = best_model.predict(X_test_scaled)

    rmse, r2 = starbucks_metrics_reg(y_test, sb_pred_test)

    return rmse, r2

def get_ford_Q1_2023_data_for_prediction():
    
    # prepare 2023 quarter 1 data for prediction of quarter 2 revenue
    this_is_it = pd.read_csv('values_for_prediction_ford_adjusted.csv')

    # select statistically significant features
    ford_revenue_prediction = this_is_it[['population','median_house_income','misery_index',
                                      'gdp_deflated','violent_crime_rate','cpi_all_items_avg',
                                      'eci','prime', 'gini', 'hdi', 'cli','velocity_of_money',
                                      'consumer_confidence_index','c_e_s_health',
                                      'ease_of_doing_business']]
    return ford_revenue_prediction

def ford_scaled_df(train, test, ford_revenue_prediction):
    """
    Performs polynomial regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        y_train (pandas Series): The target variable for the training data.
        starbucks_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the polynomial regression model.
    """

    X_train = train[['population', 'median_house_income', 'misery_index',
                     'gdp_deflated', 'violent_crime_rate', 'cpi_all_items_avg',
                     'eci', 'prime', 'gini', 'hdi', 'cli', 'velocity_of_money',
                     'consumer_confidence_index', 'c_e_s_health',
                     'ease_of_doing_business']]
    X_test = test[['population', 'median_house_income', 'misery_index',
                   'gdp_deflated', 'violent_crime_rate', 'cpi_all_items_avg',
                   'eci', 'prime', 'gini', 'hdi', 'cli', 'velocity_of_money',
                   'consumer_confidence_index', 'c_e_s_health',
                   'ease_of_doing_business']]

    y_train = train.adjusted_revenue_B
    y_test = test.adjusted_revenue_B

    # Making our scaler
    scaler = MinMaxScaler()
    
    # Fitting our scaler and using it to transform train and test data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns,
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_test.columns,
                                 index=X_test.index)
    
    # Scaling the Ford revenue prediction data
    ford_revenue_prediction_scaled = pd.DataFrame(scaler.transform(ford_revenue_prediction.values.reshape(1, -1)),
                                                  columns=ford_revenue_prediction.columns,
                                                  index=ford_revenue_prediction.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, ford_revenue_prediction_scaled

def ford_metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def ford_baseline_model(train, y_train):
    """
    Creates a baseline model using the mean of the target variable and evaluates its performance.

    Parameters:
        train (pandas DataFrame): The training data containing the feature variables.
        y_train (pandas Series): The target variable for the training data.

    Returns:
        pandas DataFrame: A DataFrame containing the evaluation metrics of the baseline model.

    The function creates a baseline model by setting the predicted value as the mean of the target variable (y_train).
    It calculates the root mean squared error (RMSE) and R^2 score of the baseline model using the y_train values
    and an array filled with the mean value. The RMSE and R^2 score are added to a DataFrame for comparison.

    Additionally, the function prints the baseline value and returns the DataFrame with the evaluation metrics.
    """
    #set baseline
    baseline = round(y_train.mean(),2)

    #make an array to send into my mean_square_error function
    baseline_array = np.repeat(baseline, len(train))

    # Evaluate the baseline rmse and r2
    rmse, r2 = ford_metrics_reg(y_train, baseline_array)

    # add results to a dataframe for comparison
    ford_metrics_df = pd.DataFrame(data=[
    {
        'model':'Baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    
    # print baseline
    baseline = round(y_train.mean(),2)
    print(f' Baseline mean is : {baseline}')
    return ford_metrics_df

def ford_LassoLars_model(X_train_scaled, y_train, ford_metrics_df):
    """
    Performs LassoLars regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the LassoLars model.
    """
    # Define the model and the hyperparameter grid
    model = LassoLars(normalize=False)
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01,0.1,0.25,0.5,0.75, 1],  
        'normalize': [True, False]
    }
    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its rmse and r2
    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)
  
    # Add evaluation metrics to the provided metrics DataFrame
    ford_metrics_df.loc[1] = ['LassoLars', abs(best_score), best_r2]

    return ford_metrics_df

def ford_Generalized_Linear_Model(X_train_scaled, y_train, ford_metrics_df):
    """
    Fits a Generalized Linear Model (GLM) and evaluates its performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the GLM.
    """
    # Define the model and the hyperparameter grid
    model = TweedieRegressor()
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01,0.1,0.25,0.5,0.75, 1],  # Example values for alpha
        'power': [0, 1, 2]  # Example values for power
    }
    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best models rmse and r2
    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)
   
    # Add evaluation metrics to the provided metrics DataFrame
    ford_metrics_df.loc[2] = ['Generalized Linear Model', abs(best_score), best_r2]

    return ford_metrics_df

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def ford_polynomial_regression(X_train_scaled, y_train, ford_metrics_df):
    """
    Performs polynomial regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the polynomial regression model.
    """
    # Create the pipeline
    pipeline = Pipeline([
        ('polynomialfeatures', PolynomialFeatures()),
        ('linearregression', LinearRegression())
    ])

    # Define the hyperparameter grid
    param_grid = {
        'polynomialfeatures__degree': [1, 2, 3, 4, 5]  
    }
    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring = scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    

    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)
  
    # Add evaluation metrics to the provided metrics DataFrame
    ford_metrics_df.loc[3] = ['Polynomial Regression(PR)', abs(best_score), best_r2]

    return ford_metrics_df, best_model

def ford_prediction_Q2_2023_revenue(model, ford_revenue_prediction_scaled):
    # Pass the preprocessed single line of data to the best_model
    pred_value = model.predict(ford_revenue_prediction_scaled)

    # Print the predicted value
    print("Ford Motor Company's predicted 2023 Q2 revenue is" , pred_value)

def ford_best_model_on_test(X_test_scaled, best_model, y_test):
    sb_pred_test = best_model.predict(X_test_scaled)

    rmse, r2 = ford_metrics_reg(y_test, sb_pred_test)

    return rmse, r2

def get_att_Q1_2023_data_for_prediction():
    # prepare 2023 quarter 1 data for prediction of quarter 2 revenue
    this_is_it = pd.read_csv('values_for_prediction_ford_adjusted.csv')

    # select statistically significant features
    att_revenue_prediction = this_is_it[['home_ownership_rate','hdi','violent_crime_rate',
                                         'ease_of_doing_business','population','c_e_s_health',
                                         'construction_res','federal_fund_rate','auto_loan',
                                         'eci','gdp_deflated','velocity_of_money','cpi_all_items_avg'
                                        ]]
    return att_revenue_prediction

def att_scaled_df(train, test, att_revenue_prediction):
    """
    This function scales the train, validate, and test data using the MinMaxScaler.

    Parameters:
    train (pandas DataFrame): The training data.
    test (pandas DataFrame): The test data.
    ford_revenue_prediction (pandas DataFrame): The data for Ford revenue prediction.

    Returns:
    Tuple of:
        X_train_scaled (pandas DataFrame): The scaled training data.
        X_test_scaled (pandas DataFrame): The scaled test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        X_train (pandas DataFrame): The original training data.
        ford_revenue_prediction_scaled (pandas DataFrame): The scaled Ford revenue prediction data.
    """

    X_train = train[['home_ownership_rate','hdi','violent_crime_rate',
 'ease_of_doing_business','population','c_e_s_health',
 'construction_res','federal_fund_rate','auto_loan',
 'eci','gdp_deflated','velocity_of_money','cpi_all_items_avg'
]]
    X_test = test[['home_ownership_rate','hdi','violent_crime_rate',
 'ease_of_doing_business','population','c_e_s_health',
 'construction_res','federal_fund_rate','auto_loan',
 'eci','gdp_deflated','velocity_of_money','cpi_all_items_avg'
]]

    y_train = train.adjusted_revenue_A
    y_test = test.adjusted_revenue_A

    # Making our scaler
    scaler = MinMaxScaler()
    
    # Fitting our scaler and using it to transform train and test data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns,
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_test.columns,
                                 index=X_test.index)
    
    # Scaling the Ford revenue prediction data
    att_revenue_prediction_scaled = pd.DataFrame(scaler.transform(att_revenue_prediction.values.reshape(1, -1)),
                                                  columns=att_revenue_prediction.columns,
                                                  index=att_revenue_prediction.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, att_revenue_prediction_scaled

def att_metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def att_baseline_model(train, y_train):
    """
    Creates a baseline model using the mean of the target variable and evaluates its performance.

    Parameters:
        train (pandas DataFrame): The training data containing the feature variables.
        y_train (pandas Series): The target variable for the training data.

    Returns:
        pandas DataFrame: A DataFrame containing the evaluation metrics of the baseline model.

    The function creates a baseline model by setting the predicted value as the mean of the target variable (y_train).
    It calculates the root mean squared error (RMSE) and R^2 score of the baseline model using the y_train values
    and an array filled with the mean value. The RMSE and R^2 score are added to a DataFrame for comparison.

    Additionally, the function prints the baseline value and returns the DataFrame with the evaluation metrics.
    """
    #set baseline
    baseline = round(y_train.mean(),2)

    #make an array to send into my mean_square_error function
    baseline_array = np.repeat(baseline, len(train))

    # Evaluate the baseline rmse and r2
    rmse, r2 = att_metrics_reg(y_train, baseline_array)

    # add results to a dataframe for comparison
    att_metrics_df = pd.DataFrame(data=[
    {
        'model':'Baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    
    # print baseline
    baseline = round(y_train.mean(),2)
    print(f' Baseline mean is : {baseline}')
    return att_metrics_df

def att_LassoLars_model(X_train_scaled, y_train, att_metrics_df):
    """
    Performs LassoLars regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the LassoLars model.
    """
    # Define the model and the hyperparameter grid
    model = LassoLars(normalize=False)
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01,0.1,0.25,0.5,0.75, 1],
        'normalize': [True, False]
    }

    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring = scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its rmse and r2
    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)
    
    # Add evaluation metrics to the provided metrics DataFrame
    att_metrics_df.loc[1] = ['LassoLars', abs(best_score), best_r2]

    return att_metrics_df

def att_Generalized_Linear_Model(X_train_scaled, y_train, att_metrics_df):
    """
    Fits a Generalized Linear Model (GLM) and evaluates its performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the GLM.
    """
    # Define the model and the hyperparameter grid
    model = TweedieRegressor()
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01,0.1,0.25,0.5,0.75, 1],  
        'power': [0, 1, 2]  
    }

    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring = scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its best rmse and r2
    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)
    
    # Add evaluation metrics to the provided metrics DataFrame
    att_metrics_df.loc[2] = ['Generalized Linear Model(GLM)', abs(best_score), best_r2]

    return att_metrics_df

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def att_polynomial_regression(X_train_scaled, y_train, att_metrics_df):
    """
    Performs polynomial regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the polynomial regression model.
    """
    # Create the pipeline
    pipeline = Pipeline([
        ('polynomialfeatures', PolynomialFeatures()),
        ('linearregression', LinearRegression())
    ])

    # Define the hyperparameter grid
    param_grid = {
        'polynomialfeatures__degree': [1, 2, 3, 4, 5]  
    }

    scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'r2': make_scorer(r2_score),
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring = scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best models rmse and r2 
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train_scaled, y_train)

    # Add evaluation metrics to the provided metrics DataFrame
    att_metrics_df.loc[3] = ['Polynomial Regression(PR)', abs(best_score), best_r2]

    return att_metrics_df, best_model

def att_prediction_Q2_2023_revenue(best_model, att_revenue_prediction_scaled):
    # Pass the preprocessed single line of data to the best_model
    pred_value = best_model.predict(att_revenue_prediction_scaled)

    # Print the predicted value
    print("ATT's predicted 2023 Q2 revenue is" , pred_value)

def att_best_model_on_test(X_test_scaled, best_model, y_test):
    sb_pred_test = best_model.predict(X_test_scaled)

    rmse, r2 = ford_metrics_reg(y_test, sb_pred_test)

    return rmse, r2