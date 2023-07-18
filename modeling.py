import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
#import r2
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import make_scorer



def get_starbucks_Q1_2023_data_for_prediction():
    """
    Retrieves and prepares Quarter 1, 2023 data for Starbucks revenue prediction.

    This function reads the data from the file 'values_for_prediction_ford_adjusted.csv',
    selects statistically significant features relevant for revenue prediction, and returns
    a pandas DataFrame containing these features.

    Returns:
        pandas DataFrame: A DataFrame containing the statistically significant features
                          for Starbucks revenue prediction in Quarter 1, 2023.
    """
    # Read the data for Quarter 1, 2023
    this_is_it = pd.read_csv('values_for_prediction_ford_adjusted.csv')

    # Select statistically significant features for revenue prediction
    starbucks_revenue_prediction = this_is_it[['population', 'median_house_income', 'unemp_rate',
                                               'home_ownership_rate', 'government_spending',
                                               'gdp_deflated', 'violent_crime_rate',
                                               'cpi_all_items_avg', 'eci', 'dow', 's_and_p',
                                               'Man_new_order', 'hdi', 'auto_loan', 'velocity_of_money',
                                               'wti', 'brent_oil', 'case_shiller_index', 'number_of_disaster',
                                               'c_e_s_housing', 'c_e_s_health', 'ease_of_doing_business']]
    return starbucks_revenue_prediction

def starbucks_scaled_df(train, test, starbucks_revenue_prediction):
    """
    Scale the train, validate, and test data using the MinMaxScaler.

    Parameters:
        train (pandas DataFrame): The training data.
        test (pandas DataFrame): The test data.
        starbucks_revenue_prediction (pandas DataFrame): The data for Starbucks revenue prediction.

    Returns:
        Tuple of:
            X_train_scaled (pandas DataFrame): The scaled training data.
            X_test_scaled (pandas DataFrame): The scaled test data.
            y_train (pandas Series): The target variable for the training data.
            y_test (pandas Series): The target variable for the test data.
            X_train (pandas DataFrame): The original training data.
            starbucks_revenue_prediction_scaled (pandas DataFrame): The scaled Starbucks revenue prediction data.
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

def starbucks_metrics_reg(y, yhat):
    """
    Calculate evaluation metrics for regression predictions.

    Parameters:
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values.

    Returns:
        tuple: A tuple containing the Root Mean Squared Error (RMSE) and R-squared (R2) values.
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

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
    """
    # Set baseline
    baseline = round(y_train.mean(), 2)

    # Make an array to send into my mean_square_error function
    baseline_array = np.repeat(baseline, len(train))

    # Evaluate the baseline RMSE and R2
    rmse, r2 = starbucks_metrics_reg(y_train, baseline_array)

    # Add results to a DataFrame for comparison
    starbucks_metrics_df = pd.DataFrame(data=[
        {
            'model': 'Baseline',
            'rmse': rmse,
            'r2': r2
        }
    ])

    # Print baseline
    print(f'Baseline mean is: {baseline}')

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

def starbucks_polynomial_regression(X_train_scaled, y_train, starbucks_metrics_df):
    """
    Performs polynomial regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        y_train (pandas Series): The target variable for the training data.
        starbucks_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the polynomial regression model.
        sklearn Pipeline: The best-fit polynomial regression model.

    The function creates a pipeline for polynomial regression, which includes transforming the features with different
    polynomial degrees and applying linear regression. It then performs hyperparameter tuning to find the best polynomial
    degree using GridSearchCV based on R2 score. The best-fit polynomial regression model and the evaluation metrics
    (RMSE and R2) are added to the provided metrics DataFrame.
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
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, cv=5, refit='r2')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_

    # Get the best model rmse and r2
    best_score = grid_search.best_score_
    best_r2 = best_model.score(X_train_scaled, y_train)

    # Add evaluation metrics to the provided metrics DataFrame
    starbucks_metrics_df.loc[3] = ['Polynomial Regression(PR)', abs(best_score), best_r2]

    return starbucks_metrics_df, best_model

def starbucks_prediction_Q2_2023_revenue(model, starbucks_revenue_prediction_scaled):
    """
    Make revenue predictions for Starbucks in Q2 2023 using the provided model.

    Parameters:
        model: The trained model for revenue prediction.
        starbucks_revenue_prediction_scaled: The preprocessed single line of data containing scaled feature variables
                                            for predicting Starbucks revenue in Q2 2023.

    Returns:
        None

    This function takes a preprocessed single line of data (`starbucks_revenue_prediction_scaled`) containing the
    scaled feature variables needed for revenue prediction in Q2 2023. It uses the provided `model` to make predictions
    for the Starbucks revenue and then prints the predicted value.

    Note: The model should be compatible with the scaled input data (`starbucks_revenue_prediction_scaled`) to ensure
    correct predictions. The predicted revenue value will be displayed as output using the 'print' statement.
    """

    # Pass the preprocessed single line of data to the best_model
    pred_value = model.predict(starbucks_revenue_prediction_scaled)

    # Print the predicted value
    print('Starbucks predicted 2023 Q2 revenue is' , pred_value)

def best_model_on_test(X_test_scaled, best_model, y_test):
    """
    Evaluate the best-fit model on the test dataset.

    Parameters:
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        best_model: The best-fit regression model obtained from hyperparameter tuning.
        y_test (pandas Series): The true target variable for the test data.

    Returns:
        tuple: A tuple containing the Root Mean Squared Error (RMSE) and R-squared (R2) metrics.

    This function evaluates the performance of the best-fit regression model on the test dataset. It uses the model
    to make predictions on the scaled feature variables (`X_test_scaled`) and then compares the predictions with the
    true target variable (`y_test`). The evaluation metrics, RMSE, and R2 are calculated based on this comparison.

    The function returns a tuple containing the RMSE and R2 metrics, providing an assessment of the model's performance
    on the test data. A lower RMSE and a higher R2 indicate better model performance.
    """

    sb_pred_test = best_model.predict(X_test_scaled)

    rmse, r2 = starbucks_metrics_reg(y_test, sb_pred_test)

    return rmse, r2

def get_ford_Q1_2023_data_for_prediction():
    """
    Get Ford's 2023 Q1 data for revenue prediction in Q2.

    Parameters:
        filename (str): The path or name of the CSV file containing the data.

    Returns:
        pandas DataFrame: A DataFrame with statistically significant features for Ford's Q1 2023 revenue prediction.

    This function reads the data for Ford's Q1 2023 revenue prediction from the specified CSV file. It selects
    statistically significant features needed for the prediction, including population, median_house_income, misery_index,
    gdp_deflated, violent_crime_rate, cpi_all_items_avg, eci, prime, gini, hdi, cli, velocity_of_money,
    consumer_confidence_index, c_e_s_health, and ease_of_doing_business.

    The function returns a DataFrame containing these selected features for further use in revenue prediction modeling.
    """

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
    Scale the data for Ford's revenue prediction and prepare the datasets.

    Parameters:
        train (pandas DataFrame): The training data containing the feature variables and the adjusted revenue for Ford.
        test (pandas DataFrame): The test data containing the feature variables and the adjusted revenue for Ford.
        ford_revenue_prediction (pandas DataFrame): The data for Ford's revenue prediction in Q2 2023.

    Returns:
        tuple: A tuple containing the scaled training data, scaled test data, target variables for training and test,
               original training data, and the scaled Ford's Q1 2023 revenue prediction data.

    This function scales the feature variables in both the training and test datasets using MinMaxScaler.
    The function then prepares the necessary datasets for training and evaluation, including scaled training and
    test data, target variables (adjusted revenue) for both training and test data, and the scaled data for Ford's
    revenue prediction in Q2 2023.

    The function returns a tuple containing the scaled datasets and relevant variables for further analysis and modeling.
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
    print(f' Baseline mean is : {baseline}')

    return ford_metrics_df

def ford_LassoLars_model(X_train_scaled, y_train, ford_metrics_df):
    """
    Performs LassoLars regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        y_train (pandas Series): The target variable for the training data.
        ford_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

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
        y_train (pandas Series): The target variable for the training data.
        ford_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

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
        y_train (pandas Series): The target variable for the training data.
        ford_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

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
    """
    Predicts Ford Motor Company's revenue for the second quarter of 2023.

    Parameters:
        model: Trained model to make predictions.
        ford_revenue_prediction_scaled (pandas DataFrame): The scaled feature variables of the Ford revenue prediction data.

    Returns:
        None: The function prints the predicted revenue for Ford Motor Company for the second quarter of 2023.
    """

    # Pass the preprocessed single line of data to the best_model
    pred_value = model.predict(ford_revenue_prediction_scaled)

    # Print the predicted value
    print("Ford Motor Company's predicted 2023 Q2 revenue is" , pred_value)

def ford_best_model_on_test(X_test_scaled, best_model, y_test):
    """
    Evaluates the best model on the test data.

    Parameters:
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        best_model: The best trained model to be evaluated.
        y_test (pandas Series): The target variable for the test data.

    Returns:
        Tuple: Root Mean Squared Error (RMSE) and R^2 score of the best model on the test data.
    """
    ford_pred_test = best_model.predict(X_test_scaled)

    rmse, r2 = ford_metrics_reg(y_test, ford_pred_test)

    return rmse, r2

def get_att_Q1_2023_data_for_prediction():
    """
    Retrieves the ATT's 2023 Quarter 1 data for revenue prediction.

    Returns:
        pandas DataFrame: A DataFrame containing the statistically significant features for predicting ATT's revenue in Quarter 2 of 2023.
    """
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
    att_revenue_prediction (pandas DataFrame): The data for Ford revenue prediction.

    Returns:
    Tuple of:
        X_train_scaled (pandas DataFrame): The scaled training data.
        X_test_scaled (pandas DataFrame): The scaled test data.
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        X_train (pandas DataFrame): The original training data.
        att_revenue_prediction_scaled (pandas DataFrame): The scaled ATT revenue prediction data.
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
        y_train (pandas Series): The target variable for the training data.
        att_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

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
        y_train (pandas Series): The target variable for the training data.
        att_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

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
        y_train (pandas Series): The target variable for the training data.
        att_metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

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
    """
    Make predictions for AT&T's revenue in Q2 2023 using the best selected model.

    Parameters:
        best_model (sklearn model): The best performing model obtained from the evaluation process.
        att_revenue_prediction_scaled (pandas DataFrame): Scaled feature variables for AT&T's revenue prediction.

    Returns:
        None

    This function takes the best performing model obtained from the model evaluation process and the scaled
    feature variables for AT&T's revenue prediction in Q2 2023 as input arguments. It then uses the `predict`
    method of the best_model to make revenue predictions for the specified quarter.

    The predicted revenue value is stored in the variable `pred_value`, and it is printed to the console along
    with an informative message indicating that it represents AT&T's predicted revenue for Q2 2023.

    Note: The `att_revenue_prediction_scaled` should contain the same features used during model training and
    should be preprocessed in the same way as the training data to ensure consistent results.
    """

    # Pass the preprocessed single line of data to the best_model
    pred_value = best_model.predict(att_revenue_prediction_scaled)

    # Print the predicted value
    print("ATT's predicted 2023 Q2 revenue is" , pred_value)

def att_best_model_on_test(X_test_scaled, best_model, y_test):
    """
    Evaluate the best model's performance on the test data for AT&T's revenue prediction.

    Parameters:
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        best_model (sklearn model): The best performing model obtained from the evaluation process.
        y_test (pandas Series): The target variable for the test data.

    Returns:
        Tuple (float, float): The root mean squared error (RMSE) and R^2 score of the best model's predictions.

    This function takes the scaled feature variables of the test data (X_test_scaled), the best_model selected from
    the evaluation process, and the target variable for the test data (y_test). It then uses the best_model to make
    predictions on the test data and calculates the root mean squared error (RMSE) and R^2 score to evaluate the
    performance of the model.

    The function returns a tuple containing the RMSE and R^2 score as the evaluation metrics for the best model.
    """
    att_pred_test = best_model.predict(X_test_scaled)

    rmse, r2 = att_metrics_reg(y_test, att_pred_test)

    return rmse, r2