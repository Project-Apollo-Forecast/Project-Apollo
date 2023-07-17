import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def split_data(df):
    '''
    Take in a DataFrame and perform a train-test split with a 70/30 ratio.
    Return train and test DataFrames.
    '''
    train, test = train_test_split(df, test_size=0.25, random_state=123)
    return train, test

def plot_ford_target(train):
    """
    Visualize the target variable.

    Parameters:
    -----------
    train: pandas DataFrame
        
    Returns:
    --------
    countplot of revenue in the train data set
    """
    sns.histplot(data=train, x='adjusted_revenue_B')
    plt.title('Ford revenue in Billions')
    plt.xlabel('Revenue in Billions')
    plt.ylabel('Quarters')
    plt.show

    return

def check_ford_normalcy(train):
    '''
    Check a data distribution for normalcy
    '''
    #check target for normalcy
    statistic, p_value = stats.shapiro(train.adjusted_revenue_B)

    # Print the test results
    print("Shapiro-Wilk Test")
    print("Statistic:", statistic)
    print("p-value:", p_value)

def plot_att_target(train):
    """
    Visualize the target variable.

    Parameters:
    -----------
    train: pandas DataFrame
        
    Returns:
    --------
    countplot of revenue in the train data set
    """
    sns.histplot(data=train, x='adjusted_revenue_A')
    plt.title('ATT revenue in Billions')
    plt.xlabel('Revenue in Billions')
    plt.ylabel('Quarters')
    plt.show

    return

def check_att_normalcy(train):
    '''
    Check a data distribution for normalcy
    '''
    #check target for normalcy
    statistic, p_value = stats.shapiro(train.adjusted_revenue_A)

    # Print the test results
    print("Shapiro-Wilk Test")
    print("Statistic:", statistic)
    print("p-value:", p_value)

def plot_starbucks_target(train):
    """
    Visualize the target variable.

    Parameters:
    -----------
    train: pandas DataFrame
        
    Returns:
    --------
    countplot of revenue in the train data set
    """
    sns.histplot(data=train, x='adjusted_revenue_S')
    plt.title('Starbucks revenue in Billions')
    plt.xlabel('Revenue in Billions')
    plt.ylabel('Quarters')
    plt.show

    return

def check_starbucks_normalcy(train):
    '''
    Check a data distribution for normalcy
    '''
    #check target for normalcy
    statistic, p_value = stats.shapiro(train.adjusted_revenue_S)

    # Print the test results
    print("Shapiro-Wilk Test")
    print("Statistic:", statistic)
    print("p-value:", p_value)

def spearman_test(df, target_variable, exclude_columns=[], alpha=0.05):
    '''
    Take in a dataframe and run spearmans rank correlation test against a target variable.

    There is an exclude column option, this allows the user the ability to omit any columns 
    not needing to be tested.
    '''
    results = []
    
    for column in df.columns:
        if column == target_variable or column in exclude_columns:
            continue
        
        target_values = df[target_variable]
        column_values = df[column]
        
        correlation, p_value = stats.spearmanr(target_values, column_values)
        if p_value <= alpha:
            result = "Reject the null hypothesis"
        else:
            result = "Fail to reject the null hypothesis"
        
        results.append({'Variable': column, 'P-Value': p_value, 'Result': result})
    
    return pd.DataFrame(results)

def run_starbucks_stats(train):
    '''
    Run the stats function on Starbucks revenue
    '''
    # run spearman test on target
    starbucks_stats = spearman_test(train, target_variable='adjusted_revenue_S', exclude_columns=['adjusted_revenue_A','adjusted_revenue_B','p_election', 'midterm_election'], alpha = 0.05)
    return starbucks_stats

def run_ford_stats(train):
    '''
    Run the stats function on Ford's revenue
    '''
    # Run Spearmanr function on Ford target + other continuous variables in the dataframe (excluding other targets)
    ford_stats = spearman_test(train, target_variable='adjusted_revenue_B', exclude_columns=['adjusted_revenue_S','adjusted_revenue_A','p_election', 'midterm_election'], alpha = 0.05)
    return ford_stats

def run_att_stats(train):
    '''
    Run the stats function on ATTs revenue
    '''
    # Run Spearmanr function on ATT target + other continuous variables in the dataframe (excluding other targets)
    att_stats = spearman_test(train, target_variable='adjusted_revenue_A', exclude_columns=['adjusted_revenue_S','adjusted_revenue_B','p_election', 'midterm_election'], alpha = 0.05)
    return att_stats