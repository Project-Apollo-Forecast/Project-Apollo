import pandas as pd

def get_data():
    '''
    import csv for exploration
    '''
    df = pd.read_csv('entire_df_ford_starbucks_att_adjusted.csv')

    return df