#import libraries for plotting the results
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot


from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#ingore tensorflow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# a funcition that auto runs the multilayer perceptron model
# uses a nested for loops to try different combinations of parameters
def auto_perceptron(company, X_train_scaled, y_train, X_test_scaled, y_test, this_is_it_scaled):
    #a list of dictionaries to store parameters
    params = []
    # create a list of optimizers to try
    optimizers = ['rmsprop', 'adam']
    # create a list of learning rates to try
    learning_rates = [0.001, 0.01, 0.1]
    # create a list of activation functions to try
    activations = ['relu', 'tanh']
    # create a list of number of neurons to try
    neurons = [8, 13, 21]
    # loop over optimizers
    for opt in optimizers:
        # loop over learning rates
        for rate in learning_rates:
            # loop over activation functions
            for act in activations:
                # loop over number of neurons
                for n in neurons:
                    # create dictionary
                    param = {'optimizer': opt, 'learning_rate': rate, 'activation': act, 'neurons': n}
                    # append to list of dictionaries
                    params.append(param)
    # create lists to store results
    mse, rmse = list(), list()
    # loop over combinations of parameters
    for param in params:
        # create model
        n_features = X_train_scaled.shape[1]
        model = Sequential()
        model.add(Dense(param['neurons'], activation=param['activation'], kernel_initializer='he_normal', input_shape=(n_features,)))
        model.add(Dense(param['neurons'], activation=param['activation'], kernel_initializer='he_normal'))
        model.add(Dense(1))
        # compile the model
        model.compile(optimizer=param['optimizer'], loss='mse')
        # fit the model
        model.fit(X_train_scaled, y_train, epochs=5000, batch_size=32, verbose=0)
        # evaluate the model
        error = model.evaluate(X_test_scaled, y_test, verbose=0)
        # append to lists
        mse.append(error)
        rmse.append(sqrt(error))
        
    # find lowest rmse
    min_rmse, idx = min((val, idx) for (idx, val) in enumerate(rmse))
    # get the best parameter set
    print('Best MSE: %.3f' % mse[idx])
    print('Best RMSE: %.3f' % min_rmse)
    print('Best Parameters: %s' % params[idx])
    #predict on best parameters
    n_features = X_train_scaled.shape[1]
    model = Sequential()
    model.add(Dense(params[idx]['neurons'], activation=params[idx]['activation'], kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(params[idx]['neurons'], activation=params[idx]['activation'], kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer=params[idx]['optimizer'], loss='mse')
    # fit the model
    model.fit(X_train_scaled, y_train, epochs=5000, batch_size=32, verbose=0)
    # make a prediction
    yhat = model.predict(this_is_it_scaled)
    print(f"This is the predictied next quarter revenue for {company}: {yhat} billion dollars")
    #save model with a unique name
    model.save(f"models/{company}_MPL_model.h5")

#a function that loads a model and predicts the next quarter revenue
# takes in the company name and the scaled data
# returns the predicted revenue
def predict_next_quarter(company, X_train_scaled, X_test_scaled, y_test, y_train, this_is_it_scaled):
    #load model
    model = tf.keras.models.load_model(f"models/{company}_MPL_model.h5")
    model.fit(X_train_scaled, y_train, epochs=5000, batch_size=32, verbose=0)
    error = model.evaluate(X_test_scaled, y_test, verbose=0)
    rmse=(sqrt(error))
    #make prediction
    yhat = model.predict(this_is_it_scaled)
    #return prediction
    return yhat, rmse

#loops though predict_next_quarter function 100 and averages the results
# takes in the company name and the scaled data
# returns the average predicted revenue
def predict_next_quarter_avg(company, X_train_scaled, X_test_scaled, y_test, y_train, this_is_it_scaled):
    #create a list to store results
    results = []
    results_rmse = []
    #loop 10 times
    for i in range(10):
        #make prediction
        yhat, rmse = predict_next_quarter(company, X_train_scaled, X_test_scaled, y_test, y_train, this_is_it_scaled)
        #append to list
        results.append(yhat)
        results_rmse.append(rmse)
    #find average
    avg_rmse = sum(results_rmse) / len(results_rmse)
    avg = sum(results) / len(results)
    #return average
    print(f"This is the average predicted next quarter revenue for {company}: {avg} billion dollars")
    print(f"This is the average rmse for {company}: {avg_rmse} billion dollars")

#a function that plots the results of the model
# takes in the company name, the scaled data, and the actual revenue
# returns a plot of the actual revenue and the predicted revenue
def plot_results(company, this_is_it_scaled, y_test):
    #load model
    model = tf.keras.models.load_model(f"models/{company}_MPL_model.h5")
    #make prediction
    yhat = model.predict(this_is_it_scaled)
    #plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Revenue')
    plt.plot(yhat, label='Predicted Revenue')
    plt.title(f"{company} Revenue Prediction")
    plt.xlabel("Quarters")
    plt.ylabel("Revenue (in billions)")
    plt.legend()
    plt.savefig(f"images/{company}_MPL_model.png")
    plt.show()



