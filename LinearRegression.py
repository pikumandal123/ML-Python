# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:32:27 2020

@author: Saptarshi_Mandal
"""

#***********  Simple Linear Regression  ***********

# %% Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


#%% Read data
df = pd.read_csv('FuelConsumptionCo2.csv')

# take a look at the data
df.head()
print(df.columns)

#---- summarize the data (get descriptive stats)
df.describe()

# select some features to explore
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS' ]] # get all rows for the colnames given
cdf.head()

# lets plot some data
viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS' ]]
viz.hist() # generate histogram
plt.show() # show the plot 


#%% Plot X(independent) vs Y(dependent) data
#----  Fuel consumption  -----
plt.scatter(x = cdf.FUELCONSUMPTION_COMB, y = cdf.CO2EMISSIONS,  color = 'red')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('EMISSION')
plt.show()

#----  Cylinder VS CO2 -----
plt.scatter(x = cdf.CYLINDERS, y = cdf.CO2EMISSIONS,  color = 'red')
plt.xlabel('CYLINDERS')
plt.ylabel('EMISSION')
plt.show()


#%% TEST/TRAIN split
# Test/Train = 20/80 

#----  plot the random number generated  ------
a = np.random.rand(len(df))
plt.plot(a, 'ro')
plt.show()

#---- generate random number to divide test and train  ---
rand_rows =  np.random.rand(len(df))< 0.8
train = cdf[rand_rows]
test  = cdf[~rand_rows]


#%% Model the Regression model
from sklearn.linear_model import LinearRegression as lr

train_x = np.asanyarray(train[['ENGINESIZE']]) # x data
train_y = np.asanyarray(train[['CO2EMISSIONS']]) # y data

#--- fit regression model ----
reg_model = lr().fit(train_x, train_y)

#--- get the coefficients  ----
beta_1 = reg_model.coef_[0][0]  # get slope
beta_0 = reg_model.intercept_[0]  # get intercept
print('Coefficients: ', beta_1)
print('Intercept: ', beta_0 )


#%% Plot the outputs of the linear regression model

#---  plot the data points  ----
plt.scatter(train_x, train_y, color = 'blue') # plot raw data

#--- calculate y values using regression model ----
predicted_y = beta_0 + beta_1*train_x # calculate y values
plt.plot(train_x, predicted_y, '-r' ) # plot the line
plt.xlabel('Engine size')
plt.ylabel('Emission')


#%% Model Evaluation
from sklearn.metrics import r2_score

#---- extract test data sets ----
test_x = test[['ENGINESIZE']]
test_y = test[['CO2EMISSIONS']]

#--- use the fitted model to predict new dependent values
test_y_predicted = reg_model.predict(test_x) 

#--- calculate error values  ---
mse = np.mean((test_y - test_y_predicted)**2) # mean square error
r_square = r2_score(test_y, test_y_predicted) # R-square value

print('Mean square error: %.2f' %mse)
print('R-Square : %.2f' %r_square)














