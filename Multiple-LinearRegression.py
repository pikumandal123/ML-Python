# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 01:19:07 2020

@author: Saptarshi_Mandal
"""

#************ MULTIPLE LINEAR REGRESSION  *************
# %% Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


#%% Read data
df = pd.read_csv('FuelConsumptionCo2.csv')
print(df.columns)


#%% Test/train data divisions

# select some features to explore
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS' ]] # get all rows for the colnames given
cdf.head()

#---- generate random number to divide test and train  ---
rand_rows =  np.random.rand(len(df))< 0.8
train = cdf[rand_rows]
test  = cdf[~rand_rows]


#%% Multiple Regression Model
from sklearn.linear_model import LinearRegression as lr

#--- get train data ----
# extract multiple column for X data
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]) # x data
train_y = np.asanyarray(train[['CO2EMISSIONS']]) # y data

#--- fit multiple regression model ----
mul_reg_model = lr().fit(train_x, train_y)

#--- get the coefficients  ----
coeff = mul_reg_model .coef_  # get slope
intercept = mul_reg_model .intercept_  # get intercept
print('Coefficients: ', coeff)
print('Intercept: ', intercept)


#---- get individual coefficients ---
coeff_enginesize     = coeff[0][0]
coeff_cylinder       = coeff[0][1]
coeff_fuelcomsuption = coeff[0][2]

#%% Model Evaluation
from sklearn.metrics import r2_score

#extract test data sets 
test_x = test[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB']]
test_y = test[['CO2EMISSIONS']]

# use the fitted model to predict new dependent values
test_y_predicted = mul_reg_model.predict(test_x) 

#calculate error values 
mse = np.mean((test_y - test_y_predicted)**2) # mean square error
r_square1 = r2_score(test_y, test_y_predicted) # R-square value

print('Mean square error: %.2f' %mse)
print('R-Square : %.2f' %r_square1)

#%% Fit new multiple regression model
# --- extract new column data
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'CO2EMISSIONS' ]] # get all rows for the colnames given
cdf.head()

#---- generate random number to divide test and train  ---
rand_rows =  np.random.rand(len(df))< 0.8
train = cdf[rand_rows]
test  = cdf[~rand_rows]

#--- get train data ----
# extract multiple column for X data
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]) # x data
train_y = np.asanyarray(train[['CO2EMISSIONS']]) # y data

#--- fit multiple regression model ----
mul_reg_model2 = lr().fit(train_x, train_y)

#---  extract test data sets ---
# add new variables : 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY'
test_x = test[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
test_y = test[['CO2EMISSIONS']]

# use the fitted model to predict new dependent values
test_y_predicted = mul_reg_model2.predict(test_x) 

#calculate error values 
mse       = np.mean((test_y - test_y_predicted)**2) # mean square error
r_square2 = r2_score(test_y, test_y_predicted) # R-square value

print('Mean square error: %.2f' %mse)
print('R-Square : %.2f' %r_square2)

#---- print both rsquare values ----
print('R-square model_1: ', r_square1, "R-square model_2: ", r_square2 )


