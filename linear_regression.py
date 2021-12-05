# linear regression part of the project

# load csv
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

global xtrain
global xtest
global ytrain
global ytest
global main_model

global min_aic
min_aic = 200000000
global min_bic
min_bic = 200000000
global min_cp
min_cp = 2000000000
global max_rsq
max_rsq = 0
global max_rsq_adj
max_rsq_adj = 0

global subset_aic
subset_aic = ""
global subset_bic
subset_bic = ""
global subset_rsq
subset_rsq = ""
global subset_rsq_adj
subset_rsq_adj = ""
global subset_cp
subset_cp = ""


traindata = pd.read_csv('/Users/kscott/Documents/Columbia/Linear Regression Models/Project/Project-5205/ks_train.csv')
testdata = pd.read_csv('/Users/kscott/Documents/Columbia/Linear Regression Models/Project/Project-5205/ks_test.csv')

ytrain = traindata['price']
xtrain = traindata.drop(columns=['price'])

ytest = testdata['price']
xtest = testdata.drop(columns=['price'])

sm.add_constant(1)
full_model = sm.OLS(ytrain, xtrain).fit()
full_model_mse = full_model.mse_resid

for subset in itertools.combinations(xtrain.columns, 15):
    # get the model and aic, bic, rsquared, rsquared adj, cp
    sm.add_constant(1)
    model = sm.OLS(ytrain, xtrain[np.asarray(subset)]).fit()
    aic = model.aic
    bic = model.bic
    rsq = model.rsquared
    rsq_adj = model.rsquared_adj
    cp = model.aic / full_model_mse

    # see if stats are improved
    if aic < min_aic:
        subset_aic = subset
    if bic < min_bic:
        subset_bic = subset
    if rsq > max_rsq:
        subset_rsq = subset
    if rsq_adj > max_rsq_adj:
        subset_rsq_adj = subset
    if cp < min_cp:
        subset_cp = subset


#aic model
#sm.add_constant(1)
print(subset_aic)
aic_model = sm.OLS(ytrain, xtrain[np.asarray(subset_aic)]).fit()

#bic model
#sm.add_constant(1)
print(subset_bic)
bic_model = sm.OLS(ytrain, xtrain[np.asarray(subset_bic)]).fit()

#rsq model
#sm.add_constant(1)
print(subset_rsq)
rsq_model = sm.OLS(ytrain, xtrain[np.asarray(subset_rsq)]).fit()

#rsq adj model
print(subset_rsq_adj)
#sm.add_constant(1)
rsq_adj_model = sm.OLS(ytrain, xtrain[np.asarray(subset_rsq_adj)]).fit()

#cp model
#sm.add_constant(1)
print(subset_cp)
cp_model = sm.OLS(ytrain, xtrain[np.asarray(subset_cp)]).fit()

#stepwise model
subset_step=('age','bathrooms','bedrooms', 'condition',
                  'grade', 'lat', 'long', 'renovated', 'sqft_above',
                  'sqft_living', 'sqft_living15', 'sqft_lot15', 'view',
                  'waterfront', 'yr_sale')

step_model = sm.OLS(ytrain, xtrain[np.asarray(subset_step)]).fit()

#get mse for each model, lowest mse = the model we want to try to use for prediction
aic_score = aic_model.mse_resid
bic_score = bic_model.mse_resid
rsq_score = rsq_model.mse_resid
rsq_adj_score = rsq_adj_model.mse_resid
cp_score = cp_model.mse_resid
step_score = step_model.mse_resid

min_score=min(np.asarray([aic_score, bic_score, rsq_score,
                          rsq_adj_score, cp_score, step_score]))

if min_score == aic_score:
    main_model = aic_model
if min_score == bic_score:
    main_model = bic_model
if min_score == rsq_score:
    main_model = rsq_model
if min_score == rsq_adj_score:
    main_model = rsq_adj_model
if min_score == cp_score:
    main_model = cp_model
if min_score == step_score:
    main_model = cp_model

print("aic model %(first)d " % {"first": aic_score})
print("bic model %(first)d " % {"first": bic_score})
print("cp model %(first)d " % {"first": cp_score})
print("rsq model %(first)d " % {"first": rsq_score})
print("rsq_adj model %(first)d " % {"first": rsq_adj_score})
print("stepwise model %(first)d " % {"first": step_score})

print(main_model.summary())

#use main model on test data
pred_values = main_model.predict(xtest[np.asarray(subset_aic)])

#graph performances of model on train and test data
# scatter-plot data
ax = motif.plot(x=main_model.fittedvalues, y=ytrain, kind='scatter')

# plot regression line
abline_plot(main_model, ax=ax)

plt.show()

bx = motif.plot(x=pred_values, y=ytest, kind='scatter')
abline_plit(main_model, ax=bx)

plt.show()



