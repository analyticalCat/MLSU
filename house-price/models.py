import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, ElasticNetCV
from scipy.stats import skew
import matplotlib.pyplot as plt
import csv
from sklearn.svm import SVR
import sys
import xgboost as xgb



#load data
train = pd.read_csv("data/train_update.csv")
test = pd.read_csv("data/test_update.csv")

y_train=train.SalePrice
X_train=train.drop("SalePrice", axis=1)
print("train : " + str(X_train.shape)) #1458, 286
print("test : " + str(test.shape))  #1459, 285


################
if str(sys.argv[1]) == 'xgboost':
    
    model = xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
    predictions = model.predict(test)
    #tried learning rate 0.01, 0.03, 0.04, 0.05, 0.06, 0.1.  0.05 is the best.
    #then try estimator 100, 200, 300, 400
    
    feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

####################
if str(sys.argv[1]) == 'decisiontree':
    from sklearn.tree import DecisionTreeRegressor
    # Define model
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    #Decision tree on all features scored 0.202


####################
if str(sys.argv[1]) == 'randomforest':
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()  
    # with all features, random forest scored 0.16173

    # Fit model
    model.fit(X_train, y_train)


#predict the test dataset

X_test=test
predictions=model.predict(X_test)

if len(sys.argv)>2:
    if str(sys.argv[2]) == 'log':
        #since we are using log transformation, we need to transform the result back
        predictions = np.expm1(predictions)
        #df=pd.DataFrame(predictions)



###############################
#try linear regression with log transformation.  TODO: wait until later
#if str(sys.argv[1]) == 'linearregression':
#from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV



####################
if str(sys.argv[1]) == 'SVM': 
    #SVG_rbf 0.4 -- the worst of all methods. not a good method.
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X_train, y_train).predict(test)
    y_lin = svr_lin.fit(X_train, y_train).predict(test)
    y_poly = svr_poly.fit(X_train, y_train).predict(test)



#Write result to datafile

result_dataset = pd.DataFrame({'Id':test.Id,'SalePrice':predictions})
result_dataset.to_csv("data/submission.csv",index=False)




#print("shape of result: ", result_dataset.shape)
#print(result_dataset)


