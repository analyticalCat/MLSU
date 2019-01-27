import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, ElasticNetCV
from scipy.stats import skew
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import sys
from sklearn.svm import SVR

#first, read the dataset from EDA2.py
#load data
train = pd.read_csv("data/train_update.csv")
test = pd.read_csv("data/test_update.csv")

y_train=train.SalePrice
X_train=train.drop("SalePrice", axis=1)
print("train : " + str(X_train.shape)) #1458, 286
print("test : " + str(test.shape))  #1459, 285


#Feature selection for RF

#Since Alley was listed as least relevant feature, try to remove Alley and see. not better

# X_train = X_train.drop("Alley_Pave", axis = 1 )
# X_train = X_train.drop("Alley_Grvl", axis = 1 )
# test = test.drop("Alley_Pave", axis = 1 )
# test = test.drop("Alley_Grvl", axis = 1 )
# performance not improved.

#try remove bldgtype_xxx?  not better 0.159
#try to remove 'condition2_xxx' 0.154 not better
#cols = [c for c in X_train.columns if c.lower()[:8] != 'bldgtype']
#cols = [c for c in X_train.columns if c.lower()[:10] != 'condition2']
#X_train = X_train[cols] 
#test = test[cols]




#try to remove 3SsnPorch 0.155. not better
#try to remove MiscVal 0.157. not better
#try to remove BsmtHalfBath 0.151.  It's a winner!
#try to remove poolarea, 0.159 not better
#X_train = X_train.drop(['BsmtHalfBath','PoolArea'], axis = 1)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()  
# with all features, random forest scored 0.153 with general feature cleaning in EDA2.

# Fit model
model.fit(X_train, y_train)

#Feature importance
print("Features sorted by their score:")
names = list(X_train)
df2 = pd.DataFrame( sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), 
             reverse=True))
df2.to_csv("data/feature_selection_rf.csv", index=False)


X_test=test
predictions=model.predict(test)

#Write result to datafile

result_dataset = pd.DataFrame({'Id':test.Id,'SalePrice':predictions})
result_dataset.to_csv("data/submission.csv",index=False)