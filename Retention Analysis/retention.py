import pandas as pd
import numpy as np
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.linear_model import LinearRegression, ElasticNetCV
# from scipy.stats import skew
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import sys


#load data
data = pd.read_csv("data/train.csv")
#test = pd.read_csv("data/test.csv")
#print(train["Active"].describe())  #The describe is not working on boolean.  moving forward with the missing value check


#1. check for missing values:  there are totally 19 features contain null values.  May need to do something with it.  poolQC(1453), fence(1179)  and MiscFeature(1406)
#todo: we may need to do something about the missing values.  For poolQC and miscFeature, maybe remove these features.
#remove the extra non-value columns 46:
y = data[['ACTIVE']]
train = data.iloc[:, 2:]
train = train.drop("lost_reasons_count", axis = 1)  # duplicate of "loss_reason_count"
# train.fillna(value=0, inplace=True)
# train.to_csv('data/train_update.csv', index=False)


missing = train.isnull().sum()
missing = missing[missing > 0]
print("# of features with missing values:" + str(missing.shape[0]))
missing.sort_values(inplace=True)
print(missing)


#2. Change all category data
train['PLAN'] = pd.Categorical(train['PLAN'])
X = train.iloc[:, 1:]


# Get the correlation metrics
def correlationMatrix(X):
    corMatrix = X.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corMatrix, vmax=.8, square=True)
    plt.show
    plt.close()

# correlationMatrix(X)
# correlation observations:  
# deals -- seats
# activities columns -- seats created
# activities columns -- Deals
# activities columns -- Contacts
# Task_activities_per_entity -- Task_activities_per_entity
# loss_reason_count -- lost_reason_count (duplicate)
# loass_reason_count -- total_deal_lost

#See the Y distribution
def YDistribution(Y):
    sns.set(style="white")
    sns.set(style="whitegrid", color_codes=True)
    sns.countplot(x="ACTIVE", data = Y, palette = 'hls')
    plt.show()
    plt.close()

# YDistribution(y)
# observation: active around 8000, inactive around 7000


#Explore the means of each label
def getMeansoflabels(data):
    data.groupby('ACTIVE').mean().to_csv('data/means.csv', index=False)

# getMeansoflabels(data)
# observation: most catogries the data means are very different between active and inactive counts.  In regression data sheet  on g drive


# Recursive Feature Elimination
def recursiveFeatureElimination(X, y):

    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression()
    
    rfe = RFE(logreg, 10) # not sure this will work for 46 features
    rfe = rfe.fit(X, y.values.ravel())
    output=pd.DataFrame(columns = X.columns)
    output.loc[0]=X.columns
    output.loc[1]=rfe.support_
    output.loc[2]=rfe.ranking_
    output.to_csv('data/ranks.csv', index=False)
    print(output)
    print(X.columns)
    print(rfe.support_)
    print(rfe.ranking_)
    

recursiveFeatureElimination(X,y)







# use logistic regression to analyze the data
# from sklearn.linear_model import LogisticRegression
# from  sklearn.model_selection import train_test_split







