import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, ElasticNetCV
from scipy.stats import skew
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import sys


#load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print(train['SalePrice'].describe())
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
#prices.hist()
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


#Get the correlation matrix
X_train = train.drop("SalePrice", axis=1)
corrmat = X_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show
plt.close()



#1. check for missing values:  there are totally 19 features contain null values.  May need to do something with it.  poolQC(1453), fence(1179)  and MiscFeature(1406)
#todo: we may need to do something about the missing values.  For poolQC and miscFeature, maybe remove these features.
missing = train.isnull().sum()
missing = missing[missing > 0]
print("# of features with missing values:" + str(missing.shape[0]))
missing.sort_values(inplace=True)
print(missing)

#Remove poolQC feature: too many missing values. It lowered the final score. 0.161 is the best with QC in the mix.
train = train.drop("PoolQC", axis = 1)
test = test.drop("PoolQC", axis = 1)

#drop bsmthalfbath
X_train = X_train.drop('BsmtHalfBath', axis = 1)
test = test.drop('BsmtHalfBath', axis = 1 )


#filling NA's with the mean of the column:
test['SalePrice']=0
all_Data = train.append(test)
all_Data = all_Data.fillna(all_data.mean())


train = train.fillna(train.mean())
test = test.fillna(test.mean())
print("training set shape :" + str(train.shape))
print("testing set shape :" + str(test.shape))

#log seems to be causing much worse results.
if  len(sys.argv)>1:
    if sys.argv[1] == 'log':
        train['SalePrice'] =  np.log1p(train['SalePrice'])
        #log transform skewed numeric features:
        numeric_feats = train.dtypes[train.dtypes != "object"].index

        skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index

        train[skewed_feats] = np.log1p(train[skewed_feats])
        test[skewed_feats] = np.log1p(test[skewed_feats])


#Find all qualitatives and quanitatives
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
#print('qualitative: '  + str(len(qualitative)) + str(qualitative))
#print('quantitative: ' + str(len(quantitative)) + str(quantitative))
#Qualitative: 43, ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature','SaleType', 'SaleCondition']
#Quantitative: 36, ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']


#2. looking for outliers

#Remove the outliers. 
#this should be adjusted based on the result.  Not always adjustable.
train = train[train.GrLivArea < 4000] #removed 4 outliers, 523, 691, 1182, 1298
train = train[train.LotFrontage < 300] #934, 1298
 #train = train[train.LotArea < 200000] # removed 313
# train = train[train.TotalBsmtSF < 6000] #1298
# train = train[train["1stFlrSF"] < 4000] #1298  maybe back here later to see if 3000 should be the gate.

plt.scatter(train.LotArea, train.SalePrice, c = "blue", marker = "s")
plt.title("Looking for outliers -- LotArea")
plt.xlabel("LotArea")
plt.ylabel("SalePrice")
plt.show()
plt.close


print("List of Outliers: " + str(train[train.LotArea >200000].LotArea))


#List all unique values of a column:
print("MSZoning values:" + str(train.MSZoning.unique()))
print("LotShape values:" + str(train.LotShape.unique()))
print("LotUtilities values:" + str(train.LotShape.unique()))
print("HouseStyle values:" + str(train.HouseStyle.unique()))
print("RoofStyle values:" + str(train.RoofStyle.unique()))
print("ExterQual values:" + str(train.ExterQual.unique()))
print("Foundation values:" + str(train.Foundation.unique()))
print("BsmtQual values:" + str(train.BsmtQual.unique()))
print("Heating values:" + str(train.Heating.unique()))
print("CentralAir values:" + str(train.CentralAir.unique()))
print("Electrical values:" + str(train.Electrical.unique()))
print("KitchenQual values:" + str(train.KitchenQual.unique()))
print("Functional values:" + str(train.Functional.unique()))
print("SaleType values:" + str(train.SaleType.unique()))
print("SaleCondition values:" + str(train.SaleCondition.unique()))
print("Alley values:" + str(train.Alley.unique()))
print("Alley values:" + str(train.Alley.value_counts()))

#Check the relationship between the categorical value and the sales price
#this graph shows that overall quality affects the sales price along with OverallCond, CentralAir, Functional, etc.
#var = 'OverallQual'
#data = pd.concat([train['SalePrice'], train[var]], axis=1)
#f, ax = plt.subplots(figsize=(8, 6))
#fig = sns.boxplot(x=var, y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=800000)
#plt.show()
#plt.close



#3. Convert categorical data to numeric:
test['SalePrice'] = np.zeros(len(test.index))
all_data = pd.concat((train, test))
all_data = pd.get_dummies(all_data)

train = all_data[:train.shape[0]]
test = all_data[train.shape[0]:]
test = test.drop("SalePrice", axis = 1)


#train = pd.get_dummies(train)
#test = pd.get_dummies(test)
print("new shape of data: " + str(train.shape))
print("new shape of data: " + str(test.shape))


#...

#4. update the dataset with the updated training set
train.to_csv('data/train_update.csv', index=False)
test.to_csv('data/test_update.csv', index=False)
