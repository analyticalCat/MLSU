import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# train = pd.read_csv("data/train_update.csv")
# test = pd.read_csv("data/test_update.csv")

# y_train=train.SalePrice
# X_train=train.drop("SalePrice", axis=1)
# print("train : " + str(X_train.shape)) #1458, 286
# print("test : " + str(test.shape))  #1459, 285


# #Feature selection for RF

# #Since Alley was listed as least relevant feature, try to remove Alley and see.
# newX = pd.concat((X_train, test))
# newX = newX.drop("Alley_Pave", axis = 1 )
# newX = newX.drop("Alley_Grvl", axis = 1 )

# X_train= newX[:train.shape[0]]

print(int(5/10))
