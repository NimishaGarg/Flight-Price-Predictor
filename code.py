import numpy as np
import pandas as pd
import sklearn
import random
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def linearregression():
    lm = LinearRegression()
    lm.fit(train_X, train_Y)
    y_pred = lm.predict(test_X)
    score1 = r2_score(test_Y, y_pred)
    print("LinearRegression R^2 score = ", score1)
    return lm

def RandomForest():
    regr = RandomForestRegressor(max_depth = 3,random_state = 10, n_estimators = 100)
    regr.fit(train_X,train_Y.flatten())
    # print(regr.feature_importances_)
    y_pred1 = regr.predict(test_X)
    score2 = regr.score(test_X, test_Y.flatten())
    print("RandomForestRegressor R^2 score = ",score2)
    return regr

data = pd.read_csv('/home/nimisha/sem5/ml_lab/test2/data.csv')
nparr = data.values
nparr = nparr[1:,:]

######################################################################################
train, test = train_test_split(nparr, random_state = 10)

train_X = train[:,:8]
train_Y = train[:,8:]

test_X = test[:,:8]
test_Y = test[:,8:]
np.savetxt("Test_X.csv", test_X, delimiter = ",")
np.savetxt("Test_Y.csv", test_Y, delimiter = ",")

print("\n-------Before Data Augmentation----------")
lm = linearregression()
regr = RandomForest()

##################### DATA AUGMENTATION #############################################

k=0
random.seed(100)
while k < 400 :
    ind = random.randint(0,1359)
    stop = train_X[ind][4]
    if stop == 0:
        train_data = train_X[ind]
        train_data[7] = random.randint(1,7)
        train_Y=np.vstack((train_Y, train_Y[ind]))
        train_X = np.vstack((train_X, train_data))
        k+=1
# print(train_X.shape)
print("\n-------After Data Augmentation----------")
lm = linearregression()
regr = RandomForest()
fname = "RandomForestRegressor.pkl"
model_pkl = open(fname,'wb')
pickle.dump(regr,model_pkl)
model_pkl.close()

fname = "LinearRegression.pkl"
model_pkl = open(fname,'wb')
pickle.dump(lm,model_pkl)
model_pkl.close()
#####################################################################################
