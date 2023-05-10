# **DATASET PRE-PROCESSING**

import numpy as np 
import pandas as pd 

# **DATA VISUALIZATION**

import seaborn as sns 

# **DATA PREPROCESSING**

from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

# **MACHINE LEARNING MODELS**

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# **METRICS**

from sklearn.metrics import classification_report

# **INPUT**

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")
data.drop(["id"] , axis = 1 , inplace = True)

data["Gender"] = np.where(data["Gender"] == "Male" , 1 , 0)
data["Vehicle_Damage"] = np.where(data["Vehicle_Damage"] == "Yes" , 1 , 0)

data["Annual_Premium"] = np.where(data["Annual_Premium"] >80000 , 0 , data["Annual_Premium"])

data = pd.concat([pd.get_dummies(data["Vehicle_Age"]) , 
                 data.drop("Vehicle_Age" , axis = 1)] , 
                axis = 1 , join = "inner")

train , test = np.split(data.sample(frac = 1) , [int(0.8 * len(data))])

def pre(dataframe):
    x = dataframe.drop("Response" , axis = 1)
    y = dataframe["Response"]
    
    x = pd.DataFrame(PCA(n_components = 2).fit_transform(x))
    
    return x , y

X_train , y_train = pre(train)
X_test , y_test = pre(test)

KNN = KNeighborsClassifier()
KNN.fit(X_train , y_train)

DTC = DecisionTreeClassifier()
DTC.fit(X_train , y_train)

RFC = RandomForestClassifier()
RFC.fit(X_train , y_train)

GNB = GaussianNB()
GNB.fit(X_train , y_train)

print(KNN , classification_report(y_test , KNN.predict(X_test)))
print(DTC , classification_report(y_test , DTC.predict(X_test)))
print(RFC , classification_report(y_test , RFC.predict(X_test)))
print(GNB , classification_report(y_test , GNB.predict(X_test)))
