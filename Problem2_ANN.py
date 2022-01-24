# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:50:56 2021

@author: Yaren
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization

from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv("Churn-Modelling.csv")
dataset.info()

x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

print(x)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column

le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size = 0.25,
                                                random_state = 1000)
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = Sequential()
model.add(Dense(units=10, input_dim = xtrain.shape[1], activation="relu"))

model.add(Dense(units=10, activation="relu"))

model.add(Dense(units=10, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(units=10, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

model.fit(xtrain, ytrain, batch_size=16, epochs=100)

ypred = model.predict(xtest)
ypred = (ypred > 0.5)
print(np.concatenate((ypred.reshape(len(ypred),1), ytest.reshape(len(ytest),1)),1))

confusionmatrix = confusion_matrix(ytest, ypred)
print("Confusion Matrix:\n", confusionmatrix)
print("Accuracy Score:", accuracy_score(ytest, ypred))
