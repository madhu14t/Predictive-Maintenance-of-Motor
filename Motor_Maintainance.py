import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

df = pd.read_csv("feeds.csv")

#Removing Null values

update_df = df.drop([199, 200, 201,202])

x = update_df.iloc[:,[0,1,2,3,4]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++', random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


kmeans = KMeans(n_clusters = 2,init = 'k-means++', random_state = 0)

y = kmeans.fit_predict(x)

from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")


classifier.fit(x,y)
ypred= classifier.predict(x)
accuracy_score(y, ypred)


# In[276]:


# input
#0 5.8	232	59.22	68	0
#1 5.1	229	65.23	68	0

#current = 5.8
#voltage = 232
#temperature = 59.22
#humidity = 68
#vibration = 0

current = 5.1
voltage = 229
temperature = 65.23
humidity = 68
vibration = 0


test = [[current, voltage, temperature, humidity, vibration]]

def give_pred(test):
    prediction = classifier.predict(test)

    if prediction == 0:
        return ('Your System Failed')
    else:
        return ('Your System Works')
    