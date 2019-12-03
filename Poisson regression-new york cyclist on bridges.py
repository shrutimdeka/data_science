#!/usr/bin/env python
# coding: utf-8

# In[38]:

import pandas as pd
df_data = pd.read_csv("C:/Users/Shruti/Downloads/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
df_data = df_data.drop(['Unnamed: 0', 'Day'], axis=1) #since date and day are the same, no time specified
df_data.columns = ['Date', 'High_Temp(F)', 'Low_Temp(F)', 'Precipitation', 'Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro', 'Total']
df_data.head(10)
df_data.shape

# In[39]:


#Problem Statement : find relationship between the number of bicyclists who cross Brooklyn bridge in New York
#Number of crossings of different bridges each day with weather conditions and total crossings

#Data Exploration
val_list = [df_data['Brooklyn'].mean()]
val_list.extend([df_data['Manhattan'].mean()]) 
val_list.extend([df_data['Williamsburg'].mean()])
val_list.extend([df_data['Queensboro'].mean()])
table = pd.DataFrame(columns = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro', 'measure'])
table.loc[0, 'Brooklyn'] = val_list[0]
table.loc[0, 'Manhattan'] = val_list[1]
table.loc[0, 'Williamsburg'] = val_list[2]
table.loc[0, 'Queensboro'] = val_list[3]
table.loc[0, 'measure'] = 'average'
table #shows average no of cyclist going across a bridge
table.loc[1, 'Brooklyn'] = df_data['Brooklyn'].min()
table.loc[1, 'Manhattan'] = df_data['Manhattan'].min()
table.loc[1, 'Williamsburg'] = df_data['Williamsburg'].min()
table.loc[1, 'Queensboro'] = df_data['Queensboro'].min()
table.loc[1, 'measure'] = 'min'
table #shows minimum number of cyclists
table.loc[2, 'Brooklyn'] = df_data['Brooklyn'].max()
table.loc[2, 'Manhattan'] = df_data['Manhattan'].max()
table.loc[2, 'Williamsburg'] = df_data['Williamsburg'].max()
table.loc[2, 'Queensboro'] = df_data['Queensboro'].max()
table.loc[2, 'measure'] = 'max'
table #shows max cyclists across a bridge


# In[40]:


df_data.isna().mean() #Check if any empty cell exists for each feature -> none exists
optimum_weather = df_data[df_data['Total'] == df_data['Total'].max()] #days when most cyclists were active
worst_weather = df_data[df_data['Total'] == df_data['Total'].min()] #days when most cyclists were LEAST active
optimum_weather.iloc[:, 0:4] #Notice if weather has some similarities - yes, all features are identical
worst_weather.iloc[:, 0:4] #Notice all features are identical and opposite to optimum weather slice


# In[41]:


df_data['High_Temp(F)'].min()
df_data['Low_Temp(F)'].min()
pd.unique(df_data['Precipitation']) #what is T precipitation?
#Conclusion: least active cyclist doesn't correspond to minimum high temp or low temp
#IT DEPENDS ON Precipitation -> Best weather has 0 precipitation and Worst has snow that day
#See tableau visualization to confirm


# In[51]:


#Data Engineering (prepare data for processing)
#Assumption of poisson regression: mean  = variance, or very small difference
#Otherwise neg binomial regression for large difference

#Precipitation has some values as x(S), T -> most probably signifies snow and ? (can only have non neg integers as feature values)

#prep for clustering
prec = pd.to_numeric((df_data['Precipitation']).astype(str).str.replace('0.47 (S)','0.47'), errors='coerce')  #all strings are nan= 0.47(S) and T
prec
#label prec for clustering
import numpy as np
np.sort(pd.unique(prec))

# In[48]:


#z score to find outlier (can't as string type precipitation values)
#consider string rows as test data
#as no obvious linear relationship exists, we will try clustering as an initial exploration


# In[98]:

#Clusters to find what T could be replaced with- take mean/ centroid precipitation of similar points (but data too small)
cluster_data = df_data
cluster_data = cluster_data.drop(['Precipitation'], axis=1 )
cluster_data['Precip'] = prec
cluster_data.isna().sum()

#Dropping nan values
X = pd.DataFrame(cluster_data.dropna()) #training/ valid values
X['Precip'] = pd.to_numeric((X['Precip'])) #string type to numeric
(df_data.shape[0])- X.shape[0] #14 rows of nan values, 7= (S), rest 7= T

#find if temperatures are  correlated to precipitation
import seaborn as sns
sns.clustermap(X.iloc[:, 1:]) #no date considered
X.columns
np.cov(X['High_Temp(F)'], X['Precip'])
plt.scatter(X['High_Temp(F)'], X['Precip'])
#NO relationship exists
# In[103]:
#target variable must be labeled data

X.loc[X['Precip'].isin([0.00, 0.01, 0.05, 0.09]), 'label'] = 'low'
X.loc[X['Precip'].isin([0.15]), 'label'] = 'average'
X.loc[X['Precip'].isin([0.16, 0.2 , 0.24]), 'label'] = 'high'
X[X['Precip'] > 0.15]
#X.label.isna().sum() #all are labeled data now

#df_data[df_data['Precipitation'] == 'T'].count()
from sklearn.neighbors import KNeighborsClassifier as knn
for i in range(2, 20, 2):
    model = knn(n_neighbors=i).fit(X.iloc[0:100, 1:7], X.iloc[0:100, 9])
    test_acc = np.mean(model.predict(X.iloc[100:200, 1:7]) == X.iloc[100:200, 9])
    print(test_acc) #2-10 neighbors are fine
#choose 12 as there's a bend at that point
model = knn(n_neighbors=12).fit(X.iloc[:, 1:7], X.iloc[:, 9])
# In[97]:
test_indice = cluster_data.index[cluster_data['Precip'].isna()] #index for predicting data (nan values that were dropped)
for r in test_indice:
    print(r)
    print(model.predict(cluster_data.iloc[r, 1:7 ].values.reshape(1, 6)))
#So all are low
#Looks like to predict precipitation the features are not enough. Temp for both precipitation types are similar and ambiguous
#Clustering is NOT the way to find value of T
#--------------------------------------------------------------------------------------------------------------
#Decision: Since precipitation is a major decider for number of cyclist in this data
#we will not try to 'guess' or impute the data (there's no sure way to do this here)
#We will simply ignore the 'T' Precipitation rows( they fall under average total anyway) 
#and keep 0.47(S) as 0.47


# In[102]:

#Prepare data for POISSON Regression
#df_data.head()
#convert string to float
df_data.loc[df_data['Precipitation'] == '0.47 (S)', 'Precipitation'] = '0.47' #Remove (S) from 0.47 (S)
df_data[df_data['Precipitation'] == '0.47']
df_data ['Precipitation']= pd.to_numeric((df_data['Precipitation']).astype(str), errors='coerce') #to numeric, T is converted to nan
df_data = pd.DataFrame(df_data.dropna())
df_data.shape

#convert date to seperate formats
from datetime import datetime
df_data.shape
df_data.tail(10) #repeated data unfortunately
#---------------------------------------------------------will not include date---------------
#Extract date
for i in range(0, df_data.shape[0]):
    df_data.loc[i, 'date'] = datetime.strptime(df_data.loc[i, 'Date'], "%Y-%m-%d %H:%M:%S").day #2016 format needs %Y, otherwise %y

#Extract month
for i in range(0, df_data.shape[0]):
    df_data.loc[i, 'month'] = datetime.strptime(df_data.loc[i, 'Date'], "%Y-%m-%d %H:%M:%S").month #2016 format needs %Y, otherwise %y

#Extract year
for i in range(0, df_data.shape[0]):
    df_data.loc[i, 'year'] = datetime.strptime(df_data.loc[i, 'Date'], "%Y-%m-%d %H:%M:%S").year #2016 format needs %Y, otherwise %y

#------------------------------------------------------------------------------------------------
# In[ ]:
from patsy import dmatrices
import statsmodels.api as sm
df_data.shape
#train and test data
train = df_data.iloc[0:150, :]
test = df_data.iloc[150: , :]
train.shape #176
test.shape #56
train.columns = ['date', 'high_temp', 'low_temp' ,'precipitation', 'brooklyn', 'manhattan', 'williamsburg', 'queensboro', 'total']
test.columns = ['date', 'high_temp', 'low_temp' ,'precipitation', 'brooklyn', 'manhattan', 'williamsburg', 'queensboro', 'total']
exp = """brooklyn~high_temp+low_temp+precipitation"""
y_train, x_train = dmatrices(exp, train, return_type = 'dataframe') #USE this training data
y_test, x_test = dmatrices(exp, test, return_type = 'dataframe') #USE this test data

#Regression variables : high temp, lowtemp, precipitation
model = sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit() #y variable, x variables
model.summary()

#prediction
predicted = model.get_prediction(x_test).summary_frame() #predictions as dataframe(contains mean, std dev and upper & lower CI for each test data)
#predicted['mean'].to_csv("predicted_test_brooklyn.csv")
#y_test.to_csv("actuals_brooklyn.csv")
#see actual vs predicted graph in tableau

#Analysis of result!!

#1) P value < 0.05 (95% confidence) for ALL regressors.
#So all variables contribute significantly to the number of cyclist on brooklyn bridge

#2)Coeficients: negative means a negative correlation exists between regressor and dependent count variable
#Small Low temp and low precipitation will see high cyclist number on the bridge and reverse or high temp

#3)Intercept: Absorbs bias in the model, because all regressors bing simultaneously constant is not realistic!
#but setting it zero will distort the model and create bias.

#4)Degree of effect of regressor: Precipitation has highest effect
#Per unit change in precipitation will cause almost 3X change in cyclist count, Negatively

#5)Log likelihood can be used to compare different sample sizes and determine best fit
#Higher the better

#6)Lower and upper bounds: closer the more better model
#large intervals are not very useful

#7) z score:  indication as to how much uncertainty "surrounds" the regressor coefficient
#higher a score, more certain we are about the coefficient. 
#Standard error gets smaller with larger z score

#8)std err: distance of observed value from regression line, like r2 but for each regressor