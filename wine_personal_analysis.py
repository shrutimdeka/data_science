# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:41:29 2019

@author: Shruti
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
wine = pd.read_csv("C:/Users/Shruti/Downloads/winequality-white.csv", sep=';')
wine.head()
wine.shape #(4898,12)

for col in wine.columns:
    x = type(col[0]) 
    print(col, x, "-" ,wine[col].dtypes) #see types of each feature
    
################################# EDA ######################################## To visualize in tableau, must save in proper format of tables
#summary statistics of each feature
for col in wine.columns:
    print(wine[col].describe(), "/n")

np.sort(pd.unique(wine['quality'])) #see the target variable categories (3-9)

wine.isna().any() #not a number(invalid)
wine.isnull().any() #(empty)

#histogram
wine.hist()
#quality classes
sns.countplot(x='quality', data=wine) #Very unbalanced classes

#qqplot for normality of data
import pylab
from matplotlib import pyplot
import scipy.stats as stat
columns = wine.columns
print(columns)

stat.probplot(np.log(wine.iloc[:, 0]), dist='norm', fit=True, plot=pylab) # fixed acidity- outliers
stat.probplot(np.log(wine.iloc[:, 1]), dist='norm', fit=True, plot=pylab) #volatile acidity

stat.probplot(np.log(wine.iloc[:, 2]), dist='norm', fit=True, plot=pylab) # citric acid

stat.probplot(np.log(wine.iloc[:, 3]), dist='norm', fit=True, plot=pylab) # residual sugar

stat.probplot(np.log(wine.iloc[:, 4]), dist='norm', fit=True, plot=pylab) # chlorides

stat.probplot(np.log(wine.iloc[:, 5]), dist='norm', fit=True, plot=pylab) # free sulfur dioxide-one outlier

stat.probplot(np.log(wine.iloc[:, 6]), dist='norm', fit=True, plot=pylab) # total sulfur dioxide- low outliers

stat.probplot(np.log(wine.iloc[:, 7]), dist='norm', fit=True, plot=pylab) # density-high outliers
stat.probplot(np.log(wine.iloc[:, 8]), dist='norm', fit=True, plot=pylab) # pH
stat.probplot(np.log(wine.iloc[:, 9]), dist='norm', fit=True, plot=pylab) # sulphates

stat.probplot((wine.iloc[:, 10]), dist='norm', fit=True, plot=pylab) # alcohol

#histogram to view the distribution
pyplot.hist((wine.iloc[:, 10])) #right skewed

#see boxplot to find outlier
pyplot.boxplot(np.sort(wine.iloc[10])) #one high outlier

#histogram to view the distribution-density
pyplot.hist((wine.iloc[:, 7]))
#see boxplot to find outlier-density
pyplot.boxplot(np.sort(wine.iloc[7])) #two outliers

#histogram to view the distribution-total sulpher dioxide
pyplot.hist(np.log(wine.iloc[:, 6])) 
#see boxplot to find outlier-total sulpher dioxide
pyplot.boxplot(np.sort(wine.iloc[6])) #2 outliers

#histogram to view the distribution-free sulpher dioxide
pyplot.hist(np.log(wine.iloc[:, 5])) 
#see boxplot to find outlier- free sulpher dioxide
pyplot.boxplot(np.sort(wine.iloc[5])) #2 outliers

#histogram to view the distribution-chloride
pyplot.hist(np.log(wine.iloc[:, 4])) 
#see boxplot to find outlier- chloride
pyplot.boxplot(np.sort(wine.iloc[4])) #2 outliers

#histogram to view the distribution-residual sugar
pyplot.hist(np.log(wine.iloc[:, 3])) 
#see boxplot to find outlier- residual sugar
pyplot.boxplot(np.sort(wine.iloc[3])) #high outliers

#histogram to view the distribution-citric acid
pyplot.hist(np.log(wine.iloc[:, 2])) 
#see boxplot to find outlier- citric acid
pyplot.boxplot(np.sort(wine.iloc[2])) #lots of outliers

#histogram to view the distribution-
pyplot.hist(np.log(wine.iloc[:, 0])) #left skewed slightly
#see boxplot to find outlier- free sulpher dioxide
pyplot.boxplot(np.sort(wine.iloc[0])) #2 outliers

#pairplot
sns.pairplot(wine) #no straightforward linear relationship - Density and sugar has some kind of positive correlation (check if strong enough to be autocorrelation)
#and ofcourse- the data needs to be normalized

############################ DATA Engineering ################################
#remove outliers- z score (too far from zero mean, mark as outlier)
#from scipy import stats
#z = np.abs(stats.zscore(wine))
#z #each data point in eah feature (TOO Complicated to get back the desired data )

#remove outliers - use IQR score of boxplot
Q1 = wine.quantile(0.25)
Q3 = wine.quantile(0.75)
IQR = Q3 - Q1 #for each feature
cutoff = 1.5 * IQR

#separate out outliers of each feature
outliers = []

for c in columns:
    outliers.append([x for x in wine[c] if x < (Q1 - cutoff)[c] or x > (Q3 +  cutoff)[c]]) #outliers in each column
    
len(outliers[11]) #12 features, different number of outliers in each inner list
#119 + 186 + 7 + 212 + 50 + 19 + 5 + 75 + 124 + 0 + 200 = 997 outliers total

#separate out valid data
new_wine = wine[~((wine < (Q1 - 1.5 * IQR)) |(wine > (Q3 + 1.5 * IQR))).any(axis=1)]
new_wine.shape

#----------------------------------check how removing outliers affect the distributions-----------
#boxplot of new data
pyplot.boxplot(np.log(new_wine.iloc[8])) #NNO extreme outliers
#qqplot
stat.probplot(np.log(new_wine.iloc[9]), dist='norm', fit=True, plot=pylab) #almost normal


sns.countplot(data=new_wine, x='quality') #so non-outlier data has lost the 3 smallest classes of quality 
#We need to solve imbalanced data and THEN tackle outliers
#Start from the original data again
#------------------------------------------------------------------------------------------------

#Solve unbalanced data issue (see countplot of quality)-use wine
sns.countplot(data= wine, x='quality')

#count number of observation for the smallest class (quality=3)
wine[wine['quality'] == 3].shape[0] #20 observations only
#the proportion is...
wine[wine['quality'] == 3].shape[0]/wine.shape[0] #0.4 % ONLY
#same goes for the other two smallest classes (8 and 9)

#WE CAN NOT UNDERSAMPLE - only 20 observation of every class is not enough to build an efficient model
#OVERSAMPLING - We will need ALOT of synthesized samples from mere 20 observations- MISLEADING data
#We can use a MIX of these two techniques-ensemble
#OR use SMOTE (slow) for oversampling
#OR CLASS WEIGHTS

#try straightforward smote
import sys
sys.path
sys.path.append("C:/Users/Shruti/AppData/Local/Programs/Python/Python37-32/Lib/site-packages")

#from imblearn.over_sampling import SMOTE #tensor backend
#separate X and Y
X = wine
X = X.drop(['quality'], axis=1)
X.head()
Y = wine.quality
Y.tail(10)

#Convert Y from float to string as so...
for i in range(3, 10):
    Y[Y ==i] = "quality_"+str(i)

#train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=27)

#keep synthasized data only in training set so that the test data are real and realistic evaluation can happen

#Oversampling
from sklearn.utils import resample

#concatenate training data back as one!
train_wine = pd.concat([X_train, Y_train], axis=1)
#separate small and large classes
indice_3 = train_wine[train_wine['quality'] != 3].index
small_class = train_wine
small_class.drop(indice_3, inplace=True) #keep only class 3

train_wine = pd.concat([X_train, Y_train], axis=1)
indice = train_wine[train_wine['quality'] != 6].index
large_class =  train_wine
large_class.drop(indice, inplace=True) #only 6 included as largest

#upsample small_class
small_3_upsampled = resample(small_class,
                          replace=True, # sample with replacement
                          n_samples=len(large_class), # match number in majority class
                          random_state=27) # reproducible results

#combine small_upsampled and large_class to form training set
#upsample_train = pd.concat([large_class, small_3_upsampled])

#check if balanced classes
#sns.countplot(data=upsample_train, x='quality')
#upsample_train.quality.value_counts()

#repeat for each small class-4
train_wine = pd.concat([X_train, Y_train], axis=1)
indice_4 = train_wine[train_wine['quality'] != 4].index
small_class = train_wine
small_class.drop(indice_4, inplace=True) #keep only class 4

large_class
#upsample small_class
small_4_upsampled = resample(small_class,
                          replace=True, # sample with replacement
                          n_samples=len(large_class), # match number in majority class
                          random_state=27) # reproducible results
#combine small_upsampled and large_class to form training set
#upsample_train = pd.concat([large_class, small_3_upsampled, small_4_upsampled])
#check classes
#sns.countplot(x='quality', data=upsample_train) #works

#repeat for 5, 7, 8, 9
train_wine = pd.concat([X_train, Y_train], axis=1)
indice_5 = train_wine[train_wine['quality'] != 5].index
indice_7 = train_wine[train_wine['quality'] != 7].index
indice_8 = train_wine[train_wine['quality'] != 8].index
indice_9 = train_wine[train_wine['quality'] != 9].index

#repeat the entire process of upsampling-----------------------
train_wine = pd.concat([X_train, Y_train], axis=1)
small_class = train_wine
small_class.drop(indice_5, inplace=True) #keep only class 5
large_class
#upsample small_class
small_5_upsampled = resample(small_class,
                          replace=True, # sample with replacement
                          n_samples=len(large_class), # match number in majority class
                          random_state=27) # reproducible results
#-----------------------------------------------------------------
train_wine = pd.concat([X_train, Y_train], axis=1)
small_class = train_wine
small_class.drop(indice_7, inplace=True) #keep only class 7
large_class
#upsample small_class
small_7_upsampled = resample(small_class,
                          replace=True, # sample with replacement
                          n_samples=len(large_class), # match number in majority class
                          random_state=27) # reproducible results
#------------------------------------------------------------
train_wine = pd.concat([X_train, Y_train], axis=1)
small_class = train_wine
small_class.drop(indice_8, inplace=True) #keep only class 8
large_class
#upsample small_class
small_8_upsampled = resample(small_class,
                          replace=True, # sample with replacement
                          n_samples=len(large_class), # match number in majority class
                          random_state=27) # reproducible results

#---------------------------------------------------------------------
train_wine = pd.concat([X_train, Y_train], axis=1)
small_class = train_wine
small_class.drop(indice_9, inplace=True) #keep only class 9
large_class
#upsample small_class
small_9_upsampled = resample(small_class,
                          replace=True, # sample with replacement
                          n_samples=len(large_class), # match number in majority class
                          random_state=27) # reproducible results
#---------------------------------------------------------------------------
#combine small_upsampled and large_class to form training set
upsample_train = pd.concat([large_class, small_3_upsampled, small_4_upsampled, small_5_upsampled, small_7_upsampled,
                            small_8_upsampled, small_9_upsampled])
#check classes
sns.countplot(x='quality', data=upsample_train) #balanced training data!
#--------------------------------------------------------------------------

#is normality very imp in used algorithms below?
pyplot.hist(np.log(upsample_train.iloc[:, 0]))
pyplot.hist(np.log(upsample_train.iloc[:, 1]))
pyplot.hist(upsample_train.iloc[:, 2])
pyplot.hist(np.log(upsample_train.iloc[:, 3])) #very skewedto not normal transformation
pyplot.hist(np.log(upsample_train.iloc[:, 4]))#some outliers
pyplot.hist(np.log(upsample_train.iloc[:, 5])) 
#outliers can not be due to small classes anymore
#as classes have been balanced, so remove them
pyplot.hist(upsample_train.iloc[:, 6]) #high outliers
pyplot.hist(upsample_train.iloc[:, 7]) #discrete-like values (CHECK it)
pyplot.hist(np.log(upsample_train.iloc[:, 8])) #beautifully symmetric
pyplot.hist(np.log(upsample_train.iloc[:, 9])) #slight skewed
pyplot.hist(np.log(upsample_train.iloc[:, 10])) #not normal at all

###############################################################################
#Normality of upsampled data has not been addressed anywhere
#Transform above features to become normal?

######################################## Model Building & Training ############
upsample_x_train = upsample_train.drop(['quality'], axis=1) 
upsample_y_train = upsample_train.quality

#logistic regression model- assuming features are normal
#using upsampled training data/after outlier analysis on balanced data
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
lor = lor.fit(upsample_x_train, upsample_y_train)
pred = lor.predict(X_test)
correct_pred = [x for x in pred==Y_test if x is True]        #on UPsampling
accuracy = len(correct_pred)/len(Y_test) #31%
#look into what happened- normality violation/synthesized data not mimicking real ones

#--------------------------------------------------------------------------
#random forest-doesn't need normalized/scaled data (not distance based)
#with wine data
X= wine.drop(['quality'], axis=1)
Y = wine.quality
#Y[Y=='quality_9'] = 9

#create train and test from original wine data
row_train = int(X.shape[0]*(3/4))
x_train = X.iloc[0:row_train, :]
x_test = X.iloc[row_train: , :]
y_train = Y[0:row_train]
y_test = Y[row_train: ]

from sklearn.ensemble import RandomForestClassifier
rf_2 = RandomForestClassifier(n_estimators = 1000, random_state = 42, class_weight="balanced")
rf_2.fit(x_train, list(y_train.values))
pred = rf_2.predict(x_test)
correct_pred = [x for x in pred == y_test if x is True]  #on wine-original (BALANCED class weight)
accuracy = len(correct_pred)/len(y_test) #55% - not much improved
#So wine data with unbalanced data is not worth much

#with Upsampled data
rf_3 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf_3.fit(upsample_x_train, upsample_y_train)
pred = rf_3.predict(X_test)
correct_pred = [x for x in pred == Y_test if x is True]  #on wine-original (BALANCED class weight)
accuracy = len(correct_pred)/len(Y_test) #68% - not much improved

#Solution- adaboost
from sklearn.ensemble import AdaBoostClassifier

rf_3 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf_3.fit(upsample_x_train, upsample_y_train)
predictions = rf_3.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, predictions)
correct_pred = [x for x in predictions == Y_test if x is True]
acc = len(correct_pred)/len(Y_test) #68% - same as random forest on upsampled data
#--------------------------------------------------------------------------
#neural net
#with wine data/upsampled data
from keras.models import Sequential ##64 bit python for tensorflow module
from keras.layers import Dense
