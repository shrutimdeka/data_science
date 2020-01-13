# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:34:35 2019

@author: Shruti
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

movies = pd.read_csv("C:/Users/Shruti/Downloads/Excelr-assignment/Ass17-Association/my_movies.csv")
movies.head()
#mix of two types of data presentation

#-----------------data transformation------------------------------
movie_slice = movies.iloc[:, 0:5] #transforming to dummy style columns

#create list of list representing each transaction in inner lists
movie_list= []
for col in movie_slice.columns:
    movie_list.append([i for i in movie_slice.loc[0:, col]]) 

mov  = pd.Series(movie_list)

# creating a dummy columns for the each item ... Using item name as columns
X = mov.str.join(sep='*').str.get_dummies(sep='*')

#join above dummy style data with rest of the original data columns, merging duplicate column names
movie_data = pd.concat([X, movies.iloc[:, 5:]], axis=0)
movie_data.columns
movie_data.head()

#remove NaN to 0
for col in movie_data.columns:
    movie_data.loc[movie_data[col].isna(), col] = '0'

#Apriori---------------------------------------------algorithm--------------------------
aprior = apriori(movie_data, min_support=0.05, max_len=3,use_colnames = True)
aprior # 70 association rules

#sort based on max support 
rules = aprior.sort_values('support',ascending = False)
lifted_rules = association_rules(aprior, metric="lift", min_threshold=1) #threshold 2 yields no rules
lifted_rules.head()
lifted_rules.sort_values('lift',ascending = False) #rules with lift metric 
