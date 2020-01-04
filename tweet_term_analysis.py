# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:23:43 2019

@author: Shruti
"""

import string
import tweepy  #move entire folder to Ananconda3/Lib/site-packages
help(tweepy)
import nltk
import sys
print(sys.path)
sys.path.append("C:/Users/Shruti/AppData/Local/Programs/Python/Python37-32/Lib/site-packages")

from wordcloud import WordCloud 
from bs4 import BeautifulSoup as bs
#Twitter API credentials
consumer_key = "VX6UxbFfJiDa9P9xBZ3uaNIV5"
consumer_secret = "UTi4c720q6HzOTjyp6k2CsaVUNzO6CHuFjLPUzrQ36yIExRYSF"
access_key = "1183235731453403136-uSrpxEv7iAkuryoxYVXaWnL2NU7cEL"
access_secret = "VW71fMl1RQFV5Esgsz1IWpZ6NptT7uH6wD91wDMcvJB9R"

#function to read all tweets of a given user
def read_tweets(search_term):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    	
    # Collect tweets
    tweets = tweepy.Cursor(api.search,
              q=search_term,
              lang="en",  #can add, since = "2018-11-16"
              ).items(1000)  #1000 most recent tweets
    l= " "
    # Iterate and print tweets
    for tweet in tweets:
        #print(tweet.text)
        l = l + tweet.text
    #to csv
    with open(search_term+"_tweets.txt", "w", encoding="utf8") as obj:
        obj.write(str(l))
    return tweets

t= read_tweets("Modi")

#read from the txt
twit=""
with open("Modi_tweets.txt", "r", encoding="utf8") as obj:
    twit = obj.read()
len(twit) #1 lakhs

#clean data
import re
import pandas as pd
whitelist = set('abcdefghijklmnopqrstuvwxyz ')  #the gap matters
twit_string = ''.join(filter(whitelist.__contains__, twit.lower()))
clean_tweets = re.sub("[^A-Za-z" "]+"," ", twit_string).lower()
#must remove 'https?', bjp/s, pm/s, narendra, specialized words used in the context needs to be added to list
clean_tweets = re.sub("modi?","", str(clean_tweets)) #hindutva, bhakt, hinduvta, guru, mahan,- add
clean_tweets = re.sub("http?","", str(clean_tweets)) #ok
#tokenize the words
tokens = clean_tweets.split(" ")  #18k words
token_no = len(tokens)-1

#redundant words
words= []
for i in range(0, token_no) :
    if len(tokens[i]) > 3 :
        words.append(tokens[i])
len(words) #12k

#due to bad extraction/ partial tweets, meaningful words have merged forming a meaningless word. extract them
#stopwords
with open("C:/Users/Shruti/Downloads/Excelr-assignment/Ass11-Text_Mining/stop.txt", 'r') as file:
    stopwords = file.read()
words_stop = [w for w in words if not w in stopwords]
len(words_stop) #18k

#remove obvious words like narendra, bjp..
l = ['bjp', 'bjps', 'narendra', 'india', 'modi', 'modis', 'indias', 'indian', 'indians', 'raydalio', 'taseer', 'pokershash', 'aatish', 'bjpindia', 'bjpindian', 'govt']

for k in words_stop:
    if k in l:
        words_stop.remove(k)

import matplotlib.pyplot as plt

#import wordcloud
from wordcloud import WordCloud
# Joinining all the reviews into single paragraph 
string_modi = " ".join(words_stop) #tokens to single string
fig = plt.figure(figsize = (50, 50))
wordcloud_ip = WordCloud(
                      background_color='white',
                      width=4000, height=500).generate(string_modi)

plt.imshow(wordcloud_ip) 

#add positive words
with open("C:/Users/Shruti/Downloads/Excelr-assignment/Ass11-Text_Mining/positive-words.txt", 'r') as obj:
    pos= obj.read().split("\n")
pos_review = [word for word in words_stop if word in pos ] #take tokenized version, not string
# Joinining all the reviews into single paragraph 
string_modi = " ".join(pos_review) #tokens to single string
fig = plt.figure(figsize = (50, 50))
wordcloud_ip = WordCloud(
                      background_color='white',
                      width=4000, height=500).generate(string_modi)

plt.imshow(wordcloud_ip) 

#negative
with open("C:/Users/Shruti/Downloads/Excelr-assignment/Ass11-Text_Mining/negative-words.txt", 'r') as obj:
    neg= obj.read().split("\n")
neg.append('bhakts')
neg.append('hindutva')
neg_review = [word for word in words_stop if word in neg ] #take tokenized version, not string

# Joinining all the reviews into single paragraph 
string_modi = " ".join(neg_review) #tokens to single string
fig = plt.figure(figsize = (50, 50))
wordcloud_ip = WordCloud(
                      background_color='white',
                      width=4000, height=500).generate(string_modi)

plt.imshow(wordcloud_ip) 
