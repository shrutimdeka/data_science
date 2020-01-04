# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:23:43 2019

@author: Shruti
"""

import string
import tweepy 
help(tweepy)
import nltk
from wordcloud import WordCloud 
from bs4 import BeautifulSoup as bs
#Twitter API credentials
consumer_key = "########"
consumer_secret = "########"
access_key = "#######"
access_secret = "##########"

#function to read all tweets on a topic
def read_tweets(search_term):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)  #keys are confidential
    auth.set_access_token(access_key, access_secret)  #use your own after applying for access 
    api = tweepy.API(auth)  #twitter Api
    	
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
        obj.write(str(l))  #write the tweets into a file
    return tweets

t= read_tweets("Modi") #topic = Modi

#read from the txt file
twit=""
with open("Modi_tweets.txt", "r", encoding="utf8") as obj:
    twit = obj.read()
len(twit) #1 lakhs

#clean data
import re
import pandas as pd
whitelist = set('abcdefghijklmnopqrstuvwxyz ')  #the gap at the end matters
twit_string = ''.join(filter(whitelist.__contains__, twit.lower()))  #only keep the whitelisted characters from tweets and join with a space
clean_tweets = re.sub("[^A-Za-z" "]+"," ", twit_string).lower()  #all characters turned to lower case

#Now...remove unneccesary words and meaningless terms
#must remove 'https?', bjp/s, pm/s, narendra.
#also, specialized words used in the context needs to be added to list
clean_tweets = re.sub("modi?","", str(clean_tweets)) #hindutva, bhakt, hinduvta, guru, mahan,- add to list
clean_tweets = re.sub("http?","", str(clean_tweets)) #cleaned

#tokenize the words
tokens = clean_tweets.split(" ")  #18k words
token_no = len(tokens)-1

#redundant words removal
words= []
for i in range(0, token_no) :
    if len(tokens[i]) > 3 :
        words.append(tokens[i])
len(words) #12k

#due to bad extraction/ partial tweets, meaningful words have merged forming a meaningless word. extract them
#use stopwords
with open("C:/Users/Shruti/Downloads/Excelr-assignment/Ass11-Text_Mining/stop.txt", 'r') as file:
    stopwords = file.read()
words_stop = [w for w in words if not w in stopwords]
len(words_stop) #18k

#remove obvious words(non-sentimental) like narendra, bjp, india..
l = ['bjp', 'bjps', 'narendra', 'india', 'modi', 'modis', 'indias', 'indian', 'indians', 'raydalio', 'taseer', 'pokershash', 'aatish', 'bjpindia', 'bjpindian', 'govt']

for k in words_stop:
    if k in l:
        words_stop.remove(k)

import matplotlib.pyplot as plt

#import wordcloud
# Joining all the reviews into single paragraph 
string_modi = " ".join(words_stop) #tokens to single string
fig = plt.figure(figsize = (50, 50))
wordcloud_ip = WordCloud(
                      background_color='white',
                      width=4000, height=500).generate(string_modi)

plt.imshow(wordcloud_ip) #all wordcloud 

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

plt.imshow(wordcloud_ip) #positive wordcloud

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

plt.imshow(wordcloud_ip) #negative wordcloud
