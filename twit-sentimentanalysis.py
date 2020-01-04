# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:31:06 2019

@author: Shruti
"""
import string
import tweepy  #move entire folder to Ananconda3/Lib/site-packages
help(tweepy)
import nltk
import sys
print(sys.path)
sys.path.insert(0, 'C:\\Users\\Shruti\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages') #temporary path

import sys
sys.path.append("C:/Users/Shruti/AppData/Local/Programs/Python/Python37-32/Lib/site-packages")

import tweepy
from wordcloud import WordCloud 
import re
from bs4 import BeautifulSoup as bs
#Twitter API credentials
consumer_key = "VX6UxbFfJiDa9P9xBZ3uaNIV5"
consumer_secret = "UTi4c720q6HzOTjyp6k2CsaVUNzO6CHuFjLPUzrQ36yIExRYSF"
access_key = "1183235731453403136-uSrpxEv7iAkuryoxYVXaWnL2NU7cEL"
access_secret = "VW71fMl1RQFV5Esgsz1IWpZ6NptT7uH6wD91wDMcvJB9R"

#function to read all tweets of a given user
def read_tweets(user_name):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    alltweets = []	
    new_tweets = api.user_timeline(screen_name = user_name,count=200)
    alltweets.extend(new_tweets)  #all tweets in one
    
    oldest = alltweets[-1].id - 1 #last of all tweets = oldest
    while 0<len(new_tweets)<2000:
        new_tweets = api.user_timeline(screen_name = user_name,
                                       count=200, 
                                       max_id=oldest
                                       )
        #save most recent tweets
        alltweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))          #monitor
 
    outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                  tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                  tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                  tweet._json["user"]["utc_offset"]] for tweet in alltweets]
    
    import pandas as pd
    tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                    "geo","id_str","lang","place","retweet_count","retweeted","source",
                                    "text","location","name","time_zone","utc_offset"])
    tweets_df["time"]  = pd.Series([str(i[0]) for i in outtweets])
    tweets_df["hashtags"] = pd.Series([str(i[1]) for i in outtweets])
    tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in outtweets])
    tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in outtweets])
    tweets_df["geo"] = pd.Series([str(i[4]) for i in outtweets])
    tweets_df["id_str"] = pd.Series([str(i[5]) for i in outtweets])
    tweets_df["lang"] = pd.Series([str(i[6]) for i in outtweets])
    tweets_df["place"] = pd.Series([str(i[7]) for i in outtweets])
    tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in outtweets])
    tweets_df["retweeted"] = pd.Series([str(i[9]) for i in outtweets])
    tweets_df["source"] = pd.Series([str(i[10]) for i in outtweets])
    tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
    tweets_df["location"] = pd.Series([str(i[12]) for i in outtweets])
    tweets_df["name"] = pd.Series([str(i[13]) for i in outtweets])
    tweets_df["time_zone"] = pd.Series([str(i[14]) for i in outtweets])
    tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in outtweets])
    tweets_df.to_csv(user_name+"_tweets.csv")
    return tweets_df


user_acc = read_tweets("realDonaldTrump")
