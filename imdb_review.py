# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:14:31 2019

@author: Shruti
"""
import sys
sys.path.append("C:/Users/Shruti/AppData/Local/Programs/Python/Python37-32/Lib/site-packages")
from selenium import webdriver
browser = webdriver.Chrome(executable_path = "C:/Users/Shruti/chromedriver_win32/chromedriver.exe")

from bs4 import BeautifulSoup  #for pulling data out of HTML and XML files
page = "https://www.imdb.com/title/tt0120338/reviews?ref_=tt_urv"
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotVisibleException
browser.get(page)
import time
  
imdb=[]
for i in range(1, 9):
    #i=i+25
    try:
        button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]')
        button.click()
        time.sleep(5)
        ps = browser.page_source
        soup=BeautifulSoup(ps,"html.parser")
        rev = soup.findAll("div",attrs={"class","text"})
        imdb.extend(rev)
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break 
#too much data collected - ran out of memory, no further processing is possible
len(imdb) # 1k
set(imdb)
import re
#remove redundant symbols and words
rev_string = re.sub("[^A-Za-z" "]+"," ",str(imdb)).lower()
rev_string = re.sub("[0-9" "]+"," ",str(imdb))
clean_string = re.sub("[^A-Za-z" "]+"," ", rev_string).lower()

#tokenize the words
imdb_tokens = clean_string.split(" ")  #6k words
token_no = len(imdb_tokens)-1 #3 lakhs
imdb_words=[]
#stop words
for i in range(0, token_no) :
    if len(imdb_tokens[i]) > 3 :
        imdb_words.append(imdb_tokens[i])

with open("C:/Users/Shruti/Downloads/Excelr-assignment/Ass11-Text_Mining/stop.txt", 'r') as file:
    stopwords = file.read()
    
#add business specific stopwords, like film
stopwords = stopwords+'\n'+'film'+'\n'+'text'+'\n'+'show'+'\n'+'control''\n'+'movie''\n'+'titanic'
imdb_final = [w for w in imdb_words if not w in stopwords] #tokens
len(imdb_final) #1 lakhs

#wordcloud
import pandas as pd
import matplotlib.pyplot as plt

#import wordcloud  #also has stopwords file
from wordcloud import WordCloud
# Joinining all the reviews into single paragraph 
string = " ".join(imdb_final) #tokens to single string

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=5000, height=5000).generate(string)

plt.imshow(wordcloud_ip) 

#Positive wordcloud
with open("C:/Users/Shruti/Downloads/Excelr-assignment/Ass11-Text_Mining/positive-words.txt", 'r') as obj:
    pos= obj.read().split("\n")
pos_review = [word for word in imdb_words if word in pos ] #take tokenized version, not string
# Joinining all the reviews into single paragraph 
airpod_pos = " ".join(pos_review) 
wordcloud_pos = WordCloud(
                      background_color='black',
                      width=2400, height=2000).generate(airpod_pos)

plt.imshow(wordcloud_pos) 

#Negative wordcloud
with open("C:/Users/Shruti/Downloads/Excelr-assignment/Ass11-Text_Mining/negative-words.txt", 'r') as obj:
    neg= obj.read().split("\n")
neg_review = [word for word in imdb_words if word in neg ] #take tokenized version, not string

#for x in ['sinking' ,'tragedy','tragic', 'fall']:
#    neg_review.remove(x)

# Joinining all the reviews into single paragraph 
airpod_neg = " ".join(neg_review) 
wordcloud_neg = WordCloud(
                      background_color='black',
                      width=2400, height=2000).generate(airpod_neg)

plt.imshow(wordcloud_neg) #words like 'sinking' shouldn't be considered negative, it is part of the plot

#Positive to negative ratio of freq
len(pos_review)/len(neg_review) #positive words > negative words
pd.unique(pos_review)
pd.unique(neg_review)

#So overall, the reviews were largely positive!!