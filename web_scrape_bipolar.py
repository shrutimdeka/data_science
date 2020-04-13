#!/usr/bin/env python
# coding: utf-8

# In[142]:


from selenium import webdriver
browser = webdriver.Chrome(executable_path = "C:/Users/Shruti/Downloads/chromedriver_win32_1/chromedriver.exe") #create session

from bs4 import BeautifulSoup  #for pulling data out of HTML and XML files


# In[173]:


import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.cluster import KMeans


# In[143]:


#################################### web scrape for new/test data ##########################
page = "https://www.bbc.com/news/world"
browser.get(page)


# In[146]:


test_article=[]
ps = browser.page_source
soup=BeautifulSoup(ps,"html.parser")
section = soup.body.find_all(attrs={"class": ( "gel-layout__item")})
for sec in section:
    test_article.append(sec.get_text()) #yep, separated stories on each line


# In[147]:


test_article ##new articles that may or may not be viral


# In[17]:


#remove frequently occuring words like 'am', 'pm', 'Jan', 'Feb', 'Mar'..., 
custom_stopwords = ['am', 'pm', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
                    'Oct', 'Nov', 'Dec', 'November', 'December', 'January', 'Monday', 
                   'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
#The other common stopwords will go away after we create
#a tfidf vector from the cleaned sentences
################################## Clean features function #########################

def cleaning_crew(articles):     #list of all articles
    articles_cleaned = []
    articles_no_stop = []
    
    #remove digits and symbols
    for ele in articles:
        ele = re.sub('[^a-zA-Z]+', " ", ele)
        articles_cleaned.append(ele) 
    
    #remove stopwords defined above
    for ele in articles_cleaned:
        tokens = ele.split()
        for words in custom_stopwords:
            if words in tokens: 
                tokens.remove(words)
            sentence = ' '.join(tokens)
        articles_no_stop.append(sentence)
    return articles_no_stop


# In[118]:


############################## CREATE tfidf vector ################
def create_tfidf(proper_sentence_all):                 #proper_df.cleaned_text
    Tfidf_vect = TfidfVectorizer(max_features=500, analyzer = 'word', 
                                 stop_words = 'english',
                                ngram_range =(1, 2))
    Tfidf_vect.fit(proper_sentence_all)                            #fit on entire document
    return Tfidf_vect

def transform_tfidf(Tfidf_vect, proper_sentence_all):
    Tfidf_all = Tfidf_vect.transform(proper_sentence_all)                  #transform data
    return Tfidf_all


# In[119]:


################################# training data ###################
articles = pd.read_csv("articles_virality.csv") #load prepared training data with label 1/0

#create tfidf vector
tfidf_vect = create_tfidf(articles['news'])
tfidf_vector = transform_tfidf(tfidf_vect, articles['news']) #transformed tfidf_vector for training


# In[128]:


## see training data as sparse array
tfidf_vect.get_feature_names() #features
#top 10 scored words
train = pd.DataFrame(tfidf_vector.toarray(), columns = tfidf_vect.get_feature_names())


# In[121]:


#see top 20 from tfidf_vect
indices = np.argsort(tfidf_vect.idf_)[::-1]
features = tfidf_vect.get_feature_names()
top_n = 20
top_features = [features[i] for i in indices[:top_n]]
top_features


# In[92]:


#append viral & news articles
#articles= viral.append(news_df, ignore_index=False, verify_integrity=False, sort=None)


# In[148]:


##################### create test set/ web scrape any info/news website ############

test_cleaned = cleaning_crew(test_article)
len(test_cleaned) #46 test cases

tfidf_vector_test = transform_tfidf(tfidf_vect, test_cleaned)


# In[149]:


################################ create model SVM ####################
#Chosen SVM model
def model_train(tfidf_train, y_train):                       #proper_df.cleaned_text (all text to train)
    
    SVM = svm.SVC(C=2.5, kernel='linear', gamma='auto')
    SVM = SVM.fit(tfidf_train, y_train)                     #train
    return SVM

 # predict the labels on new dataset
def model_predict(SVM, new_email_Tfidf):            #one or more tfidf docs
    predictions = SVM.predict(new_email_Tfidf)
    return predictions


# In[166]:


articles.virality=articles.virality.astype('int')
svm_model = model_train(tfidf_vector, articles.virality)


# In[167]:


svm_pred = model_predict(svm_model, tfidf_vector_test)


# In[169]:


test_results = pd.DataFrame(test_cleaned, columns=['text'])
test_results['virality_pred'] = svm_pred
test_results  #almost random/opposite prediction -> too many viral videos
###################################################################


# In[174]:


############################### clustering ###########################
k_model = KMeans(n_clusters = 2)
k_model.fit(tfidf_vector)


# In[176]:


k_pred =k_model.predict(tfidf_vector_test) #predict


# In[183]:


k_pred #1=viral


# In[178]:


test_results['kmeans_pred'] = k_pred #put into our test results dataframe


# In[ ]:


########################### Result ################
#1)Clustering was way better than SVM, not in terms of accuracy, but exclusivity
#    Since, not every other article can become viral as SVM had predicted.

#2) K_means clustering has predicted exactly one article as potential viral article
#   and frankly, it's cryptic headline do not look very promising

#3) News in times of coronavirus is an exception and a machine can not learn from
#    previous periods of headlines/viral articles. We will have to consider the articles
#    in these times as special cases or outliers.

#4) Most virals are based on videos, and the content and quality matters rather than
#    the nature (eg, singing, dancing, recipe videos). It is really interesting how
#    we could be able to quantify virality of a content.

