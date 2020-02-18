#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############################# Create features with only swear words ########################


# In[31]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# In[4]:


swear_words = pd.read_csv("C:/Users/Shruti/Downloads/swear_words.txt", header=None)
swear_words[0]


# In[5]:


#stemming of the swear words to match the stemmed words in corpus
ps = PorterStemmer()
stemmed_swears = [ps.stem(w) for w in swear_words[0]]


# In[97]:


### USE X_train Deploy instead for here on out
X_train = pd.read_csv("X_train_deploy.csv")
X_train.head()


# In[60]:


#clean email
def keep_text(email):                  #argument = one email
    body_text = []
    crude_text = (re.sub('[^a-zA-Z]+', ' ', email))
    body_text.append(crude_text)
    return body_text

def load_stopwords():
    stopwords = pd.read_csv("stop.txt", header = None)  #use STOP.txt file (sent to you)
    #add Subject, Subject: Re, etc to stopwords
    #Add name of days, name of people, http, etc
    stopwords.columns = ['words'] #name the column to access it easily
    stopwords.words.head() #check
    length = stopwords.shape[0] #number of rows, starts from 0
    stopwords.words[length] = 'subject'
    stopwords.words[length+1]='re'
    stopwords.words[length+2]='fw'
    stopwords.words[length+3]='cc'
    stopwords.words[length+4]='forwarded'
    stopwords.words[length+5]='hotmail'
    stopwords.words[length+6]='mail'
    stopwords.words[length+7]='image'
    return stopwords.words

def clean_email(email):  #single email
    ps = PorterStemmer()
    sentence = keep_text(email)
    sentence = sentence[0].strip()
    stopwords = load_stopwords()
    words = word_tokenize(sentence) 
    stemmed_words = [ps.stem(w) for w in words]
    #print(stemmed_words)
    removed_stop = [word for word in stemmed_words if word.lower() not in np.sort(stopwords)]  #STOPWORD REMOVAL
    clean_sentence = ' '.join(removed_stop)
    return clean_sentence

def remove_gibberish(row):
    proper_words_list= []
    proper_words_again = []
    tokens = row.split(' ')
    proper_words_list.append([p for p in tokens if len(p) > 3])  #too small words
    proper_words_again.append([k for k in proper_words_list[0] if len(k) < 13])  #too big words
    proper_sentence = ' '.join(proper_words_again[0])
    #print(proper_sentence)
    return(proper_sentence)


#call all clean functions
def preprocessing(email_all):                #give all emails at once
    proper_sentence_all =[]
    for i in range(0, len(email_all)):
        clean_sentence = clean_email(email_all[i])
        proper_sentence= remove_gibberish(clean_sentence)
        proper_sentence_all.append(proper_sentence)
    #proper_sentence_all = proper_sentence_all.dropna()
    return proper_sentence_all 


# In[101]:


#Clean
cleaned_email = preprocessing(list(X_train.content))


# In[103]:


cleaned_email = pd.DataFrame(cleaned_email, columns = ['text'])
cleaned_email['Class'] = X_train['Class']  #put corresponding labels of email before dropping na rows
cleaned_email = cleaned_email.dropna()    #drop empty rows
pd.DataFrame(cleaned_email).to_csv("cleaned_mail_blog.csv")     #to csv


# In[ ]:


cleaned_email = pd.read_csv("cleaned_mail_blog.csv")  #read csv


# In[105]:


cleaned_email.iloc[:, 1]


# In[106]:


#create freq vector for training data (word_vector)
vector_list=[]
for sentence in cleaned_email.iloc[:, 0]:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for swears in word_vector.columns:
        if swears in sentence_tokens:
            #print(swears)
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    vector_list.append(sent_vec)
    


# In[107]:


#create swear words as features
word_vector = pd.DataFrame(vector_list, columns = stemmed_swears) #451 columns  #created count vector for model


# In[43]:


def tranform_y_to_numerical(y_train):              #all email classes, proper_df.class
    for x in y_train:
        if x == 'Abusive':
            x = 1
        else:
            x= 0
    return y_train  

#Chosen SVM model
def model_train(tfidf_train, y_train):                       #proper_df.cleaned_text (all text to train)
    
    SVM = svm.SVC(C=2.5, kernel='linear', gamma='auto')
    SVM = SVM.fit(tfidf_train, y_train)                     #train
    return SVM

 # predict the labels on new dataset
def model_predict(SVM, new_email_Tfidf):            #one or more tfidf docs
    predictions = SVM.predict(new_email_Tfidf)
    return predictions


# In[108]:


#call functions after creating feature
y_train = tranform_y_to_numerical(cleaned_email.iloc[:, 1])


# In[150]:


############ TRAIN model ####################
svm = model_train(word_vector, y_train)


# In[120]:


############ TEST ##############
#read csv
x_test = pd.read_csv("X_test_deploy.csv")


#preprocess
cleaned = preprocessing(list(x_test.content))

#combine with y_test
df = pd.DataFrame(cleaned, columns=['text'])


# In[145]:


y_test = pd.read_csv("y_test_deploy.csv", header=None)
df['class'] = y_test.iloc[:, 1]
#dropna
df = df.dropna()


# In[147]:


#create vector
vector=[]
for sentence in df.iloc[:, 0]:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for swears in word_vector.columns:
        if swears in sentence_tokens:
            #print(swears)
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    vector.append(sent_vec)

#create swear words as features
w_vector = pd.DataFrame(vector, columns = stemmed_swears)


# In[148]:


#call functions after creating feature

y = tranform_y_to_numerical(df.iloc[:, 1])


# In[151]:



#predict
pred = svm.predict(w_vector)

#accuracy
print(f1_score(pred, y)) #45% HAHA

print(confusion_matrix(pred, y))   #11079   595
                                   # 70     274
print("accuracy ", accuracy_score(pred, y))  #94%


# In[153]:


#study the predictions
results_swear = pd.DataFrame(df.text, columns=['cleaned_text', 'prediction', 'actual'])
results_swear['prediction'] = pred
results_swear['actual'] = y
results_swear['cleaned_text'] = df.text
results_swear.to_csv('results_swear.csv')


# In[154]:


############################################################################################################


# In[67]:


############################ Start FRESH clustering ####################################
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# In[39]:


## load cleaned_train data with labels #################
cleaned_email = pd.read_csv("cleaned_mail_blog.csv")  #read csv
cleaned_email = cleaned_email.dropna()


# In[3]:


##create tfidf
def create_tfidf(proper_sentence_all):                 #proper_df.cleaned_text
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(proper_sentence_all)                            #fit on entire document
    return Tfidf_vect

def transform_tfidf(Tfidf_vect, proper_sentence_all):
    Tfidf_all = Tfidf_vect.transform(proper_sentence_all)                  #transform data
    return Tfidf_all


# In[13]:


#call function tfidf
tfidf_vect = create_tfidf(cleaned_email.text)
tfidf_trans = transform_tfidf(tfidf_vect,cleaned_email.text)


# In[14]:


cls = MiniBatchKMeans(n_clusters=2, random_state=0)
cls.fit(tfidf_trans)


# In[20]:


cls.predict(tfidf_trans)
print("labels ", cls.labels_)


# In[23]:


labels = cls.labels_
cleaned_email['labels'] = labels


# In[38]:


pd.unique(cleaned_email['Class'])


# In[32]:


##################### SVM using tfidf 1 gram ####################
#Chosen SVM model
def model_train(tfidf_train, y_train):                       #proper_df.cleaned_text (all text to train)
    
    SVM = svm.SVC(C=2.5, kernel='linear', gamma='auto')
    SVM = SVM.fit(tfidf_train, y_train)                     #train
    return SVM

 # predict the labels on new dataset
def model_predict(SVM, new_email_Tfidf):            #one or more tfidf docs
    predictions = SVM.predict(new_email_Tfidf)
    return predictions


# In[42]:


############# TRAIN ###############
#emails = pd.read_csv("train.csv")

#text = preprocessing(emails.content)
#df = merge_email_class(text, emails.Class[0:20])
tfidf_vect = create_tfidf(cleaned_email['text'])
tfidf = transform_tfidf(tfidf_vect, cleaned_email['text'])


# In[55]:


#y_train = tranform_y_to_numerical(cleaned_email['Class'])

cleaned_email[cleaned_email['Class'] == 'Abusive'] = 1
cleaned_email[cleaned_email['Class'] == 'Non Abusive'] = 0
pd.unique(cleaned_email.Class)


# In[58]:


svm = model_train(tfidf_trans,cleaned_email.Class)


# In[61]:



############ TEST ##############
#read csv
x_test = pd.read_csv("X_test_deploy.csv")


#preprocess
cleaned = preprocessing(list(x_test.content))

#combine with y_test
df = pd.DataFrame(cleaned, columns=['text'])

y_test = pd.read_csv("y_test_deploy.csv", header=None)
df['class'] = y_test.iloc[:, 1]
#dropna
df = df.dropna()


# In[65]:


#transform y
df[df['class'] == 'Abusive'] = 1
df[df['class'] == 'Non Abusive'] = 0


# In[62]:


#test tfidf
tfidf_test = transform_tfidf(tfidf_vect, df['text'])
#predict
pred = svm.predict(tfidf_test)

#accuracy
print(f1_score(pred, df['class'])) #91% - average

print(confusion_matrix(pred, df['class']))   #11108   104
                                             # 42     765
print("accuracy ", accuracy_score(pred, df['class']))  #98%


# In[64]:




