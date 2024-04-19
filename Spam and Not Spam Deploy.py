# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:36:58 2024

@author: Admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df1 =  pd.read_csv(r"C:\Users\Admin\Downloads\spam.csv",encoding="latin")
df1

df1.info()

df1.drop(columns = ["Unnamed: 2","Unnamed: 3", "Unnamed: 4"],inplace =True)
df1
df1.isnull().sum()
df1.duplicated().sum()

df1.drop_duplicates(inplace =True)

df1.reset_index(drop =True,inplace =True)
df1
#steps
# 1. Data Cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvement 
# 7. Website 
# 8. Deploy on Heroku

df1.rename(columns = {"v1":"Target","v2":"Text"},inplace =True)
df1


from sklearn.preprocessing import LabelEncoder

lr =  LabelEncoder()
df1["Target"] =  lr.fit_transform(df1.Target)
df1


# ----------------------------EDA---------------------------------
df1.Target.value_counts()

plt.pie(df1.Target.value_counts(),autopct = "%1.1f%%",labels=["Not Spam","Spam"])
plt.title("Distribution of SPAM & NOT SPAM")

# -----------------------------Data Is Imbalanced------------------------------------------------
# import nltk

# nltk.download()
# nltk.download('punkt')

# here we are creating 3 columns for deeper analysis

# number of char
# number of words
# number of sentences



length = []
for x in df1.Text:
    length.append(len(x))
    

df1["length"] = length
df1
# --------------------------------------------------

# number of words
word = []

for x in df1.Text:
    word.append(len(x.split()))    

word
# ----------------------------------------------
from nltk.tokenize import word_tokenize
word = []
for x in df1.Text:
    word.append(len(word_tokenize(x)))

df1["word"] =word
df1

# ----------------------------------------------
# number of sentence
from nltk.tokenize import sent_tokenize
sent =[]

for x in df1.Text:
    sent.append(len(sent_tokenize(x)))

df1["sent_len"] =sent
df1
# -------------------------------------------------
df1.describe()
# here we can analyze that max 910 ltters are there
# 220 word and 38 sentence in one mail.

# avg 78 character in one mail
# avg 18 word in one mail
# avg 1.96  means 2 sentences in each mail are there

ham = df1.loc[df1.Target ==0]
ham
ham.describe()



spam = df1.loc[df1.Target ==1]
spam
spam.describe()


# here we can see that spam msg or mails are more lengthy then ham
import seaborn as sns
sns.histplot(data = ham,x ="length")
sns.histplot(data = spam,x ="length")



sns.histplot(data = ham,x ="word")
sns.histplot(data = spam,x ="word")

# here we can see that maximum ham messages made of less characters, wehre spam more character

# here we can see that maximum ham messages made of less word, wehre spam more words

# outliers are also there....see in ham

sns.pairplot(df1, hue ="Target")

# sns.heatmap(df1.corr())
#lower case
#toknize
#stopwords
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps  = PorterStemmer()

def clean_data(text):
    text  = text.lower()
    
    text =  word_tokenize(text)
    
    y  = []
    for x in text:
        if x.isalnum():
            y.append(x)
            
    text = y[:]
    y.clear()
    
    for x in text:
        if x not in stopwords.words("english") and x not in string.punctuation:
            y.append(x)
            
    
    text = y[:]
    y.clear()
    for x in text:
        y.append(ps.stem(x))
        
    
    return " ".join(y)
    
    
clean_data("Hello, Taral MEhta lecture speaking game playing  5664 fs5%% ##%5 !!")


df1["Cleaned_Text"] = df1.Text.apply(clean_data)
df1["Cleaned_Text"]
df1
ham = df1.loc[df1.Target ==0]
ham
ham.describe()



spam = df1.loc[df1.Target ==1]
spam
spam.describe()

from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')

spam_wc = wc.generate(spam['Cleaned_Text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')

ham_wc = wc.generate(ham['Cleaned_Text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# --------------------------SPAM 30 WORDS ----------------------------    
spam_word = []

for sentences in spam.Cleaned_Text.tolist():
    for word in sentences.split():
        print(word)
        spam_word.append(word) 
        
spam_word        
len(spam_word) #total 9939 words are there 

from collections import Counter
Counter(spam_word)
mc = Counter(spam_word).most_common(30)

mc_df = pd.DataFrame(mc,columns =["Repeated Word","Number Of Frequescy"])
mc_df


plt.figure(figsize=(12,12))
sns.barplot(data = mc_df,x = "Repeated Word", y= "Number Of Frequescy")
plt.xticks(rotation ="vertical")

# --------------------------HAM 30 WORDS ----------------------------    

ham_word = []


ham.Cleaned_Text.tolist()

for sentence in ham.Cleaned_Text.tolist():
    for word in sentence.split():
        print(word)
        ham_word.append(word)

len(ham_word)

from collections import Counter
hmc= Counter(ham_word).most_common(30)
hmc_df = pd.DataFrame(hmc,columns =["Repeated Word","Number Of Frequescy"])
hmc_df

plt.figure(figsize=(12,12))
sns.barplot(data =hmc_df,x ="Repeated Word",y ="Number Of Frequescy" )
plt.xticks(rotation ="vertical")

# --------------------------------MODEL BUILDING---------------------------------
# here we have to conevert text into vectors or number
# 1. Bag of words
# 2. TFIDF
# 3. wrod2 vector

# ----------------------------bag of words--------------------------
from sklearn.feature_extraction.text import CountVectorizer  #bag of words
cv = CountVectorizer()
X = cv.fit_transform(df1["Cleaned_Text"]).toarray()
X.shape
cv.get_feature_names_out()

df_cv = pd.DataFrame(X,columns = cv.get_feature_names_out())
df_cv
y = df1.Target.values
y
df_cv_out = pd.DataFrame(y,columns=["output"])
df_cv_out


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df_cv,df_cv_out,test_size=0.2,random_state=222)

x_train
x_test
y_train
y_test

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=222)

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,classification_report
from sklearn.naive_bayes import GaussianNB
gnb =GaussianNB()
gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
y_pred1
gnb.score(x_test,y_test)
confusion_matrix(y_test,y_pred1)
precision_score(y_test,y_pred1)

print(classification_report(y_test,y_pred1))



from sklearn.naive_bayes import MultinomialNB
mnb =MultinomialNB()
spam_detect_model = mnb.fit(x_train,y_train)
mnb.predict(x_test)
mnb.score(x_test,y_test)
y_pred2 = mnb.predict(x_test)
y_pred2
confusion_matrix(y_test,y_pred2)
precision_score(y_test,y_pred2)


from sklearn.naive_bayes import BernoulliNB
bnb= BernoulliNB()
bnb.fit(x_train,y_train)
bnb.predict(x_test)
bnb.score(x_test,y_test)
y_pred3= bnb.predict(x_test)
y_pred3
confusion_matrix(y_test,y_pred3)
precision_score(y_test,y_pred3)


# ----------------------------TF IDF--------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df1["Cleaned_Text"]).toarray()
X.shape
y = df1.Target.values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=222)



from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.naive_bayes import GaussianNB
gnb =GaussianNB()
gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
y_pred1
gnb.score(x_test,y_test)
confusion_matrix(y_test,y_pred1)
precision_score(y_test,y_pred1)



from sklearn.naive_bayes import MultinomialNB
mnb =MultinomialNB()
mnb.fit(x_train,y_train)
mnb.predict(x_test)
mnb.score(x_test,y_test)
y_pred2 = mnb.predict(x_test)
y_pred2
confusion_matrix(y_test,y_pred2)
precision_score(y_test,y_pred2)



from sklearn.naive_bayes import BernoulliNB
bnb= BernoulliNB()
bnb.fit(x_train,y_train)
bnb.predict(x_test)
bnb.score(x_test,y_test)
y_pred3= bnb.predict(x_test)
y_pred3
confusion_matrix(y_test,y_pred3)
precision_score(y_test,y_pred3)

#tf-idf --> MNB 
# import pickle
# saved_model=pickle.dumps(spam_detect_model)
# saved_model


# modelfrom_pickle = pickle.loads(saved_model) 
# modelfrom_pickle

# y_pred=modelfrom_pickle.predict(x_test)
# print(accuracy_score(y_test,y_pred))

# import joblib
# joblib.dump(spam_detect_model,'pickle.pkl')

# joblib.dump(X,'transform.pkl')

# joblib.dump(X,"")


# --------------------------------------------------
# https://medium.com/@abinayamahendiran/building-an-end-to-end-nlp-application-an-overview-ef0221c4ab1f#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjkzYjQ5NTE2MmFmMGM4N2NjN2E1MTY4NjI5NDA5NzA0MGRhZjNiNDMiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDQ4NDQxNDMzMDgyMTk0NDEzMDgiLCJlbWFpbCI6IjkxbHdtaDA5NkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmJmIjoxNzEyOTg4NDYxLCJuYW1lIjoiTGl2ZXRlY2hJTkRJQSIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKMjZyTU1teWJkVEpZeENoRnl1cmxkWVNFRDFCOFR0bTV1SjNrZ1FrYjRCV0Q2N2FUNT1zOTYtYyIsImdpdmVuX25hbWUiOiJMaXZldGVjaElORElBIiwiaWF0IjoxNzEyOTg4NzYxLCJleHAiOjE3MTI5OTIzNjEsImp0aSI6ImQ0M2U5YzdiODE5ZDJjZWVlZWIyMDY4ZmNkMzViNDk5YjdlZTcxYmUifQ.n-p3mQAuS6uqnegBFsj6tq2oPD02P1t6ae1KYgCj0z6NgChvVN37MgVjGglZZkqDKUHLlMrNPdy0VEIqDiAME0UnX4IQP8NKOKQurEdCrqVnbkCmPPwXJeJh3WIv-yPI_vwH6zpBCYx8LMwaqQbXpI6Fu9b0KdJES_unY1vMFzGJApIJ7l9J6wCZ_V_OmD8aVpQRDFhQVO7AzOJUEdDCYTx_YeGPxcy8HpNI3w83ykps9HzxF68__iLqoPqr6gcls-pMvV3P2i5MVoWecpqC6S3tgNhWG9BdGWFt-0f2qjmglwIQtQ6dd8405CCdX2t0YaY78MnSmz-w4i9oOfQGdw

# ref  -https://www.youtube.com/watch?v=yY1FXX_GSco

# ref -  https://www.youtube.com/watch?v=YncZ0WwxyzU&list=PLKnIA16_RmvY5eP91BGPa0vXUYmIdtfPQ&index=3
# ref - https://www.youtube.com/watch?v=1xtrIEwY_zY&list=PLKnIA16_RmvY5eP91BGPa0vXUYmIdtfPQ
# https://www.analyticsvidhya.com/blog/2022/06/an-end-to-end-guide-on-nlp-pipeline/



 
    
