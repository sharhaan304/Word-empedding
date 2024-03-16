#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


paragraph="""Mahinda Rajapaksa (Sinhala: මහින්ද රාජපක්ෂ; Tamil: மஹிந்த ராஜபக்ஷ; born Percy Mahendra Rajapaksa; 18 November 1945) is a Sri Lankan politician. He served as the President of Sri Lanka from 2005 to 2015; the Prime Minister of Sri Lanka from 2004 to 2005, 2018, and 2019 to 2022;[2] the Leader of the Opposition from 2002 to 2004 and 2018 to 2019, and the Minister of Finance from 2005 to 2015 and 2019 to 2021. He has been a Member of Parliament (MP) for Kurunegala since 2015.[3]

Rajapaksa is a lawyer by profession and was first elected to the Parliament of Sri Lanka in 1970. He served as the leader of the Sri Lanka Freedom Party from 2005 to 2015. Rajapaksa was sworn in for his first six-year term as president on 19 November 2005. He was subsequently re-elected in 2010 for a second term.[4] Rajapaksa was defeated in his bid for a third term in the 2015 presidential election by Maithripala Sirisena, and he left office on 9 January 2015. Later that year, Rajapaksa unsuccessfully sought to become prime minister in the 2015 parliamentary election; that year, the United People's Freedom Alliance was defeated but was elected as the Member of Parliament for the Kurunegala District.[5]

On 26 October 2018, Rajapaksa was appointed to the office of prime minister by President Maithripala Sirisena after the United People's Freedom Alliance withdrew from the unity government. The incumbent, Ranil Wickremesinghe, refused to accept his dismissal, stating that it was unconstitutional. This disagreement resulted in a constitutional crisis. The Sri Lankan Parliament passed two no-confidence motions brought against Rajapaksa on 14 and 16 November 2018. Failing to follow proper procedures, President Sirisena rejected both. On 3 December 2018, a court suspended Rajapaksa's powers as prime minister, ruling that his cabinet could not function until establishing its legitimacy. Rajapaksa resigned from the post of prime minister on 15 December 2018. Wickremesinghe was re-appointed as prime minister, and Rajapaksa was appointed Leader of the Opposition.[6]

Rajapaksa became the leader of the Sri Lanka Podujana Peramuna in 2019, splitting the Sri Lanka Freedom Party. He became prime minister again on 21 November 2019 after being appointed by his brother, Gotabaya Rajapaksa, who had become president on 18 November after winning the 2019 Sri Lankan presidential election. On 9 August 2020, Rajapaksa was sworn in as Prime Minister of Sri Lanka for the fourth time at a Buddhist temple on Colombo's outskirts. On 3 May 2022, a motion of no confidence aimed at Rajapaksa and his cabinet was declared by opposition leaders.[7] He was targeted during the 2022 Sri Lankan protests over the corruption and mismanagement by the Rajapaksa family which led to an economic crisis that brought Sri Lanka to the point of bankruptcy as it defaulted on its loans for the first time in its history since independence. Protesters called him "Myna" and demanded his resignation which he resisted. On 9 May 2022, Mahinda Rajapaksa organised his supporters at his official residence who were brought by buses and led by SLPP MPs. The loyalists then attacked protestors at Temple Trees before assaulting protestors at Galle Face as attacks were carried out simultaneously against protests in other areas; however this intensified protests and retaliatory violence against Rajapaksa loyalists erupted islandwide and Mahinda Rajapaksa submitted his letter of resignation the same day.[2][8]

During Rajapaksa's political career, he has been accused of multiple crimes including war crimes during the last years of the Sri Lankan civil war as well as other criminal accusations including human rights violations during his presidency, corruption and for instigating violence on anti-government protestors on 9 May 2022.[9][10][11][12] As of 2023 he has been sanctioned by Canada for human rights violations.[13]"""


# In[3]:


paragraph


# In[19]:


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[20]:


## tokenization-- converts paragraph-sentences-words
nltk.download('punkt')
sentences=nltk.sent_tokenize(paragraph)


# In[6]:


print(sentences)


# In[7]:


type(sentences)


# In[8]:


stemmer=PorterStemmer()


# In[9]:


stemmer.stem('Myna')


# In[24]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


# In[25]:


lemmatizer=WordNetLemmatizer()


# In[26]:


lemmatizer.lemmatize('going')


# In[27]:


import re
corpus=[]
for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ',sentences[i])
    review=review.lower()
    corpus.append(review)


# In[16]:


corpus


# In[28]:


##stemming
for i in corpus:
    words=nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(stemmer.stem(word))


# In[29]:


##lemmatization
for i in corpus:
    words=nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(lemmatizer.lemmatize(word))


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[31]:


X=cv.fit_transform(corpus)


# In[32]:


cv.vocabulary_


# In[33]:


corpus[0]


# In[35]:


X[0].toarray()


# In[ ]:




