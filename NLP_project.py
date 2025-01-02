# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 00:43:44 2024

@author: BeButton
"""

#%%
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import ngrams

#%%
df_note = pd.read_csv('DATA//test.csv', usecols=[1])
df_depression = pd.read_csv('DATA//Suicide_Detection.csv', usecols=[1,2])

#%%
df_note.dropna(subset=['text'], axis=0,  inplace=True)
df_note = df_note.reset_index(drop=True)

#%%
stops = set(stopwords.words('english'))

#%% 

# Word Frequencies


#%% Before cleaning
df_note['length'] = 0
words = []
clean = []
tokens = []
tags = []
lemmed = []
stemmed = []

all_tokens = []

#%%
ps = PorterStemmer() 
lemmatizer = WordNetLemmatizer()

#%%
for i, row in df_note.iterrows():
    
    text = row['text'].lower()
    
    tx_tokenized = word_tokenize(text)
    lemma = [lemmatizer.lemmatize(w) for w in tx_tokenized]
    stemma = [ps.stem(w) for w in tx_tokenized]
    
    all_tokens.extend(tx_tokenized)
    
    pos_tags = nltk.pos_tag(tx_tokenized)
    pos_tags = [x[1] for x in pos_tags]
    
    text = text.strip()
    text = text.split()
    text = [re.sub(r'\W+', '', word) for word in text]
    text = [re.sub(r'\d+', '', word) for word in text]
    text = [word for word in text if len(word) >= 1]
    
    tx_clean = [re.sub(r'\W+', '', word) for word in stemma]
    tx_clean = [re.sub(r'\d+', '', word) for word in tx_clean]
    tx_clean = [word for word in tx_clean if (word not in stops) and (word != '')]
    
    df_note.loc[i, 'length'] = len(text)
    
    words.append((text))
    clean.append((tx_clean))
    tokens.append((tx_tokenized))
    tags.append((pos_tags))
    lemmed.append((lemma))
    stemmed.append((stemma))
    
#%%
df_note['words'] = words
df_note['clean'] = clean
df_note['tokens'] = tokens
df_note['tags'] = tags
df_note['lemma'] = lemmed
df_note['stem'] = stemmed

#%%
pos_tags = nltk.pos_tag(all_tokens)
df_post_tags = pd.DataFrame(pos_tags).T

#%%    
def getFrequent(data_set, column):
    words_freq = {}
    for row in data_set[column]:
        for word in row:
            if word not in words_freq.keys():
                words_freq[word] = 1
            else:
                words_freq[word] += 1
    freq = pd.DataFrame.from_dict(words_freq, orient='index').reset_index()
    freq.columns = ['words', 'count']
    freq = freq.sort_values('count', ascending=False)
    return freq

#%% frequency calculation
words_freq = getFrequent(df_note, 'words')
words_clean_freq = getFrequent(df_note, 'clean')
words_tokens = getFrequent(df_note, 'tokens')
words_tags = getFrequent(df_note, 'tags')
words_lemma = getFrequent(df_note, 'lemma')
words_stema = getFrequent(df_note, 'stem')

counter = Counter(all_tokens)

#%%
print(counter.most_common()[:10])
print(words_tokens[:10])

#%%
