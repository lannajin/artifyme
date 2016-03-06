#import nltk
#import sklearn
import numpy as np
import pandas as pd
import re
from gensim.models import doc2vec
import gensim
#import pickle
import random
#import concurrent.futures
import os

os.chdir('C:/Users/lannajin/Documents/GitHub/artifyme')
#df = pd.read_csv('surrey-art.csv', encoding='cp1252')    #Surrey Art Data
df = pd.read_csv('van-art.csv')  #Vancouver Art Data 


#Remove rows with NaNs for description
df = df.dropna(subset=['description'])
#Create id tag for art work (in case there are multiples without data)
df['id'] = ['pid_'+str(i) for i in range(0,len(df))]

####################################################
#   1. Clean-up sentences
####################################################
#Clean References:
def cleanSent(var):
	sent = []
	for i in var:
		tmp = re.sub("[^a-zA-Z]", " ", i) #Remove non-characters
		tmp = tmp.lower().split()	#Tokenize sentences: convert to lower case and split them into individual words 
		sent.append(' '.join(tmp)) #join words back together into string
	return sent

df['Sentence'] = cleanSent(df.description) 
df['ArtState'] = cleanSent(df.artist_state) 

#Convert to LabeledSentence object to feed into gensim doc2vec model:
LabeledSentences = []
for i in range(0,len(df)):
    LabeledSentences.append(doc2vec.LabeledSentence(df.Sentence[i].split(), df.id[i])
	LabeledSentences.append(doc2vec.LabeledSentence(df.ArtState[i].split(), df.id[i])
	)
#OR, (depending on how labeled sentences need to be put in...)
LabeledSentences = []
for i in range(0,len(df)):
    LabeledSentences.append(doc2vec.LabeledSentence([df.Sentence[i].split(), df.ArtState[i].split()], df.id[i]))
  
#https://linanqiu.github.io/2015/05/20/word2vec-sentiment/

####################################################
#   2. Doc2Vec Model Training
####################################################
nfeatures = 500
model = gensim.models.doc2vec.Doc2Vec(workers=1,size=nfeatures, window=10, min_count=1,alpha=0.025, min_alpha=0.025, batch_words=500)
#, load_word2vec_format(fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict')
#Build the vocabulary table: digesting all the words and filtering out the unique words, and doing some basic counts on them
model.build_vocab(LabeledSentences) 

#Train Doc2Vec
from random import shuffle
#Randomize order of sentences
for epoch in range(10):
    shuffle(LabeledSentences)
    model.train(LabeledSentences)
    model.alpha -= 0.0002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

# store the model to mmap-able files
model.save('doc2vec_model.doc2vec')
df.to_csv('df_out.csv')
#################################################
# PREDICT
#################################################
userInput = 'Vancouver is beautiful today!'  #Input sentence

def cleanInput(userInput):
    Input1 = re.sub("[^a-zA-Z]", " ", userInput) #Only extract words
    Input = Input1.lower().split() #Tokenize sentences: convert to lower case and split them into individual words
    return Input

cI = cleanInput(userInput)  #Clean userInput sentence

#Infer a vector for given post-bulk training document. Document should be a list of (word) tokens.
userVec = model.infer_vector(cI) 
#Find the top-N most similar docvecs known from training. Positive docs contribute positively towards the similarity, negative docs negatively. 
#This method computes cosine similarity between a simple mean of the projection weight vectors of the given docs. Docs may be specified as vectors, integer indexes of trained docvecs, or if the documents were originally presented with string tags, by the corresponding tags.
#Here, doc is given as infered vector
output = model.docvecs.most_similar(positive=[userVec], topn=30)
the_result = pd.DataFrame(output, columns=['id','probability'])

#merge with original data frame
the_result.merge(df, how='left', on='id')