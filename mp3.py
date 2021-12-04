import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import gensim.downloader as api

# Download the pretrained model using gensim.downloader and save it. Run this code only once.
model = api.load('word2vec-google-news-300')
#model.save('word2vec.model')

# Loading the pretrained model 
#model - Word2Vec.load('word2vec.model')

# Loading the synonym question-words into a data frame
df = pd.read_csv("synonyms.csv", delimiter=',')
print(df.head())

# Finding synonyms row by row
for i in range(len(df.index)):
    question = df.iloc[[i]][["question"]]
    answer = df.iloc[[i]][["answer"]]
    options = [df.iloc[[i]][['0']], df.iloc[[i]][['1']], df.iloc[[i]][['2']], df.iloc[[i]][['3']]]
    scores = [model.similarity(question, options[0]), model.wv.similarity(question, options[1]),
                model.similarity(question, options[2]), model.wv.similarity(question, options[3])]
    predicted_synonym = scores[np.argmax(scores)]
    print(predicted_synonym)
