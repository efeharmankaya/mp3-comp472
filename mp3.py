import csv
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gensim.downloader as api

# Download the pretrained model word vectors using gensim.downloader and save it. Run this code only once.
# wv = api.load('word2vec-google-news-300')
# wv.save('GoogleNews300.wordvectors')

# Loading the pretrained model word vectors
wv = KeyedVectors.load('GoogleNews300.wordvectors', mmap='r')    # Read-only
# Loading the synonym question-words into a data frame
df = pd.read_csv("synonyms.csv", delimiter=',')
print(df.head())

# Setting up analysis parameters
analysis = {
    'model_name' : 'word2vec-google-news-300',
    'corpus_size' : len(wv),
    'c' : 0,
    'v' : 0
}

# Finding best synonyms row by row
GoogleNews300_details = []
with open('GoogleNews300-details.csv', 'w') as file:    
    for i in range(len(df.index)):
        question = df.iloc[i]["question"]
        answer = df.iloc[i]["answer"]
        options = [df.iloc[i]['0'], df.iloc[i]['1'], df.iloc[i]['2'], df.iloc[i]['3']]
        # Check if the question-word and at least 1 option word are contained in the word vector
        if question in wv and (options[0] in wv or options[1] in wv or options[2] in wv or options[3] in wv):
            scores = [wv.similarity(question, options[0]), wv.similarity(question, options[1]),
                    wv.similarity(question, options[2]), wv.similarity(question, options[3])]
            guess = options[np.argmax(scores)]
            if guess == answer:
                label = "correct"
                analysis['c'] = analysis.get('c', 0) + 1
            else:
                label = "incorrect"
            analysis['v'] = analysis.get('v', 0) + 1
        else:
            guess = options[np.random.randint(0,3)]
            label = "guess"
        output_line = [[question, answer, guess, label]]
        # Write line to csv file
        writer = csv.writer(file, delimiter=',')
        writer.writerows(output_line)
with open('analysis.csv', 'a') as file:
    analysis['accuracy'] = analysis.get('c') / analysis.get('v') if analysis.get('v') > 0 else 0
    writer = csv.writer(file, delimiter=',')
    writer.writerows([analysis.values()])