import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gensim.models import KeyedVectors
import gensim.downloader as api
import os.path

# Show list of available models
# pr(api.info().get('models'))

models = [
    # base model
    {'model_name' : 'word2vec-google-news-300', 'model_file_name' : 'GoogleNews300.wordvectors', 'details_file_name' : 'GoogleNews300-details.csv'},

    # # same embedding size (300)
    #{'model_name' : 'glove-wiki-gigaword-300', 'model_file_name' : 'GloveWikiGigaword300.wordvectors', 'details_file_name' : 'GloveWikiGigaword300-details.csv'},
    #{'model_name' : 'fasttext-wiki-news-subwords-300', 'model_file_name' : 'FastTextWikiNews300.wordvectors', 'details_file_name' : 'FastTextWikiNews300-details.csv'},
        
    # # different embedding size (25,100)
    #{'model_name' : 'glove-twitter-100', 'model_file_name' : 'GloveTwitter100.wordvectors', 'details_file_name' : 'GloveTwitter100-details.csv'},
    #{'model_name' : 'glove-wiki-gigaword-100', 'model_file_name' : 'GloveWikiGigaword100.wordvectors', 'details_file_name' : 'GloveWikiGigaword100-details.csv'},
]

def load_models():
    # Download the pretrained model word vectors using gensim.downloader and save it. Runs only once.
    for model in models:
        if not os.path.exists(f'models/{model.get("model_file_name")}'):
            wv = api.load(model.get('model_name'))
            wv.save(f'models/{model.get("model_file_name")}')

def run():
    # Ensure all models are loaded and saved
    load_models()    
    # Create a list of will be used to plot the performance of each model
    model_accuracies = [] 
    for model in models:
        wv = KeyedVectors.load(f'models/{model.get("model_file_name")}', mmap='r')    # Read-only
        df = pd.read_csv("models/synonyms.csv", delimiter=',')
        
        # Set analysis parameters to current model
        analysis = {
            'model_name' : model.get('model_name'),
            'corpus_size' : len(wv),
            'c' : 0,
            'v' : 0
        }
        
        with open(f'output/{model.get("details_file_name")}', 'w') as file:    
            for i in range(len(df.index)):
                question = df.iloc[i]["question"]
                answer = df.iloc[i]["answer"]
                options = [df.iloc[i]['0'], df.iloc[i]['1'], df.iloc[i]['2'], df.iloc[i]['3']]
                # Check if the question-word and at least 1 option word are contained in the word vector
                if question in wv and (options[0] in wv or options[1] in wv or options[2] in wv or options[3] in wv):
                    scores = [wv.similarity(question, option) for option in options if wv.get_index(option, -1) > -1]
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
        with open('output/analysis.csv', 'a') as file:
            analysis['accuracy'] = analysis.get('c') / analysis.get('v', -1) if analysis.get('v', 0) > 0 else 0
            writer = csv.writer(file, delimiter=',')
            writer.writerows([analysis.values()])
            
        # Append the model accuracy to the list of accuracies
        model_accuracies.append(analysis['accuracy'])
    
    # Plotting the model performances
    model_names = [m["model_name"] for m in models]
    fig = plt.figure(figsize = (10, 5))
    plt.bar(model_names, model_accuracies, color ='maroon',
            width = 0.2)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Accuracy comparison of models")
    plt.grid(color='#95a5a6', linestyle='--', axis='y')
    plt.savefig("output/model-accuracies.pdf")

if __name__ == '__main__':
    run()
