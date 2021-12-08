# Mini-Project 3 Experiments with Word Embeddings

This project aims to experiment with word embeddings to solve a synonym test and compare the results between different models, random choice and a human gold-standard.

## Setup
```bash
> pip install -r requirements.txt
```

## Run Experiments
Each model run will generate a unique details file that shows the results of each synonym check in the form `synonym word, correct answer, guess, result`. If the result is 'guess' then none of the words given in the question were found in the model's corpus and a random guess was made. 
```bash
> python mp3.py
```

## Loading Models
The required model names are automatically loaded through the form below. Additional models can be found in the [Genism Data Repo](https://github.com/RaRe-Technologies/gensim-data) or by running `api.info().get('models')`. While running mp3.py the models listed in the models array are loaded and saved for future use. The inital loading process can take upwards of 5 minutes as the models can range from 100MB to 1.2GB. However, after the first load, the models can be accessed directly through the saved wordvector files in the models directory.
```python
models = [
    {'model_name' : 'word2vec-google-news-300', 'model_file_name' : 'GoogleNews300.wordvectors', 'details_file_name' : 'GoogleNews300-details.csv'}
]
```
