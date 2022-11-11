#Libraries needed for the tests
import requests
import json
from transformers import pipeline
import re
from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

#------------------ Test Sentiment Analysis --------------------------------
sentiment_model = pipeline("sentiment-analysis")        
dictionary = sentiment_model(["i love you","i hate you"])
predictions = []
for prediction in dictionary:
        predictions.append(prediction["label"])
assert predictions[0] == "POSITIVE"
assert predictions[1] == "NEGATIVE"

#--- subTest Sentiment Analysis RE -------
regex_sub = re.sub(r"[,.;\"@#?!&$]+", ' ', 'Hello@;" everything okay" ,.,.&here')  
regex_sub = re.sub(r"\s+", ' ', regex_sub)
assert regex_sub == "Hello everything okay here"


#------------------ Test NER training model corpus --------------------------------
PERCENT_OF_DATASET_TO_TRAIN = 0.1
# define columns
columns = {0 : 'text', 1 : 'ner'}
# directory where the data resides
Path_To_Data = "dataTweeter"
# initializing the corpus
corpus: Corpus = ColumnCorpus(Path_To_Data, columns,
                            train_file = 'train.txt',
                            test_file = 'test.txt',
                            dev_file = 'dev.txt')

# tag to predict
tag_type = 'ner'
# make tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
embedding_types : List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
        ## other embeddings
        ]

embeddings : StackedEmbeddings = StackedEmbeddings(
                                embeddings=embedding_types)

tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                    embeddings=embeddings,
                                    tag_dictionary=tag_dictionary,
                                    tag_type=tag_type,
                                    use_crf=True)

# Checking corpus has all the training data set
assert len(corpus.train) == 2394

#--- subTest NER training model corpus downsample -------
downsample = corpus.downsample(PERCENT_OF_DATASET_TO_TRAIN)
assert len(downsample.train) == 239


#------------------ Test API 1 --------------------------------
url = 'https://translate.argosopentech.com/translate'
myobj = {'q': "Hello World",
                'source': "en",
                'target': "es"}
response1 = requests.request("POST", url, data=json.dumps(myobj), headers={"content-type": "application/json"})
translation = json.loads(response1.text)
textTranslated1 = translation["translatedText"]
assert textTranslated1 == "Hola Mundo"

#------------------ Message Everything Ok --------------------------------
print(f'\n{"":=^60}\n{"if this you see this message all the test are correct":=^60}\n{"":=^60}\n')