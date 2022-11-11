from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
from typing import List
import matplotlib.pyplot as plt
import pandas as pd        
from flair.training_utils import EvaluationMetric

N_EXAMPLES_TO_TRAIN = 0.1
# define columns
columns = {0 : 'text', 1 : 'ner'}
# directory where the data resides
data_folder = 'data'
# initializing the corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'trainTweeter.txt',
                              test_file = 'testTweeter.txt',
                              dev_file = 'devTweeter.txt')


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


print("--- 1 Original ---")
print(corpus)
downsample = corpus.downsample(N_EXAMPLES_TO_TRAIN)

print("--- 2 Downsampled ---")
print(downsample)
trainer : ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train('resources/taggers/example-ner',
              learning_rate=0.02,
              mini_batch_size=32,
              max_epochs=10)

trainLog = pd.read_csv('resources/taggers/example-ner/loss.tsv', sep='\t')
print(trainLog)
plt.plot(trainLog["EPOCH"],trainLog["TRAIN_LOSS"],'r--')
plt.plot(trainLog["EPOCH"],trainLog["DEV_LOSS"],'y')
plt.title('Loss')
plt.show()