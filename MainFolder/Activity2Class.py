'''Author: Liam Garay Monroy A01750632
Activity2Class for task2'''

from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import os

dirname = os.path.dirname(__file__)
PATH_TO_DATA = os.path.join(dirname, '../dataTweeter')
LOSS_TSV = os.path.join(dirname, '../resources/taggers/example-ner/loss.tsv')

class NER_training_model:
    def __init__(self,PERCENT_OF_DATASET_TO_TRAIN =0.1):
        print(f'\n{"":=^60}\n{"Second Task":=^60}\n{"":=^60}\n')
        self.PERCENT_OF_DATASET_TO_TRAIN = PERCENT_OF_DATASET_TO_TRAIN
        # define columns that we are going to use
        columns = {0 : 'text', 1 : 'ner'}
        # directory where de data is stored
        # initializing the corpus
        self.corpus = ColumnCorpus(PATH_TO_DATA, columns,
                                    train_file = 'train.txt',
                                    test_file = 'test.txt',
                                    dev_file = 'dev.txt')
        tag_type = 'ner'
        # make tag dictionary from the corpus
        tag_dictionary = self.corpus.make_label_dictionary(label_type=tag_type)
        embedding_types : List[TokenEmbeddings] = [
                WordEmbeddings('glove'),
                ]

        self.embeddings : StackedEmbeddings = StackedEmbeddings(
                                        embeddings=embedding_types)

        self.tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=self.embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    def train(self,learning_rate = 0.02, max_epochs = 10, mini_batch_size = 32):
        print(f'\n{"":=^80}\n"Corpus original: " {self.corpus}\n{"":=^80}\n')
        self.corpus.downsample(self.PERCENT_OF_DATASET_TO_TRAIN)
        print(f'\n{"":=^80}\n"Corpus with downsample: " {self.corpus}\n{"":=^80}\n')
        trainer : ModelTrainer = ModelTrainer(self.tagger, self.corpus)
        trainer.train('resources/taggers/example-ner',
                    learning_rate=learning_rate,
                    mini_batch_size=mini_batch_size,
                    max_epochs=max_epochs)

    
    def downsample_test(self):
        print(f'\n{"":=^80}\n"Corpus original: " {self.corpus}\n{"":=^80}\n')
        self.corpus.downsample(self.PERCENT_OF_DATASET_TO_TRAIN)
        print(f'\n{"":=^80}\n"Corpus with downsample: " {self.corpus}\n{"":=^80}\n')
        return f"{self.corpus}"
        
    def plot(self):
        '''Function to plot the results of the training'''
        trainLog = pd.read_csv(LOSS_TSV, sep='\t')
        print(trainLog)
        plt.plot(trainLog["EPOCH"],trainLog["TRAIN_LOSS"],'r--')
        plt.plot(trainLog["EPOCH"],trainLog["DEV_LOSS"],'y')
        plt.title('Loss')
        plt.legend(["TRAIN LOSS", "DEV LOSS"], loc ="upper right")
        plt.show()