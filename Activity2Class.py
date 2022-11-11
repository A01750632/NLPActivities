from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

class NER_training_model:
    def __init__(self,Path_To_Data,PERCENT_OF_DATASET_TO_TRAIN):
        self.PERCENT_OF_DATASET_TO_TRAIN = PERCENT_OF_DATASET_TO_TRAIN
        # define columns
        columns = {0 : 'text', 1 : 'ner'}
        # directory where the data resides
        self.Path_To_Data = Path_To_Data
        # initializing the corpus
        self.corpus = ColumnCorpus(self.Path_To_Data, columns,
                                    train_file = 'train.txt',
                                    test_file = 'test.txt',
                                    dev_file = 'dev.txt')
        
        # tag to predict
        tag_type = 'ner'
        # make tag dictionary from the corpus
        tag_dictionary = self.corpus.make_label_dictionary(label_type=tag_type)
        embedding_types : List[TokenEmbeddings] = [
                WordEmbeddings('glove'),
                ## other embeddings
                ]

        self.embeddings : StackedEmbeddings = StackedEmbeddings(
                                        embeddings=embedding_types)

        self.tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=self.embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)
    def train(self,learning_rate = 0.02, max_epochs = 10, mini_batch_size = 32):
        print("--- 1 Original ---")
        print(self.corpus)
        downsample = self.corpus.downsample(self.PERCENT_OF_DATASET_TO_TRAIN)

        print("--- 2 Downsampled ---")
        print(downsample)
        trainer : ModelTrainer = ModelTrainer(self.tagger, self.corpus)
        trainer.train('resources/taggers/example-ner',
                    learning_rate=learning_rate,
                    mini_batch_size=mini_batch_size,
                    max_epochs=max_epochs)
        
    def plot(self):
        trainLog = pd.read_csv('resources/taggers/example-ner/loss.tsv', sep='\t')
        print(trainLog)
        plt.plot(trainLog["EPOCH"],trainLog["TRAIN_LOSS"],'r--')
        plt.plot(trainLog["EPOCH"],trainLog["DEV_LOSS"],'y')
        plt.title('Loss')
        plt.show()