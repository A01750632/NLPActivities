from flair.data import Corpus
from flair.datasets import ColumnCorpus        

# define columns
columns = {0 : 'text', 1 : 'ner'}
# directory where the data resides
data_folder = 'data'
# initializing the corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'trainTweeter.txt',
                              test_file = 'testTweeter.txt',
                              dev_file = 'devTweeter.txt')
print(len(corpus.train))
print(corpus.train[0].to_tagged_string('ner'))
# tag to predict
tag_type = 'ner'
# make tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from typing import List
embedding_types : List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
        ## other embeddings
        ]
embeddings : StackedEmbeddings = StackedEmbeddings(
                                 embeddings=embedding_types)