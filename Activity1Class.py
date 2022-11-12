'''Author: Liam Garay Monroy A01750632
Activity1Class for task1'''
from transformers import pipeline
import re

class Sentiment_Analysis:
    def __init__(self,filename = "tiny_movie_reviews_dataset.txt"):
        print(f'\n{"":=^60}\n{"First Task":=^60}\n{"":=^60}\n')
        self.filename = filename
        text = open(self.filename)
        lines = text.readlines()
        cleanLines = []    
        for line in lines: 
            regex_sub = re.sub(r"[,.;\"@#?!&$]+", ' ', line)  
            regex_sub = re.sub(r"\s+", ' ', regex_sub)
            cleanLines.append(regex_sub)
        self.cleanLines = cleanLines
    
    def prediction(self,sentiment_model = pipeline("sentiment-analysis")):
        dictionary = sentiment_model(self.cleanLines)
        print(f'\n{"":=^60}\n{"Labels from sentiment Analysis":=^60}\n{"":=^60}\n')
        for prediction in dictionary:
            print(prediction["label"])


    