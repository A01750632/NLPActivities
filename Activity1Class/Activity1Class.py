'''Author: Liam Garay Monroy A01750632
Activity1Class for task1'''
from transformers import pipeline
import re
from typing import List
import os
dirname = os.path.dirname(__file__)
FILENAME = os.path.join(dirname, '../tiny_movie_reviews_dataset.txt')


class Sentiment_Analysis:
    def __init__(self):
        print(f'\n{"":=^60}\n{"First Task":=^60}\n{"":=^60}\n')
        # dont need to save filename if you only use it here! 

        # you are not closing the file here! more info: 
        # https://www.freecodecamp.org/news/with-open-in-python-with-statement-syntax-example/#:~:text=So%2C%20the%20open()%20function,the%20mode%2C%20and%20the%20encoding.

        with open(FILENAME, "r") as text: 
            lines = text.readlines()
            self.cleaned_lines = self._clean_lines(lines)
            text.close()

    def _clean_lines(self, lines: List[str]):
        cleaned_lines = []   # camel case only used in class defs! not in vars   
        for line in lines: 
            regex_sub = re.sub(r"[,.;\"@#?!&$]+", ' ', line)  
            regex_sub = re.sub(r"\s+", ' ', regex_sub)
            cleaned_lines.append(regex_sub)
        return cleaned_lines
    
    def prediction(self,sentiment_model = pipeline("sentiment-analysis")):
        dictionary = sentiment_model(self.cleaned_lines)
        print(f'\n{"":=^60}\n{"Labels from sentiment Analysis":=^60}\n{"":=^60}\n')
        for prediction in dictionary:
            print(prediction["label"])


    