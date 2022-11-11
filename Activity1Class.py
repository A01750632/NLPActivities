from transformers import pipeline
import re

class Sentiment_Analysis:
    def __init__(self,filename):
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
        for prediction in dictionary:
            print(prediction["label"])


    