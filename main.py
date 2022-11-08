from transformers import pipeline
import string
import re

def predict_sentiment(filename):
    sentiment_model = pipeline("sentiment-analysis")
    text = open(filename)
    lines = text.readlines()
    cleanLines = []    
    for line in lines:
        clean_words = line.translate(",.;@#?!&$")  
        regex_sub = re.sub(r"[,.;\"@#?!&$]+", ' ', line)  
        regex_sub = re.sub(r"\s+", ' ', regex_sub)
        cleanLines.append(regex_sub)
    dictionary = sentiment_model(cleanLines)
    for prediction in dictionary:
        print(prediction["label"])


def main():
    filename = 'tiny_movie_reviews_dataset.txt'
    predict_sentiment(filename)

main()
    
