import requests
import json
import numpy as np


url = 'https://translate.argosopentech.com/translate'
url2 = 'https://deepl-translator.p.rapidapi.com/translate'

Traduction1List = []
Traduction2List = []

textOriginal = open("europarl-v7.es-en.en", "r",encoding="utf-8")
linesOriginal = textOriginal.readlines()
print(linesOriginal)
FileLen = len(linesOriginal)
contador = 1
for lineOriginal in linesOriginal:
    myobj = {'q': lineOriginal,
		'source': "en",
		'target': "es"}
    response1 = requests.request("POST", url, data=json.dumps(myobj), headers={"content-type": "application/json"})
    translation = json.loads(response1.text)
    textTranslated1 = translation["translatedText"]
    Traduction1List.append(textTranslated1)
    print(f'First API Translation: {textTranslated1[:-1]}')
    payload = {
      "text": lineOriginal,
      "source": "EN",
      "target": "ES"
    }
    headers = {
      "content-type": "application/json",
      "X-RapidAPI-Key": "key",
      "X-RapidAPI-Host": "deepl-translator.p.rapidapi.com"
    }

    response2 = requests.request("POST", url2, json=payload, headers=headers)
    translation2 = json.loads(response2.text)
    print(translation2)
    textTranslated2 = translation2["text"]
    Traduction2List.append(textTranslated2)
    print(f'Second API Translation: {textTranslated2}')
    print(f"Translating lines: line {contador} of {FileLen}")
    contador += 1


ref = []
firstAPIscores = []
SecondAPIscores = []
textLenguage = open("europarl-v7.es-en.es", "r",encoding="utf-8")
linesLenguage = textLenguage.readlines()
from nltk.translate.bleu_score import sentence_bleu
for lineLenguage in linesLenguage:
  firstAPIscores.append(sentence_bleu(Traduction1List,lineLenguage))
  SecondAPIscores.append(sentence_bleu(Traduction1List,lineLenguage))

#Average score
AverageFistAPI = np.mean(firstAPIscores)
AverageSecondAPI = np.mean(SecondAPIscores)

print(f"First API Scores {AverageFistAPI}")
print(f"First API Scores {AverageSecondAPI}")