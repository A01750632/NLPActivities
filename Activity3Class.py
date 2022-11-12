'''Author: Liam Garay Monroy A01750632
Activity3Class for task3'''

import requests
import json
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("KEY_SECOND_API")

class Translation_APIS:
    def __init__(self):
        print(f'\n{"":=^60}\n{"Third Task":=^60}\n{"":=^60}\n')
        self.url = 'https://translate.argosopentech.com/translate'
        self.url2 = 'https://text-translator2.p.rapidapi.com/translate'
        self.Traduction1List = []
        self.Traduction2List = []

    def readfile(self,Filename = 'europarl-v7.es-en.en'):
        textOriginal = open(Filename, "r",encoding="utf-8")
        self.linesOriginal = textOriginal.readlines()
        self.FileLen = len(self.linesOriginal)

    def Traduction(self,LenguageSource = 'en',LenguageTarget = 'es'):
        count = 1
        for lineOriginal in self.linesOriginal:
            myobj = {'q': lineOriginal,
                'source': LenguageSource,
                'target': LenguageTarget}
            response1 = requests.request("POST", self.url, data=json.dumps(myobj), headers={"content-type": "application/json"})
            translation = json.loads(response1.text)
            textTranslated1 = translation["translatedText"]
            self.Traduction1List.append(textTranslated1)
            print(f'First API Translation: {textTranslated1[:-1]}')
            payload = f"source_language={LenguageSource}&target_language={LenguageTarget}&text={lineOriginal}"
            headers = {
                "content-type": "application/x-www-form-urlencoded",
                "X-RapidAPI-Key": key,
                "X-RapidAPI-Host": "text-translator2.p.rapidapi.com"
            }
            
            response2 = requests.request("POST", self.url2, data=payload, headers=headers)
            translation2 = json.loads(response2.text)   
            print(translation2)
            textTranslated2 = translation2["data"]["translatedText"]
            self.Traduction2List.append(textTranslated2)
            print(f'Second API Translation: {textTranslated2}')
            print(f"Translating lines: line {count} of {self.FileLen}")
            count += 1

    def Bleu_Scores(self):
        ref = []
        firstAPIscores = []
        SecondAPIscores = []
        textLenguage = open("europarl-v7.es-en.es", "r",encoding="utf-8")
        linesLenguage = textLenguage.readlines()
        from nltk.translate.bleu_score import sentence_bleu
        for lineLenguage in linesLenguage:
            firstAPIscores.append(sentence_bleu(self.Traduction1List,lineLenguage))
            SecondAPIscores.append(sentence_bleu(self.Traduction1List,lineLenguage))

        #Average score
        AverageFistAPI = np.mean(firstAPIscores)
        AverageSecondAPI = np.mean(SecondAPIscores)

        print(f'\n{"":=^60}\n{"Average of Bleu Scores":=^60}\n{"":=^60}\n')
        print(f'\n{"":=^60}\n"First API Scores" {AverageFistAPI}\n{"":=^60}\n')
        print(f'{"":=^60}\n"Second API Scores" {AverageSecondAPI}\n{"":=^60}\n')