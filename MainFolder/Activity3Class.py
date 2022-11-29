'''Author: Liam Garay Monroy A01750632
Activity3Class for task3'''

import requests
import json
import numpy as np
from dotenv import load_dotenv
import os
from nltk.translate.bleu_score import sentence_bleu
dirname = os.path.dirname(__file__)
FILENAME = os.path.join(dirname, '../europarl-v7.es-en.en')
FILENAME2 = os.path.join(dirname, '../europarl-v7.es-en.es')
load_dotenv()

key = os.getenv("KEY_SECOND_API")

class Translation_APIS:
    def __init__(self):
        print(f'\n{"":=^60}\n{"Third Task":=^60}\n{"":=^60}\n')
        self.argos_url = 'https://translate.argosopentech.com/translate'
        self.rapidapi_url = 'https://text-translator2.p.rapidapi.com/translate'
        self.Traduction1List = []
        self.Traduction2List = []

    def readfile(self,filename = FILENAME):
        textOriginal = open(filename, "r",encoding="utf-8")
        self.linesOriginal = textOriginal.readlines()
        self.FileLen = len(self.linesOriginal)
        textOriginal.close()

    def Traduction(self,LanguageSource = 'en',LanguageTarget = 'es'):
        for count, lineOriginal in enumerate(self.linesOriginal):
            request = {'q': lineOriginal, 
                'source': LanguageSource,
                'target': LanguageTarget}
            response1 = requests.request("POST", self.argos_url, data=json.dumps(request), headers={"content-type": "application/json"})
            translation = json.loads(response1.text)
            textTranslated1 = translation["translatedText"]
            self.Traduction1List.append(textTranslated1)
            print(f'First API Translation: {textTranslated1[:-1]}')
            payload = f"source_language={LanguageSource}&target_language={LanguageTarget}&text={lineOriginal}"
            headers = {
                "content-type": "application/x-www-form-urlencoded",
                "X-RapidAPI-Key": key,
                "X-RapidAPI-Host": "text-translator2.p.rapidapi.com"
            }
            
            response2 = requests.request("POST", self.rapidapi_url, data=payload, headers=headers)
            translation2 = json.loads(response2.text)   
            print(translation2)
            textTranslated2 = translation2["data"]["translatedText"]
            self.Traduction2List.append(textTranslated2)
            print(f'Second API Translation: {textTranslated2}')
            print(f"Translating lines: line {count} of {self.FileLen}")
            count += 1

    def Bleu_Scores(self,filename2 = FILENAME2):
        ref = []
        firstAPIscores = []
        SecondAPIscores = []
        textLanguage = open(filename2, "r",encoding="utf-8")
        linesLanguage = textLanguage.readlines()
        for lineLanguage in linesLanguage:
            firstAPIscores.append(sentence_bleu(self.Traduction1List,lineLanguage))
            SecondAPIscores.append(sentence_bleu(self.Traduction2List,lineLanguage))

        #Average score
        AverageFirstAPI = np.mean(firstAPIscores)
        AverageSecondAPI = np.mean(SecondAPIscores)

        print(f'\n{"":=^60}\n{"Average of Bleu Scores":=^60}\n{"":=^60}\n')
        print(f'\n{"":=^60}\n"First API Scores" {AverageFirstAPI}\n{"":=^60}\n')
        print(f'{"":=^60}\n"Second API Scores" {AverageSecondAPI}\n{"":=^60}\n')
        return AverageFirstAPI, AverageSecondAPI