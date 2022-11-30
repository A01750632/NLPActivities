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
        self.traduction_1List = []
        self.traduction_2List = []

    def readfile(self,filename = FILENAME):
        textOriginal = open(filename, "r",encoding="utf-8")
        self.lines_original = textOriginal.readlines()
        self.FileLen = len(self.lines_original)
        textOriginal.close()

    def Traduction(self,language_source = 'en',language_target = 'es'):
        for count, line_original in enumerate(self.lines_original):
            request = {'q': line_original, 
                'source': language_source,
                'target': language_target}
            response1 = requests.request("POST", self.argos_url, data=json.dumps(request), headers={"content-type": "application/json"})
            translation = json.loads(response1.text)
            text_translated1 = translation["translatedText"]
            self.traduction_1List.append(text_translated1)
            print(f'First API Translation: {text_translated1[:-1]}')
            payload = f"source_language={language_source}&target_language={language_target}&text={line_original}"
            headers = {
                "content-type": "application/x-www-form-urlencoded",
                "X-RapidAPI-Key": key,
                "X-RapidAPI-Host": "text-translator2.p.rapidapi.com"
            }
            
            response2 = requests.request("POST", self.rapidapi_url, data=payload, headers=headers)
            translation2 = json.loads(response2.text)   
            print(translation2)
            textTranslated2 = translation2["data"]["translatedText"]
            self.traduction_2List.append(textTranslated2)
            print(f'Second API Translation: {textTranslated2}')
            print(f"Translating lines: line {count} of {self.FileLen}")
            count += 1

    def Bleu_Scores(self,filename2 = FILENAME2):
        first_api_scores = []
        second_api_scores = []
        text_language = open(filename2, "r",encoding="utf-8")
        lines_language = text_language.readlines()
        for lineLanguage in lines_language:
            first_api_scores.append(sentence_bleu(self.traduction_1List,lineLanguage))
            second_api_scores.append(sentence_bleu(self.traduction_2List,lineLanguage))

        #Average score
        average_first_api = np.mean(first_api_scores)
        average_second_api = np.mean(second_api_scores)

        print(f'\n{"":=^60}\n{"Average of Bleu Scores":=^60}\n{"":=^60}\n')
        print(f'\n{"":=^60}\n"First API Scores" {average_first_api}\n{"":=^60}\n')
        print(f'{"":=^60}\n"Second API Scores" {average_second_api}\n{"":=^60}\n')
        return average_first_api, average_second_api