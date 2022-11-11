import requests
import json
import numpy as np

class Translation_APIS:
    def __init__(self):
        self.url = 'https://translate.argosopentech.com/translate'
        self.url2 = 'https://deepl-translator.p.rapidapi.com/translate'
        self.Traduction1List = []
        self.Traduction2List = []

    def Traduction(self,Filename = 'europarl-v7.es-en.en',LenguageSource = 'en',LenguageTarget = 'es'):
        textOriginal = open(Filename, "r",encoding="utf-8")
        linesOriginal = textOriginal.readlines()
        FileLen = len(linesOriginal)
        contador = 1
        for lineOriginal in linesOriginal:
            myobj = {'q': lineOriginal,
                'source': LenguageSource,
                'target': LenguageTarget}
            response1 = requests.request("POST", self.url, data=json.dumps(myobj), headers={"content-type": "application/json"})
            translation = json.loads(response1.text)
            textTranslated1 = translation["translatedText"]
            self.Traduction1List.append(textTranslated1)
            print(f'First API Translation: {textTranslated1[:-1]}')
            payload = {
            "text": lineOriginal,
            "source": LenguageSource.upper(),
            "target": LenguageTarget.upper()
            }
            headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": "7d2197faa6msh22bfea21ec22ee0p1525d0jsn475c30e703e9",
            "X-RapidAPI-Host": "deepl-translator.p.rapidapi.com"
            }

            response2 = requests.request("POST", self.url2, json=payload, headers=headers)
            translation2 = json.loads(response2.text)
            print(translation2)
            textTranslated2 = translation2["text"]
            self.Traduction2List.append(textTranslated2)
            print(f'Second API Translation: {textTranslated2}')
            print(f"Translating lines: line {contador} of {FileLen}")
            contador += 1

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

        print(f"First API Scores {AverageFistAPI}")
        print(f"First API Scores {AverageSecondAPI}")