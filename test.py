from nltk.translate.bleu_score import sentence_bleu
ref = [
    'this is moonlight'.split(),
    'Look, this is moonlight'.split(),
    'moonlight it is'.split()
]
test = 'it is moonlight'.split()
print('BLEU score for test-> {}'.format(sentence_bleu(ref, test)))
 
test01 = 'it is cat and moonlight'.split()
print(test01,ref)
print('BLEU score for test01-> {}'.format(sentence_bleu(ref, test01)))