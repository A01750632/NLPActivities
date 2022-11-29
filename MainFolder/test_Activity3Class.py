import unittest
from Activity3Class import Translation_APIS
import os

class Testing(unittest.TestCase):
    def test_predictions(self):
        print(f'\n{"":=^60}\n{"Test Task 3":=^60}\n{"":=^60}\n')
        Translation_APIS_object = Translation_APIS()
        dirname = os.path.dirname(__file__)
        FILENAME_TEST = os.path.join(dirname, '../europarl-v7.es-en-copy.en')
        FILENAME2_TEST = os.path.join(dirname, '../europarl-v7.es-en-copy.es')
        Translation_APIS_object.readfile(FILENAME_TEST)
        #also the traduction target and source lenguage can be modified
        #Translation_APIS_object.Traduction(self,LenguageSource = 'en',LenguageTarget = 'es'):
        Translation_APIS_object.Traduction()
        first_Api_scores_trest, second_Api_scores_trest = Translation_APIS_object.Bleu_Scores(FILENAME2_TEST)

        self.assertEqual(0.7365155184579807,first_Api_scores_trest,"Not equal")
        self.assertEqual(0.6829635801128902,second_Api_scores_trest,"Not equal")
        print(f'\n{"":=^60}\n{"Test passed task 3":=^60}\n{"":=^60}\n')


if __name__ == "__main__":
    unittest.main()