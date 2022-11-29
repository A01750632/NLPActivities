# Import code from the file of activity 1
import unittest
from Activity1Class import Sentiment_Analysis

class Testing(unittest.TestCase):
    def test_predictions(self):
        print(f'\n{"":=^60}\n{"Test Task 1":=^60}\n{"":=^60}\n')
        Sentiment_Analysis_Object = Sentiment_Analysis()
        prediction = Sentiment_Analysis_Object.prediction_test()

        self.assertEqual([{'label': 'NEGATIVE', 'score': 0.9994410872459412}, {'label': 'POSITIVE', 'score': 0.9996896982192993}, {'label': 'NEGATIVE', 'score': 0.9570616483688354}, {'label': 'NEGATIVE', 'score': 0.9974650144577026}, {'label': 'NEGATIVE', 'score': 0.9995409250259399}, {'label': 'POSITIVE', 'score': 0.9988177418708801}, {'label': 'NEGATIVE', 'score': 0.9995119571685791}, {'label': 'POSITIVE', 'score': 0.9997276663780212}, {'label': 'NEGATIVE', 'score': 0.9886009097099304}, {'label': 'POSITIVE', 'score': 0.9997406601905823}, {'label': 'POSITIVE', 'score': 0.999489426612854}, {'label': 'POSITIVE', 'score': 0.9998617172241211}, {'label': 'NEGATIVE', 'score': 0.9942620992660522}, {'label': 'NEGATIVE', 'score': 0.9994024038314819}, {'label': 'POSITIVE', 'score': 0.9995092153549194}, {'label': 'POSITIVE', 'score': 0.9986543655395508}, {'label': 'POSITIVE', 'score': 0.9994551539421082}, {'label': 'POSITIVE', 'score': 0.9960886240005493}, {'label': 'POSITIVE', 'score': 0.9992528557777405}, {'label': 'NEGATIVE', 'score': 0.9998175501823425}],prediction,"Not equal")
        print(f'\n{"":=^60}\n{"Test passed task 1":=^60}\n{"":=^60}\n')

if __name__ == "__main__":
    unittest.main()