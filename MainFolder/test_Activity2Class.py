import unittest
from Activity2Class import NER_training_model

class Testing(unittest.TestCase):
    def test_predictions(self):
        print(f'\n{"":=^60}\n{"Test Task 2":=^60}\n{"":=^60}\n')
        Ner_training_model_Object = NER_training_model()
        #the learning_rate, max_epochs, mini_batch_size
        # Ner_training_model_Object.train(self,learning_rate = 0.02, max_epochs = 10, mini_batch_size = 32)
        corpus = Ner_training_model_Object.downsample_test()

        self.assertEqual("Corpus: 239 train + 100 dev + 385 test sentences",corpus,"Not equal")
        print(f'\n{"":=^60}\n{"Test passed task 2":=^60}\n{"":=^60}\n')


if __name__ == "__main__":
    unittest.main()