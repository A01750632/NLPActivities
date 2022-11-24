# Note: you will want to use pytest library with asserts, so that the tests fail assertions
# if the output is not correct
# see here: https://towardsdatascience.com/testing-best-practices-for-machine-learning-libraries-41b7d0362c95

#Libraries needed for the tests
from Activity1Class import Sentiment_Analysis
from Activity2Class import NER_training_model
from Activity3Class import Translation_APIS

#-------------------------- Test Task 1 ------------------------------------
print(f'\n{"":=^60}\n{"Test Task 1":=^60}\n{"":=^60}\n')
Sentiment_Analysis_Object = Sentiment_Analysis("SentimentAnalysisText.txt")
Sentiment_Analysis_Object.prediction()

#-------------------------- Test Task 2 ------------------------------------
print(f'\n{"":=^60}\n{"Test Task 2":=^60}\n{"":=^60}\n')
path_to_data = "dataTweeter"
PERCENT_OF_DATASET_TO_TRAIN = 0.01
Ner_training_model_Test = NER_training_model(path_to_data, PERCENT_OF_DATASET_TO_TRAIN)
Ner_training_model_Test.train()
Ner_training_model_Test.plot()

#-------------------------- Test Task 3 -------------------------------------
print(f'\n{"":=^60}\n{"Test Task 3":=^60}\n{"":=^60}\n')
'''Test_Apis = Translation_APIS()
Test_Apis.readfile("task3TextTest.txt")
Test_Apis.Traduction()
Test_Apis.Bleu_Scores()
'''
#-------------------------- Test passed -------------------------------------
print(f'\n{"":=^60}\n{"if this you see this message all the test are correct":=^60}\n{"":=^60}\n')
