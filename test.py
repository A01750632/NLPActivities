#Libraries needed for the tests
from Activity1Class.Activity1Class import Sentiment_Analysis
from Activity2Class.Activity2Class import NER_training_model
from Activity3Class.Activity3Class import Translation_APIS

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