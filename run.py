from Activity1Class import Sentiment_Analysis
from Activity2Class import NER_training_model
from Activity3Class import Translation_APIS

#The filename can be change inside the function

#filename_Activity1 = 'tiny_movie_reviews_dataset.txt'
#Sentiment_Analysis_Object = Sentiment_Analysis(filename_Activity1)
Sentiment_Analysis_Object = Sentiment_Analysis()
Sentiment_Analysis_Object.prediction()



#the training percentage can be modified
#PERCENT_OF_DATASET_TO_TRAIN = 0.1


'''#Ner_training_model_Object = NER_training_model(PERCENT_OF_DATASET_TO_TRAIN)
#the default value of PERCENT_OF_DATASET_TO_TRAIN is 0.1
Ner_training_model_Object = NER_training_model()
#the learning_rate, max_epochs, mini_batch_size
# Ner_training_model_Object.train(self,learning_rate = 0.02, max_epochs = 10, mini_batch_size = 32)
Ner_training_model_Object.train()
Ner_training_model_Object.plot()'''

#Test faile can be modified
#Test_Apis.readfile("task3TextTest.txt")

Translation_APIS_object = Translation_APIS()
Translation_APIS_object.readfile()
#also the traduction target and source lenguage can be modified
#Translation_APIS_object.Traduction(self,LenguageSource = 'en',LenguageTarget = 'es'):
Translation_APIS_object.Traduction()
Translation_APIS_object.Bleu_Scores()