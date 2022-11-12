from Activity1Class import Sentiment_Analysis
from Activity2Class import NER_training_model
from Activity3Class import Translation_APIS

#filename_Activity1 = 'tiny_movie_reviews_dataset.txt'
#Sentiment_Analysis_Object = Sentiment_Analysis(filename_Activity1)
Sentiment_Analysis_Object = Sentiment_Analysis()
Sentiment_Analysis_Object.prediction()


#path_to_data = "dataTweeter"
#PERCENT_OF_DATASET_TO_TRAIN = 0.1
#Ner_training_model_Object = NER_training_model(path_to_data, PERCENT_OF_DATASET_TO_TRAIN)
Ner_training_model_Object = NER_training_model()
Ner_training_model_Object.train()
Ner_training_model_Object.plot()

'''Translation_APIS_object = Translation_APIS()
Translation_APIS_object.Traduction()
Translation_APIS_object.Bleu_Scores()'''