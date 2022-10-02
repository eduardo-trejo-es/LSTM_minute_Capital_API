from Trainer_Predicting_Esamble6Colum import Model_Trainer
from Forcaster_Model_6Colum import Forcast_Data

trainer_model = Model_Trainer()
forcaster = Forcast_Data("/Users/eduardo/Desktop/LSTM_Capital_API_220922/with6ColumsPerHour/TimeColumHOUR.csv")

#training_result=trainer_model.to_tain("/Users/eduardo/Desktop/LSTM_Capital_API_220922/with6ColumsPerHour/Models/Model_Twttr_6colum_hour_0","/Users/eduardo/Desktop/LSTM_Capital_API_220922/with6ColumsPerHour/TimeColumHOUR.csv",70)

#Result=forcaster.ToForcast(25,"/Users/eduardo/Desktop/LSTM_Capital_API_220922/with6ColumsPerHour/Models/Model_Twttr_6colum_hour_0","2022-09-29T05:00:00")


Result=forcaster.to_forcast_close_true_and_forcasted(1,"/Users/eduardo/Desktop/LSTM_Capital_API_220922/with6ColumsPerHour/Models/Model_Twttr_6colum_hour_0","2022-02-24T07:00:00")



print("done....")
