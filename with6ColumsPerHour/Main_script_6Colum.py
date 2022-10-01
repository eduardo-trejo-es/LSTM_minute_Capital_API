from Trainer_Predicting_Esamble6Colum import Model_Trainer
from Forcaster_Model_6Colum import Forcast_Data

trainer_model = Model_Trainer()
forcaster = Forcast_Data("/Users/eduardo/Desktop/LSTM_minute_Capital_API/with6Colums/DataMinuteTwttrTimeColum.csv")

#training_result=trainer_model.to_tain("/Users/eduardo/Desktop/LSTM_minute_Capital_API/with6Colums/Models/Model_Twttr_6colum_0_0","/Users/eduardo/Desktop/LSTM_minute_Capital_API/with6Colums/DataMinuteTwttrTimeColum.csv",80)

Result=forcaster.ToForcast(50,"/Users/eduardo/Desktop/LSTM_minute_Capital_API/with6Colums/Models/Model_Twttr_6colum_0_0","2022-08-18T09:00:00")

print("done....")
