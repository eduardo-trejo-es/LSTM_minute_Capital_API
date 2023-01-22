from Trainer_Predicting_Esamble import Model_Trainer
from Forcaster_Model import Forcast_Data

trainer_model = Model_Trainer()
#forcaster = Forcast_Data("DataMinuteTwttr.csv")

Model_Path="FFT_added_LSTM_BothInOut/ModelGen/Model/Model_LSTM_31_FFT"
Data_CSV="FFT_added_LSTM_BothInOut/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_FFT_10_50_100.csv"

training_result=trainer_model.to_train(Model_Path,Data_CSV,85)

#Result=forcaster.ToForcast(14,"Model/Model_Twttr_0_0_Minutes")
