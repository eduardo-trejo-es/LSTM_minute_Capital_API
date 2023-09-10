import sys
sys.path.append("FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/Pakages/DataSetgenPacks")

# import all classes
from Retriver_and_Processor_Dataset import DatasetGenerator

dataSet_Gen = DatasetGenerator()

#dataSet_Gen.RetivingDataPrices_Yahoo(itemName,dateStart, dateEnd,Original_Path_Retiving,Original_Path_Retiving)
#dataSet_Gen.UpdateToday(itemName,Original_Path_Retiving)