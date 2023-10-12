import sys
sys.path.append("FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/Pakages/DataSetgenPacks")

# import all classes
from Retriver_and_Processor_Dataset import DatasetGenerator

dataSet_Gen = DatasetGenerator()

dateStart='2001-01-01'
dateEnd= '2023-03-02'


#######______________ CRUDE_Oil 'CL=F' _____________  ################
itemName = 'CL=F'
OneColum=False

if OneColum:
    Original_Path_Retiving="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_Data.csv"
    Onlyonecolumn="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_Data_onlyClose.csv"
    LastOnetwice="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_Data_LastOneTwice.csv"
    DirectionPrice="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_Data_DirePrice.csv"
    DayNumAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_Dataand_DayNum.csv"
    MonthAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_Data_And_month.csv"
    yearAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_Data_And_year.csv"
    FFTAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_CloseFFT_150_5Backdys.csv"
    LastPopcolum="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/OnlyCloseColum/CRUDE_OIL_Close_lastPopcolum.csv"
else:
    Original_Path_Retiving="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_Data.csv"
    Onlyonecolumn="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_Data_onlyClose.csv"
    LastOnetwice="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_Data_LastOneTwice.csv"
    DirectionPrice="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_Data_DirePrice.csv"
    DayNumAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_Dataand_DayNum.csv"
    MonthAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_Data_And_month.csv"
    yearAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_Data_And_year.csv"
    FFTAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_CloseFFT_2400_5Backdys.csv"
    LastPopcolum="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/High_Low_Close/CRUDE_OIL_Close_lastPopcolum.csv"
    Add_To_old=False


dataSet_Gen.RetivingDataPrices_Yahoo(itemName,dateStart, dateEnd,Original_Path_Retiving,Original_Path_Retiving,Add_To_old)
#dataSet_Gen.UpdateToday(itemName,Original_Path_Retiving)

#columns to pop up
#dataSet_Gen.AddRepeatedLastOne(Original_Path_Retiving, LastOnetwice)
if OneColum:
    #Columns to remove
    columns=['Open','High','Low','Volume']
else:
    #Columns to remove
    columns=['Open','Volume']

dataSet_Gen.PopListdf(columns,Original_Path_Retiving,Onlyonecolumn)


#dataSet_Gen.AddColumnPRCNTG(Original_Path_Retiving,PRCNTGAddedPath)
#if inversed:
#    dataSet_Gen.AddColumnInverseDirePrice(Original_Path_Retiving,DirectionPrice)
#else: 
#    dataSet_Gen.AddColumnDirePrice(Original_Path_Retiving,DirectionPrice)

dataSet_Gen.AddColumnWeekDay(Onlyonecolumn, DayNumAddedPath,False)

dataSet_Gen.AddColumnMothandDay(DayNumAddedPath, MonthAddedPath,False)

dataSet_Gen.AddColumnYear(MonthAddedPath,yearAddedPath)
#
#Generate new FFT columns done :)


backdaysToconsider=6
inicialPath=yearAddedPath
FFTNew_FileData=FFTAddedPath
Column=["Close",'High','Low']
frec=[160]

dataSet_Gen.getTheLastFFTValue(backdaysToconsider,frec,Column,inicialPath, FFTNew_FileData)   


columns=['High','Low']

dataSet_Gen.PopListdf(columns,FFTNew_FileData,LastPopcolum)
