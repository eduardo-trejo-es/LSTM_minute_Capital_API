from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Activation

modelPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/High_Low_Close/Model/Models_fewColums/Model_LSTM_TutorialDigitalSreeni_26Aug2023_30Percent.keras"

model = keras.models.load_model(modelPath)

print(model.summary())