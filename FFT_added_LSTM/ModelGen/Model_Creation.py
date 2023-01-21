###   Model creation

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Activation



keras.backend.clear_session()  # Reseteo sencillo

#---------Layes are created

n_future = 1   # Number of units(day, min, hour, etc..) we want to look into the future based on the past days.
n_past =120
Columns_N=31

inputs=keras.Input(shape=(n_past,Columns_N))


LSTM_Layer1=keras.layers.LSTM(n_past, input_shape=(n_past,Columns_N), return_sequences=True, activation='sigmoid')(inputs)

Dropout_layer2=keras.layers.Dropout(0.6)(LSTM_Layer1)# modify
#x=Dropout_layer1=keras.layers.Dropout(0.2)(x)
LSTM_Layer2=keras.layers.LSTM(90, return_sequences=False)(Dropout_layer2)

Dropout_layer3=keras.layers.Dropout(0.6)(LSTM_Layer2)# modify


#---------------------------Outputs
dense_0=keras.layers.Dense(1)(Dropout_layer3)
dense_1=keras.layers.Dense(1)(Dropout_layer3)
dense_2=keras.layers.Dense(1)(Dropout_layer3)
dense_3=keras.layers.Dense(1)(Dropout_layer3)
dense_4=keras.layers.Dense(1)(Dropout_layer3)
dense_5=keras.layers.Dense(1)(Dropout_layer3)
dense_6=keras.layers.Dense(1)(Dropout_layer3)
dense_7=keras.layers.Dense(1)(Dropout_layer3)
dense_8=keras.layers.Dense(1)(Dropout_layer3)
dense_9=keras.layers.Dense(1)(Dropout_layer3)
dense_10=keras.layers.Dense(1)(Dropout_layer3)
dense_11=keras.layers.Dense(1)(Dropout_layer3)
dense_12=keras.layers.Dense(1)(Dropout_layer3)
dense_13=keras.layers.Dense(1)(Dropout_layer3)
dense_14=keras.layers.Dense(1)(Dropout_layer3)
dense_15=keras.layers.Dense(1)(Dropout_layer3)
dense_16=keras.layers.Dense(1)(Dropout_layer3)
dense_17=keras.layers.Dense(1)(Dropout_layer3)
dense_18=keras.layers.Dense(1)(Dropout_layer3)
dense_19=keras.layers.Dense(1)(Dropout_layer3)
dense_20=keras.layers.Dense(1)(Dropout_layer3)
dense_21=keras.layers.Dense(1)(Dropout_layer3)
dense_22=keras.layers.Dense(1)(Dropout_layer3)
dense_23=keras.layers.Dense(1)(Dropout_layer3)
dense_24=keras.layers.Dense(1)(Dropout_layer3)
dense_25=keras.layers.Dense(1)(Dropout_layer3)
dense_26=keras.layers.Dense(1)(Dropout_layer3)
dense_27=keras.layers.Dense(1)(Dropout_layer3)
dense_28=keras.layers.Dense(1)(Dropout_layer3)
dense_29=keras.layers.Dense(1)(Dropout_layer3)
dense_30=keras.layers.Dense(1)(Dropout_layer3)

#-------Layers outputs are linked

outputs_0=dense_0
outputs_1=dense_1
outputs_2=dense_2
outputs_3=dense_3
outputs_4=dense_4
outputs_5=dense_5
outputs_6=dense_6
outputs_7=dense_7
outputs_8=dense_8
outputs_9=dense_9
outputs_10=dense_10
outputs_11=dense_11
outputs_12=dense_12
outputs_13=dense_13
outputs_14=dense_14
outputs_15=dense_15
outputs_16=dense_16
outputs_17=dense_17
outputs_18=dense_18
outputs_19=dense_19
outputs_20=dense_20
outputs_21=dense_21
outputs_22=dense_22
outputs_23=dense_23
outputs_24=dense_24
outputs_25=dense_25
outputs_26=dense_26
outputs_27=dense_27
outputs_28=dense_28
outputs_29=dense_29
outputs_30=dense_30


#-----The model it's created
outputArray=[outputs_0,outputs_1,outputs_2,outputs_3,outputs_4,
             outputs_5,outputs_6,outputs_7,outputs_8,outputs_9,
             outputs_10,outputs_11,outputs_12,outputs_13,outputs_14,
             outputs_15,outputs_16,outputs_17,outputs_18,outputs_19,
             outputs_20,outputs_21,outputs_22,outputs_23,outputs_24,
             outputs_25,outputs_26,outputs_27,outputs_28,outputs_29,
             outputs_30]

model=keras.Model(inputs=inputs, outputs=outputArray, name='Prices_Forcasting_LSTM_FFT')
#model=keras.Model(inputs=[inputs,None], outputs=[outputs,outputs2,outputs3,outputs4,outputs5,outputs6], name='Prices_Prediction')

#keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


#------------------- Loss and optimizer ----------------------------------------
#got to ensure MeanAbsoluteError it's the good one for our data
loss_0 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_1 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_2 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_3 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_4 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_5 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_6 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_7 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_8 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_9 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_10 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_11 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_12 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_13 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_14 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_15 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_16 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_17 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_18 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_19 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_20 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_21 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_22 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_23 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_24 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_25 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_26 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_27 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_28 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_29 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss_30 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
#optim=keras.optimizers.Adam(1e-3)
optim=keras.optimizers.Adam(1e-3)
metrics=["accuracy"]

losses={
    "dense_0": loss_0,
    "dense_1": loss_1,
    "dense_2": loss_2,
    "dense_3": loss_3,
    "dense_4": loss_4,
    "dense_5": loss_5,
    "dense_6": loss_6,
    "dense_7": loss_7,
    "dense_8": loss_8,
    "dense_9": loss_9,
    "dense_10": loss_10,
    "dense_11": loss_11,
    "dense_12": loss_12,
    "dense_13": loss_13,
    "dense_14": loss_14,
    "dense_15": loss_15,
    "dense_16": loss_16,
    "dense_17": loss_17,
    "dense_18": loss_18,
    "dense_19": loss_19,
    "dense_20": loss_20,
    "dense_21": loss_21,
    "dense_22": loss_22,
    "dense_23": loss_23,
    "dense_24": loss_24,
    "dense_25": loss_25,
    "dense_26": loss_26,
    "dense_27": loss_27,
    "dense_28": loss_28,
    "dense_29": loss_29,
    "dense_30": loss_30
}

model.compile(loss=losses, optimizer=optim, metrics=metrics)

print(model.summary())

#tf.keras.utils.plot_model(model, "FFT_added_LSTM/ModelGen/Model/Model_LSTM_31_FFT.png", show_shapes=True)

model.save("FFT_added_LSTM/ModelGen/Model/Model_LSTM_31_FFT")