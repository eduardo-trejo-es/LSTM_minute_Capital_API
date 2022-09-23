from Trainer_Predicting_Esamble import Model_Trainer

trainer_model = Model_Trainer()

training_result=trainer_model.to_tain("Model/Model_Twttr_0_0","DataMinuteTwttr.csv",70)

print(training_result)
