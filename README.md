# PathSynergy
tips
Step1:create dataset for model_a,model_b
model_a:create_data.py could deal with the file like data/6271_drug_synergy_GAT.csv
model_b:get_matrix_cell.py could create matrix data.

Step2:train model
train_model_a.py is a train model by model_a\\
train_model_b.py is a train model by model_b\\
ensemble_train.py is a train model by PathSynergy
predict_independent.py or predict_sorafenib is a validation model for predicting novel drug pairs 

Additional data is available on Google Cloud Drive
https://drive.google.com/file/d/1-ndMYuyJugvAqechWNlYuehIvzCFj6yJ/view?usp=drive_link


