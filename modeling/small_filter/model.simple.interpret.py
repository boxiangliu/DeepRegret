import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import sys
sys.path.append('utils/')
import input_data_config as input_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

# Read data:
expr_file='../data/complete_dataset.txt'
reg_names_file='../data/reg_names_R.txt'

expr_data = pd.read_table(expr_file).fillna(0).drop("NAME", axis=1).drop("GWEIGHT", axis=1)
reg_names = pd.read_table(reg_names_file, names = ["UID"])
reg_data = pd.merge(reg_names, expr_data, on="UID", how="inner").drop("UID", axis=1)
reg_std=reg_data.std(axis=1)
reg_top=reg_data.iloc[np.where(reg_std>=1.25)]



# Load model:
model_fn='../logs/small_filter/simple.classification/model.34-0.5297.hdf5'
model=load_model(model_fn)
train,val,test=input_data.read_data_sets(train_pct=80,val_pct=10,test_pct=10)


ko=test['reg']
ko[:,454]=0
pred_ko=model.predict({'seq_input':test['seq'],'reg_input':ko},batch_size=100,verbose=1)
pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
pred_ko_class=pred_ko.argmax(axis=1)
pred_class=pred.argmax(axis=1)


pred_ko_norm=pred_ko/pred_ko.sum(axis=1)[:,None]
pred_norm=pred/pred.sum(axis=1)[:,None]
sum(pred_ko_norm==pred_norm)

confusion_matrix(test['class'][diff_idx],pred_class[diff_idx])
log_loss(pred.argmax(axis=1), pred_ko)



