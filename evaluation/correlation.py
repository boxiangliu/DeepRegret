from keras.models import load_model
import sys
import os
sys.path.append('utils/')
import input_data_config as input_data
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from scipy.stats import pearsonr
import numpy as np


# Variables: 
fig_dir='../figures/evaluation/correlation/'
if (not os.path.exists(fig_dir)): os.makedirs(fig_dir)

#------- Tensor product network --------
model_fn='../logs/single_layer/model/model.44-0.2131.hdf5'
model=load_model(model_fn)
train,val,test=input_data.read_data_sets(train_pct=80,val_pct=10,test_pct=10)
pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
plt.ion()
plt.scatter(test['expr'],pred[:,0])


lr=linear_model.LinearRegression()
fit=lr.fit(test['expr'][:,None],pred[:,0])
cor=pearsonr(test['expr'],pred[:,0])[0]
plt.text(-6, 4, 'y = %.02f + %.02f x\nPearson r = %.02f'%(fit.intercept_, fit.coef_,cor),fontsize=15)
plt.xlabel('Observation')
plt.ylabel('Prediction')
plt.savefig('%s/correlation_tensor_product.pdf'%fig_dir)


#---------- Concatenation network -----------
model_fn='../logs/small_filter/simple/model.53-0.2152.hdf5'
model=load_model(model_fn)
pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)

plt.scatter(test['expr'],pred[:,0])
lr=linear_model.LinearRegression()
fit=lr.fit(test['expr'][:,None],pred[:,0])
cor=pearsonr(test['expr'],pred[:,0])[0]
plt.text(-6, 4, 'y = %.02f + %.02f x\nPearson r = %.02f'%(fit.intercept_, fit.coef_,cor),fontsize=15)
plt.xlabel('Observation')
plt.ylabel('Prediction')
plt.savefig('%s/correlation_concatenation.pdf'%fig_dir)

