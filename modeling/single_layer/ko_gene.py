from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge, Activation, MaxPooling2D
import numpy as np
import pandas as pd
import sys,os
sys.path.append('/srv/scratch/bliu2/deepregret/scripts/modeling/single_layer/')
import input_data
from input_data import one_hot
from subprocess import call
import matplotlib.pyplot as plt


# Functions: 






# Constants: 
num_reg=472
seq_length=1000
filters_dir='../processed_data/single_layer/interpret/filters/'
pwm_dir='../processed_data/single_layer/interpret/pwm/'
fig_dir='../figures/single_layer/interpret/filters/'
if not os.path.exists(filters_dir): os.makedirs(filters_dir)
if not os.path.exists(pwm_dir): os.makedirs(pwm_dir)
if not os.path.exists(fig_dir): os.makedirs(fig_dir)

# Get fitted model: 
fn='../logs/single_layer/model.29-0.2093.hdf5'
model=load_model(fn)
train,val,test=input_data.read_data_sets()
pred=model.predict({'seq_input':train['seq'],'reg_input':train['reg']},batch_size=100,verbose=1)
plt.ion()
plt.scatter(train['expr'],pred)


ko=np.copy(train['reg'])
ko[:,0]=0.0
pred_ko=model.predict({'seq_input':train['seq'],'reg_input':ko},batch_size=1000,verbose=1)
plt.scatter(pred, pred_ko)
diff=pred-pred_ko
plt.hist(diff)


pred_test=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
ko_test=np.copy(test['reg'])
ko_test[:,0]=0.0
pred_test_ko=model.predict({'seq_input':test['seq'],'reg_input':ko_test},batch_size=1000,verbose=1)
plt.scatter(pred_test, pred_test_ko)
plt.scatter(test['expr'], pred_test)
plt.scatter(test['expr'], pred_test_ko)
diff_test=pred_test-pred_test_ko
plt.hist(diff,bins=1000)

pred_test_ko=np.zeros(shape=test['reg'].shape)
for i in xrange(num_reg):
	ko_test=np.copy(test['reg'])
	ko_test[:,i]=0.0
	pred_test_ko[:,i]=np.ravel(model.predict({'seq_input':test['seq'],'reg_input':ko_test},batch_size=100,verbose=1))
