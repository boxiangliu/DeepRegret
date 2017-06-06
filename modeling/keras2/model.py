from keras.models import Model
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten, Dropout, Concatenate
from keras.regularizers import l1
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import plot_model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import sys
sys.path.append('/srv/scratch/bliu2/deepregret/scripts/modeling/keras/')
import input_data

class Prediction(Callback):
	def __init__(self,log_dir):
		self.log_dir = log_dir

	def on_train_begin(self, logs={}):
		self.pred = []

	def on_epoch_end(self, epoch, logs={}):
		y_pred = self.model.predict({'seq_input':self.validation_data[1],'reg_input':self.validation_data[0]})
		np.savetxt('%s/prediction.epoch%s.txt'%(self.log_dir,epoch), y_pred)
		self.pred.append(y_pred)
 		return



num_reg=472
seq_length=1000
L1=0.01

print('INFO - %s'%('building regression model.'))
units=512
reg_input=Input(shape=(num_reg,),dtype='float32',name='reg_input')
x=Dense(units,activation='relu')(reg_input)
x=Dense(units,activation='relu')(x)
reg_output=Dense(units,activation='relu')(x)

print('INFO - %s'%('building sequence model.'))
filters=[32,256]
kernel_width=[100,15]
seq_input=Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
x=Conv2D(filters=filters[0],kernel_size=(4,kernel_width[0]),strides=4,padding='valid',activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(seq_input)
x=Conv2D(filters=filters[1],kernel_size=(1,kernel_width[1]),strides=4,padding='valid',activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(x)
seq_output=Flatten()(x)

print('INFO - %s'%('building concatenate model.'))
units=512
x=concatenate([reg_output,seq_output])
x=Dense(units,activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(x)
x=Dense(units,activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(x)
x=Dropout(0.5,seed=42)(x)
rgs_output=Dense(1,activation='linear',name='rgs_output')(x)


model=Model(inputs=[reg_input,seq_input],outputs=[rgs_output])
sgd = SGD(lr=0.01)
model.compile(loss={'rgs_output':'mean_squared_error'},optimizer=sgd)
plot_model(model, show_shapes=True,to_file='../figures/keras/model.eps')

print('INFO - %s'%('loading data.'))
train,val,test=input_data.read_data_sets()


print('INFO - %s'%('training model.'))
early_stopping=EarlyStopping(monitor='val_loss',patience=2)
checkpoint=ModelCheckpoint(filepath="../logs/keras/model.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss')
prediction=Prediction(log_dir='../logs/keras')
model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'rgs_output':train['expr']},validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'rgs_output':val['expr']}),epochs=30,batch_size=100,callbacks=[early_stopping,prediction],verbose=1)


pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=1000,verbose=1)
plt.scatter(pred,test['expr'])
plt.savefig("%s/pred_vs_obs.png"%('../figures/keras'))
output=np.column_stack((test['expr'], pred[:,0]))
np.savetxt("%s/prediction.txt"%('../processed_data/keras/'), output,delimiter='\t')
